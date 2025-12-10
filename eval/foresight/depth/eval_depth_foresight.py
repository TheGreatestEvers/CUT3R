import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm
import torch
import torch.distributed as dist

# Add repo root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append("/workspace/CUT3R/src")
sys.path.append("/workspace/CUT3R")

from eval.video_depth.tools import depth_evaluation

import eval.foresight.waymo_adapter as waymo_adapter
import eval.foresight.kitti_adapter as kitti_adapter


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_root",
        type=str,
        default="/workspace/raid/jevers/cut3r_processed_waymo/validation_full_res_depth",
        help="Dataset root. For Waymo: segment folders. For KITTI: val_selection_cropped root.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="waymo",
        choices=["waymo", "kitti"],
        help="Which dataset was used to create the predictions.",
    )

    parser.add_argument(
        "--pred_root",
        type=str,
        default="/workspace/raid/jevers/waymo_outputs/forecast_224",
        help="Root where predictions are stored (output_dir from run script).",
    )

    parser.add_argument(
        "--align",
        type=str,
        default="scale&shift",
        choices=["scale&shift", "scale", "metric"],
        help="Alignment mode: passed to depth_evaluation.",
    )

    parser.add_argument(
        "--horizon_mode",
        type=str,
        default="short",
        choices=["short", "middle", "long", "all"],
        help="Which forecast horizon to evaluate across sequences.",
    )

    parser.add_argument(
        "--max_depth",
        type=float,
        default=80.0,
        help="Max depth considered valid (None for no clipping).",
    )

    parser.add_argument(
        "--output_metrics",
        type=str,
        default=None,
        help="Optional JSON file to save aggregated metrics (single file, rank 0 only).",
    )

    return parser.parse_args()


def get_horizon_indices(context_len: int, seq_len: int, horizon_mode: str):
    """
    Return local time indices (0..seq_len-1) to evaluate for a given horizon_mode.

    Context frames:  indices [0 .. context_len-1]
    Forecast frames: indices [context_len .. seq_len-1]
    """
    forecast_indices = list(range(context_len, seq_len))

    if len(forecast_indices) == 0:
        raise ValueError(
            f"No forecast frames: context_len={context_len}, seq_len={seq_len}"
        )

    if horizon_mode == "all":
        return forecast_indices
    elif horizon_mode == "short":
        return [forecast_indices[0]]
    elif horizon_mode == "middle":
        if len(forecast_indices) < 2:
            return [forecast_indices[0]]
        return [forecast_indices[1]]
    elif horizon_mode == "long":
        return [forecast_indices[-1]]
    else:
        raise ValueError(f"Unknown horizon_mode: {horizon_mode}")


def depth_from_outputs(preds) -> np.ndarray:
    """
    Extract depth sequence [T, H, W] from a saved preds list
    (stored in the .pt file).
    """
    pts3ds_self = [pred["pts3d_in_self_view"].cpu() for pred in preds]
    pts3ds_self = torch.cat(pts3ds_self, dim=0)  # [T, H, W, 3]
    depth = pts3ds_self[..., -1]  # z coordinate, [T, H, W]
    return depth.numpy()


def get_rank_and_world_size():
    """Helper to read torchrun env vars, default to single process."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    return rank, world_size, local_rank


def maybe_init_distributed(world_size):
    if world_size > 1 and not dist.is_initialized():
        # gloo is fine even when using GPUs & allows gather_object
        dist.init_process_group(backend="gloo")


def main():
    args = parse_args()

    # ---- dataset-specific helpers ----
    data_root = Path(args.data_root)

    if args.dataset == "waymo":
        rgb_to_depth_path_fn = waymo_adapter.rgb_to_depth_path
        read_depth_fn = waymo_adapter.read_depth_exr

    elif args.dataset == "kitti":
        # For KITTI we want the version that maps from image_gathered ->
        # groundtruth_depth_gathered under the same data_root.
        def rgb_to_depth_path_fn(rgb_path: Path) -> Path:
            return kitti_adapter.rgb_to_depth_path(rgb_path, data_root)

        read_depth_fn = kitti_adapter.read_depth_kitti_png

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # ---- Multi-GPU device selection (torchrun-compatible) ----
    rank, world_size, local_rank = get_rank_and_world_size()
    maybe_init_distributed(world_size)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    print(f"[Rank {rank}/{world_size}] Using device: {device}")

    pred_root = Path(args.pred_root)
    all_seq_files = sorted(pred_root.glob("/.pt"))

    if len(all_seq_files) == 0:
        print(f"[Rank {rank}] No seq_*.pt files found under {pred_root}")
        return

    # ---- Multi-GPU sharding over sequences ----
    seq_files = all_seq_files[rank::world_size]
    print(
        f"[Rank {rank}] Evaluating {len(seq_files)} / {len(all_seq_files)} sequences "
        f"(indices {list(range(rank, len(all_seq_files), world_size))})"
    )

    gathered_metrics = []

    for pt_path in tqdm(seq_files, desc=f"Sequences (rank {rank})"):
        data = torch.load(pt_path, map_location="cpu")

        preds = data["preds"]
        segment = str(data["segment"])
        start_frame_idx = int(data["start_frame_idx"])
        stride = int(data["stride"])
        seq_len = int(data["seq_len"])
        context_len = int(data["context_len"])

        img_paths = [Path(p) for p in data["img_paths"]]

        # 0) Get predicted depth sequence from outputs
        pred_depth = depth_from_outputs(preds)  # [seq_len, H_pred, W_pred]

        # 1) Load GT depth for each frame in this predicted sequence
        gt_depths = []
        for rgb_path in img_paths:
            depth_path = rgb_to_depth_path_fn(rgb_path)
            depth = read_depth_fn(depth_path)
            gt_depths.append(depth)

        gt_depth = np.stack(gt_depths, axis=0)  # [seq_len, H_gt, W_gt]

        T, H_gt, W_gt = gt_depth.shape
        assert T == seq_len, "GT length and seq_len metadata do not match"

        # 2) Resize predictions to GT resolution if needed
        pr_depth_resized = []
        for t in range(seq_len):
            d = pred_depth[t]
            if d.shape != (H_gt, W_gt):
                d_res = cv2.resize(d, (W_gt, H_gt), interpolation=cv2.INTER_CUBIC)
            else:
                d_res = d
            pr_depth_resized.append(d_res)
        pr_depth_resized = np.stack(pr_depth_resized, axis=0)

        # 3) Build mask over time for the chosen horizon_mode
        eval_indices = get_horizon_indices(context_len, seq_len, args.horizon_mode)

        pr_eval = pr_depth_resized[eval_indices]  # [T_eval, H, W]
        gt_eval = gt_depth[eval_indices]          # [T_eval, H, W]

        # 4) Call depth_evaluation with appropriate alignment options
        align_with_lad2 = args.align == "scale&shift"
        align_with_scale = args.align == "scale"
        metric_scale = args.align == "metric"

        results, error_map, depth_pred_full, depth_gt_full = depth_evaluation(
            pr_eval,
            gt_eval,
            max_depth=args.max_depth,
            use_gpu=torch.cuda.is_available(),
            align_with_lad2=align_with_lad2,
            align_with_scale=align_with_scale,
            metric_scale=metric_scale,
        )

        gathered_metrics.append(results)

    # ---- Sync metrics across GPUs ----
    if world_size > 1:
        # Gather lists of metric dicts from all ranks to rank 0
        object_gather_list = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(
            gathered_metrics,
            object_gather_list=object_gather_list,
            dst=0,
        )

        if rank == 0:
            merged = []
            for lst in object_gather_list:
                merged.extend(lst)
            gathered_metrics = merged
        else:
            # Non-zero ranks are done after gather
            return

    # From here on: only rank 0 (world_size == 1 or gathered from all)
    if len(gathered_metrics) == 0:
        print("[Rank 0] No metrics computed.")
        return

    # 5) Aggregate metrics using valid_pixels as weights (global)
    valid_pixels_list = [m["valid_pixels"] for m in gathered_metrics]

    avg_metrics = {
        key: np.average(
            [m[key] for m in gathered_metrics],
            weights=valid_pixels_list,
        )
        for key in gathered_metrics[0].keys()
        if key != "valid_pixels"
    }

    print(
        f"\n[Rank 0] Aggregated depth metrics for "
        f"dataset={args.dataset}, horizon_mode={args.horizon_mode}, align={args.align}:"
    )
    for k, v in avg_metrics.items():
        print(f"[Rank 0]   {k}: {v:.6f}")

    # Single global JSON, only from rank 0
    if args.output_metrics is not None:
        out_path = Path(args.output_metrics)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {
                    "dataset": args.dataset,
                    "horizon_mode": args.horizon_mode,
                    "align": args.align,
                    "max_depth": args.max_depth,
                    "metrics": avg_metrics,
                },
                f,
                indent=2,
            )
        print(f"[Rank 0] Saved metrics to {out_path}")


if __name__ == "__main__":
    main()