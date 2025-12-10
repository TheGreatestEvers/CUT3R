import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# ---- repo paths (adjust if needed) ----
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))  # repo root
sys.path.append("/workspace/CUT3R/src")
sys.path.append("/workspace/CUT3R")

# ---- CUT3R / DUST3R imports ----
from dust3r.utils.camera import pose_encoding_to_camera

# relpose utils: exposes c2w_to_tumpose, get_tum_poses, eval_metrics, etc.
from eval.relpose.utils import (
    c2w_to_tumpose,
    get_tum_poses,
    eval_metrics,
)
from eval.relpose.evo_utils import (
    process_directory,
    calculate_averages,
)

# ---- your Waymo helpers ----
from eval.foresight.waymo_adapter import (
    rgb_to_cam_params_path,
    read_camera_params_npz,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pred_root",
        type=str,
        default="/workspace/raid/jevers/waymo_outputs",
        help="Root where forecast .pt files are stored (output_dir from run script)",
    )

    parser.add_argument(
        "--horizon_mode",
        type=str,
        default="all",
        choices=["short", "middle", "long", "all"],
        help=(
            "Which forecast horizon to evaluate:\n"
            "  short  -> first forecast frame\n"
            "  middle -> second forecast frame (if exists, else first)\n"
            "  long   -> last forecast frame\n"
            "  all    -> all forecast frames"
        ),
    )

    parser.add_argument(
        "--output_metrics",
        type=str,
        default=None,
        help="Optional JSON file to save aggregated ATE/RPE metrics",
    )

    return parser.parse_args()


def get_horizon_indices(context_len: int, seq_len: int, horizon_mode: str):
    """
    Same logic as your depth eval.

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


# ---------- Trajectory helpers ----------

def extract_pred_traj(outputs):
    """
    Build predicted trajectory (TUM format) from saved CUT3R/DUST3R outputs.

    outputs: dict with key 'pred' (list of per-view dicts), coming from
             your save script:
                 outputs = dict(views=batch, pred=preds)

    returns:
        pred_traj: [traj_tum, timestamps] as expected by eval_metrics()
                   where traj_tum is (N, 7) [x,y,z,qx,qy,qz,qw]
    """
    preds = outputs["pred"]  # list of dicts
    poses_c2w = []

    for pred in preds:
        # pred["camera_pose"] is the pose encoding used in original CUT3R code
        cam_enc = pred["camera_pose"].clone()  # [1, ...]
        c2w = pose_encoding_to_camera(cam_enc)  # [1, 4, 4] camera-to-world
        poses_c2w.append(c2w[0].cpu())  # 4x4

    # get_tum_poses expects a list of 4x4 matrices
    pred_traj = get_tum_poses(poses_c2w)  # -> [tum_poses, timestamps]
    return pred_traj


def build_gt_traj_from_npz(img_paths):
    """
    Build GT trajectory from per-frame camera NPZ files.

    Each NPZ contains keys 'intrinsics' and 'cam2world' (4x4).
    We interpret cam2world directly as camera-to-world pose.
    """
    tum_poses = []

    for rgb_path in img_paths:
        cam_path = rgb_to_cam_params_path(rgb_path)
        intrinsics, cam2world = read_camera_params_npz(cam_path)

        # c2w_to_tumpose converts a 4x4 camera-to-world to
        # (x, y, z, qw, qx, qy, qz)
        tum_pose = c2w_to_tumpose(cam2world)
        tum_poses.append(tum_pose)

    tum_poses = np.stack(tum_poses, axis=0)           # [N, 7]
    timestamps = np.arange(len(tum_poses)).astype(float)  # simple frame index timestamps

    gt_traj = [tum_poses, timestamps]
    return gt_traj


def subselect_traj(traj, indices):
    """
    Subselects frames from a TUM-format traj (traj_tum, timestamps).
    """
    traj_tum, timestamps = traj
    traj_tum_sel = traj_tum[indices]
    timestamps_sel = timestamps[indices]
    return [traj_tum_sel, timestamps_sel]


# ---------- Main evaluation loop ----------

def main():
    import json

    args = parse_args()
    pred_root = Path(args.pred_root)

    # All saved seq files
    seq_files = sorted(pred_root.glob("/.pt"))

    if len(seq_files) == 0:
        print(f"No seq_*.pt files found under {pred_root}")
        return

    print(f"Found {len(seq_files)} saved sequences under {pred_root}")

    ate_list = []
    rpe_trans_list = []
    rpe_rot_list = []

    # optional: write per-sequence metrics in a CUT3R-compatible way
    # so we can reuse process_directory / calculate_averages
    for pt_path in tqdm(seq_files, desc="Pose sequences"):
        data = torch.load(pt_path, map_location="cpu")

        outputs = data["outputs"]
        segment = str(data["segment"])
        start_frame_idx = int(data["start_frame_idx"])
        stride = int(data["stride"])
        seq_len = int(data["seq_len"])
        context_len = int(data["context_len"])

        img_paths = [Path(p) for p in data["img_paths"]]
        assert len(img_paths) == seq_len, "img_paths length != seq_len metadata"

        # Build full predicted and GT trajectories
        pred_traj_full = extract_pred_traj(outputs)
        gt_traj_full = build_gt_traj_from_npz(img_paths)

        # Subselect frames according to forecasting horizon
        eval_indices = get_horizon_indices(context_len, seq_len, args.horizon_mode)

        pred_traj = subselect_traj(pred_traj_full, eval_indices)
        gt_traj = subselect_traj(gt_traj_full, eval_indices)

        # Sequence name for logging
        seq_name = f"{segment}seq{start_frame_idx:05d}"

        # Per-sequence metrics file, following CUT3R naming convention
        metrics_txt = pt_path.with_suffix("").as_posix() + "_pose_eval_metric.txt"

        ate, rpe_trans, rpe_rot = eval_metrics(
            pred_traj,
            gt_traj,
            seq=seq_name,
            filename=metrics_txt,
        )

        ate_list.append(ate)
        rpe_trans_list.append(rpe_trans)
        rpe_rot_list.append(rpe_rot)

        print(
            f"[{seq_name}] ATE: {ate:.5f}, "
            f"RPE_trans: {rpe_trans:.5f}, RPE_rot: {rpe_rot:.5f}"
        )

    if len(ate_list) == 0:
        print("No pose metrics computed.")
        return

    # Option 1: simple mean over sequences (like original eval_pose_estimation_dist)
    avg_ate = float(np.mean(ate_list))
    avg_rpe_trans = float(np.mean(rpe_trans_list))
    avg_rpe_rot = float(np.mean(rpe_rot_list))

    print(
        f"\n[Simple mean over sequences] "
        f"ATE: {avg_ate:.5f}, RPE_trans: {avg_rpe_trans:.5f}, RPE_rot: {avg_rpe_rot:.5f}"
    )

    # Option 2: use evo_utils.process_directory + calculate_averages over *_metric.txt
    results = process_directory(str(pred_root))
    avg_ate_proc, avg_rpe_trans_proc, avg_rpe_rot_proc = calculate_averages(results)

    print(
        f"[process_directory / calculate_averages] "
        f"ATE: {avg_ate_proc:.5f}, "
        f"RPE_trans: {avg_rpe_trans_proc:.5f}, "
        f"RPE_rot: {avg_rpe_rot_proc:.5f}"
    )

    # Optionally dump JSON
    if args.output_metrics is not None:
        out_path = Path(args.output_metrics)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {
                    "dataset": "waymo",
                    "metric_type": "pose",
                    "horizon_mode": args.horizon_mode,
                    "metrics_simple_mean": {
                        "ATE": avg_ate,
                        "RPE_trans": avg_rpe_trans,
                        "RPE_rot": avg_rpe_rot,
                    },
                    "metrics_process_directory": {
                        "ATE": avg_ate_proc,
                        "RPE_trans": avg_rpe_trans_proc,
                        "RPE_rot": avg_rpe_rot_proc,
                    },
                },
                f,
                indent=2,
            )
        print(f"Saved aggregated metrics to {out_path}")


if __name__ == "__main__":
    main()