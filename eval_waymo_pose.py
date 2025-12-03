#!/usr/bin/env python
import os
import argparse
from collections.abc import Mapping

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import Subset

import sys
sys.path.append("/workspace/CUT3R/src")

# ------------------ CUT3R / eval imports (from your repo) ------------------ #
from add_ckpt_path import add_path_to_dust3r

from eval.relpose.utils import (
    c2w_to_tumpose,
    get_tum_poses,
)
from eval.relpose.evo_utils import eval_metrics, eval_metrics_first_pose_align_last_pose

# ------------------ Your Waymo dataset (ADJUST THIS IMPORT!) ------------------ #
# e.g.: from datasets.waymo_multi_tstride import Waymo_Multi_TStride
from dust3r.datasets.waymo import Waymo_Multi_TStride



# ----------------------------------------------------------------------
# Arg parsing
# ----------------------------------------------------------------------
def get_args_parser():
    parser = argparse.ArgumentParser("Waymo pose evaluation (DDP, direct dataset loop)")

    parser.add_argument(
        "--weights",
        type=str,
        default="/workspace/CUT3R/cut3r_512_dpt_4_64.pth",
        help="Path to the model weights (dust3r / cut3r checkpoint)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Base device string, usually 'cuda'",
    )

    # Waymo dataset root
    parser.add_argument(
        "--waymo_root",
        type=str,
        default="/workspace/raid/jevers/cut3r_processed_waymo/validation",
        help="Root folder of preprocessed Waymo data (same ROOT as in Waymo_Multi_TStride)",
    )

    # dataset sampling
    parser.add_argument(
        "--num_views",
        type=int,
        default=8,
        help="Number of views per sample (must match training / dataset config)",
    )
    parser.add_argument(
        "--temporal_stride",
        type=int,
        default=2,
        help="Temporal stride between frames inside a sample",
    )
    parser.add_argument(
        "--overlap_step",
        type=int,
        default=5,
        help="Step for overlapping windows; if None, defaults to temporal_stride",
    )
    parser.add_argument(
        "--non_overlapping",
        action="store_true",
        default=False,
        help="If set, windows are disjoint (no overlap)",
    )

    # model / inference behaviour
    parser.add_argument(
        "--revisit",
        type=int,
        default=1,
        help="Revisit parameter passed to prepare_output (same as original code)",
    )
    parser.add_argument(
        "--solve_pose",
        action="store_true",
        default=False,
        help="If set, estimate pose from depth via weighted procrustes",
    )

    return parser

def recover_cam_params(pts3ds_self, pts3ds_other, conf_self, conf_other):
    # lazy imports to avoid circular import issues at module import time
    from dust3r.post_process import estimate_focal_knowing_depth
    from dust3r.utils.geometry import weighted_procrustes

    B, H, W, _ = pts3ds_self.shape

    # principal point at the image center
    pp = (
        torch.tensor([W // 2, H // 2], device=pts3ds_self.device)
        .float()
        .repeat(B, 1)
        .reshape(B, 1, 2)
    )

    # estimate focal length from depth
    focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")

    # flatten spatial dims for Procrustes
    pts3ds_self = pts3ds_self.reshape(B, -1, 3)
    pts3ds_other = pts3ds_other.reshape(B, -1, 3)
    conf_self = conf_self.reshape(B, -1)
    conf_other = conf_other.reshape(B, -1)

    # weighted Procrustes to recover camera transforms
    c2w = weighted_procrustes(
        pts3ds_self,
        pts3ds_other,
        torch.log(conf_self) * torch.log(conf_other),
        use_weights=True,
        return_T=True,
    )
    return c2w, focal, pp.reshape(B, 2)

def prepare_output(outputs, revisit=1, solve_pose=False):
    # lazy imports to avoid circular imports
    from dust3r.utils.camera import pose_encoding_to_camera
    from dust3r.post_process import estimate_focal_knowing_depth

    valid_length = len(outputs["pred"]) // revisit
    outputs["pred"] = outputs["pred"][-valid_length:]
    outputs["views"] = outputs["views"][-valid_length:]

    if solve_pose:
        pts3ds_self = [
            output["pts3d_in_self_view"].cpu() for output in outputs["pred"]
        ]
        pts3ds_other = [
            output["pts3d_in_other_view"].cpu() for output in outputs["pred"]
        ]
        conf_self = [output["conf_self"].cpu() for output in outputs["pred"]]
        conf_other = [output["conf"].cpu() for output in outputs["pred"]]
        pr_poses, focal, pp = recover_cam_params(
            torch.cat(pts3ds_self, 0),
            torch.cat(pts3ds_other, 0),
            torch.cat(conf_self, 0),
            torch.cat(conf_other, 0),
        )
        pts3ds_self = torch.cat(pts3ds_self, 0)
    else:
        pts3ds_self = [
            output["pts3d_in_self_view"].cpu() for output in outputs["pred"]
        ]
        pts3ds_other = [
            output["pts3d_in_other_view"].cpu() for output in outputs["pred"]
        ]
        conf_self = [output["conf_self"].cpu() for output in outputs["pred"]]
        conf_other = [output["conf"].cpu() for output in outputs["pred"]]
        pts3ds_self = torch.cat(pts3ds_self, 0)

        pr_poses = [
            pose_encoding_to_camera(pred["camera_pose"].clone()).cpu()
            for pred in outputs["pred"]
        ]
        pr_poses = torch.cat(pr_poses, 0)

        B, H, W, _ = pts3ds_self.shape
        pp = (
            torch.tensor([W // 2, H // 2], device=pts3ds_self.device)
            .float()
            .repeat(B, 1)
            .reshape(B, 2)
        )
        focal = estimate_focal_knowing_depth(
            pts3ds_self, pp, focal_mode="weiszfeld"
        )

    colors = [0.5 * (output["rgb"][0] + 1.0) for output in outputs["pred"]]
    cam_dict = {
        "focal": focal.cpu().numpy(),
        "pp": pp.cpu().numpy(),
    }
    return (
        colors,
        pts3ds_self,
        pts3ds_other,
        conf_self,
        conf_other,
        cam_dict,
        pr_poses,
    )

# ----------------------------------------------------------------------
# Distributed helpers
# ----------------------------------------------------------------------
def init_distributed():
    """
    Initialize torch.distributed if launched with torchrun.

    Returns:
        rank, world_size, local_rank
    """
    if dist.is_available() and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    return rank, world_size, local_rank


# ----------------------------------------------------------------------
# Your batchify helpers (exactly what you gave, just pasted here)
# ----------------------------------------------------------------------
num = (int, float, bool, np.integer, np.floating, np.bool_)


def _to_torch_with_batch(x):
    # 1) torch / numpy -> torch, add leading batch dim
    if isinstance(x, torch.Tensor):
        return x.unsqueeze(0)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).unsqueeze(0)
    # 2) scalars -> 1-elem tensor
    if isinstance(x, num):
        # keep dtype sensibly (bool, int64, float32 default)
        if isinstance(x, (bool, np.bool_)):
            return torch.tensor([x], dtype=torch.bool)
        if isinstance(x, (int, np.integer)):
            return torch.tensor([int(x)], dtype=torch.int64)
        return torch.tensor([float(x)], dtype=torch.float32)
    # 3) strings/paths -> batch size 1 as list[str]
    if isinstance(x, str):
        return [x]
    # 4) tuples/lists:
    if isinstance(x, tuple):
        # Special: tuple of scalars (e.g., idx) -> list of 1-elem tensors
        if all(isinstance(e, num) for e in x):
            return [_to_torch_with_batch(e) for e in x]
        # Otherwise elementwise recurse, keep tuple
        return tuple(_to_torch_with_batch(e) for e in x)
    if isinstance(x, list):
        # Keep list structure but batchify elements
        if len(x) > 0 and isinstance(x[0], str):
            # list of strings: assume already per-batch; keep as is
            return x
        return [_to_torch_with_batch(e) for e in x]
    # 5) dicts: recurse
    if isinstance(x, Mapping):
        return {k: _to_torch_with_batch(v) for k, v in x.items()}
    # 6) fallback: wrap as single-item list
    return [x]


def batchify_per_view_like_you_want(views, inplace=False):
    """
    Input:  views = List[Dict] from your dataset.
    Output: List[Dict] with the SAME length/structure, but:
        - torch.Tensor -> (1, …)
        - np.ndarray   -> torch.Tensor with (1, …)
        - scalars/bools -> 1-elem tensors
        - strings/paths -> ['...']
        - tuples of scalars (e.g., idx) -> [tensor([..]), tensor([..]), ...]
        - complex tuples/lists (e.g., corres) -> elementwise torch + batch dim
    """
    assert isinstance(views, (list, tuple)) and len(views) > 0 and isinstance(
        views[0], Mapping
    )

    if inplace:
        for view in views:
            for k, v in list(view.items()):
                view[k] = _to_torch_with_batch(v)
        return views

    new_views = []
    for view in views:
        new_views.append({k: _to_torch_with_batch(v) for k, v in view.items()})
    return new_views


# ----------------------------------------------------------------------
# Trajectory helper
# ----------------------------------------------------------------------
def build_gt_traj_from_views(views):
    """
    Build ground truth trajectory in TUM format directly from the dataset views.

    Each view must contain a 'camera_pose' 4x4 cam2world matrix
    (numpy array or torch tensor, optionally with a leading batch dim).

    Returns:
        [traj_tum, timestamps] where:
          traj_tum: (N,7) array of [x y z qw qx qy qz]
          timestamps: (N,) float indices 0..N-1
    """
    tum_poses = []

    for v in views:
        c2w = v["camera_pose"]

        if isinstance(c2w, torch.Tensor):
            c2w = c2w.detach().cpu().numpy()
        else:
            c2w = np.asarray(c2w)

        # allow (4,4) or (1,4,4)
        if c2w.shape == (1, 4, 4):
            c2w = c2w[0]

        assert c2w.shape == (4, 4), f"camera_pose must be 4x4, got {c2w.shape}"

        tum_pose = c2w_to_tumpose(c2w)  # [x y z qw qx qy qz]
        tum_poses.append(tum_pose)

    tum_poses = np.stack(tum_poses, 0)  # [N,7]
    timestamps = np.arange(len(tum_poses)).astype(float)
    return [tum_poses, timestamps]


# ----------------------------------------------------------------------
# Core evaluation loop (DDP, direct dataset iteration)
# ----------------------------------------------------------------------
def eval_pose_dataset_ddp(args, model, dataset, rank, world_size, local_rank, device):
    """
    - No DataLoader, no PyTorch collate.
    - Each rank iterates a disjoint subset of indices: idx = rank, rank+world_size, ...
    - For each idx:
        * sample = dataset[idx]    # List[Dict] of views
        * gt_traj from raw sample
        * batched_views = batchify_per_view_like_you_want(sample)
        * outputs, _ = inference(batched_views, model, device)
        * pred_traj = get_tum_poses(...)
        * metrics = eval_metrics(pred_traj, gt_traj, filename=os.devnull)
    - Degenerate trajectories (Umeyama alignment failures) are skipped and counted.
    - All-reduce sums, counts, and skipped counts across ranks.
    """
    from dust3r.inference import inference  # after add_path_to_dust3r()

    n_samples = len(dataset)
    indices = list(range(rank, n_samples, world_size))

    local_sum_ate = 0.0
    local_sum_rpe_trans = 0.0
    local_sum_rpe_rot = 0.0
    local_good_count = 0
    local_bad_count = 0  # degenerate / skipped

    iterable = indices
    if rank == 0:
        iterable = tqdm(indices, desc="Waymo pose eval")

    model.eval()

    with torch.no_grad():
        for idx in iterable:
            views = dataset[idx]  # List[Dict] from your _get_views

            assert isinstance(views, (list, tuple)) and isinstance(
                views[0], Mapping
            ), "Dataset _getitem_ must return a list of view dicts"

            # 1) GT trajectory from ORIGINAL, unbatched views
            gt_traj = build_gt_traj_from_views(views)

            # 2) Batchify views so inference likes them
            batched_views = batchify_per_view_like_you_want(views, inplace=False)

            # 3) Run model
            outputs, _ = inference(batched_views, model, device)

            (
                colors,
                pts3ds_self,
                pts3ds_other,
                conf_self,
                conf_other,
                cam_dict,
                pr_poses,
            ) = prepare_output(
                outputs,
                revisit=args.revisit,
                solve_pose=args.solve_pose,
            )

            # 4) Predicted trajectory (same format as GT, [tum_poses, timestamps])
            pred_traj = get_tum_poses(pr_poses)

            # 5) Metrics, with robust handling for degenerate trajectories
            try:
                ate, rpe_trans, rpe_rot = eval_metrics(
                    pred_traj,
                    gt_traj,
                    seq=f"sample_{idx}",
                    filename=os.devnull,  # don't clutter disk
                )
                #ate_2 = eval_metrics_first_pose_align_last_pose(
                #    pred_traj, gt_traj, filename=os.devnull
                #)
                
                #print("ate align first pose: ", ate_2)
            except Exception as e:
                msg = str(e)
                if (
                    "Degenerate covariance rank" in msg
                    or "Umeyama alignment is not possible" in msg
                ):
                    local_bad_count += 1
                    if rank == 0:
                        print(
                            f"[rank0] Skipping degenerate trajectory at idx {idx}: {msg}"
                        )
                    continue  # skip this sample
                else:
                    # unexpected error: re-raise so we actually see it
                    raise

            # only reached if eval_metrics succeeded
            local_sum_ate += ate
            local_sum_rpe_trans += rpe_trans
            local_sum_rpe_rot += rpe_rot
            local_good_count += 1

    # Aggregate across ranks: sums + good_count + bad_count
    loc = torch.tensor(
        [
            local_sum_ate,
            local_sum_rpe_trans,
            local_sum_rpe_rot,
            float(local_good_count),
            float(local_bad_count),
        ],
        device=device,
        dtype=torch.float64,
    )

    if dist.is_initialized():
        dist.all_reduce(loc, op=dist.ReduceOp.SUM)

    global_sum_ate, global_sum_rpe_trans, global_sum_rpe_rot, global_good_count, global_bad_count = loc.tolist()

    if global_good_count == 0:
        avg_ate = avg_rpe_trans = avg_rpe_rot = 0.0
    else:
        avg_ate = global_sum_ate / global_good_count
        avg_rpe_trans = global_sum_rpe_trans / global_good_count
        avg_rpe_rot = global_sum_rpe_rot / global_good_count

    # small summary on rank 0
    if rank == 0:
        total_tried = int(global_good_count + global_bad_count)
        print(
            f"[pose eval] Good trajectories: {int(global_good_count)} / {total_tried}, "
            f"skipped (degenerate): {int(global_bad_count)}"
        )

    return avg_ate, avg_rpe_trans, avg_rpe_rot


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = get_args_parser()
    args = parser.parse_args()

    # init distributed
    rank, world_size, local_rank = init_distributed()
    is_main = (rank == 0)

    if is_main:
        print(f"[main] world_size={world_size}, rank={rank}, local_rank={local_rank}")

    # 🔑 set CUDA device per rank
    if args.device.startswith("cuda"):
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device)

    # make sure dust3r is importable
    add_path_to_dust3r(args.weights)

    # import model class after path patch
    from dust3r.model import ARCroco3DStereo

    if is_main:
        print(f"[main] Loading model from {args.weights}")
    model = ARCroco3DStereo.from_pretrained(args.weights)
    model.to(device)
    model.eval()

    # build full dataset (same on all ranks)
    full_dataset = Waymo_Multi_TStride(
        num_views=args.num_views,
        resolution=(224, 224),
        ROOT=args.waymo_root,
        temporal_stride=args.temporal_stride,
        overlap_step=args.overlap_step,
        non_overlapping=args.non_overlapping,
    )

    if is_main:
        print(f"[main] Loaded Waymo_Multi_TStride with {len(full_dataset)} samples total")

    # ---------- NEW: take a subset ----------
    # choose how many samples you want globally
    MAX_SAMPLES = len(full_dataset)  # <-- change this to whatever you like

    n_total = len(full_dataset)
    if MAX_SAMPLES is not None and MAX_SAMPLES > 0 and MAX_SAMPLES < n_total:
        # make sure all ranks use the SAME subset indices
        if rank == 0:
            # you can choose first N or random N; here: random subset
            torch.manual_seed=123
            perm = torch.randperm(n_total)[:MAX_SAMPLES].tolist()
        else:
            perm = None

        if dist.is_initialized():
            # broadcast the index list from rank 0 to all other ranks
            obj_list = [perm]
            dist.broadcast_object_list(obj_list, src=0)
            perm = obj_list[0]

        dataset = Subset(full_dataset, perm)

        if is_main:
            print(f"[main] Using subset of size {len(dataset)} (out of {n_total})")
    else:
        dataset = full_dataset
        if is_main:
            print(f"[main] Using full dataset of size {n_total}")
    # ---------- end subset logic ----------

    # run evaluation
    avg_ate, avg_rpe_trans, avg_rpe_rot = eval_pose_dataset_ddp(
        args, model, dataset, rank, world_size, local_rank, device
    )

    if is_main:
        print("========================================")
        print("Waymo pose evaluation (DDP, direct dataset loop, in-memory)")
        print(f"Average ATE       : {avg_ate:.6f}")
        print(f"Average RPE trans : {avg_rpe_trans:.6f}")
        print(f"Average RPE rot   : {avg_rpe_rot:.6f}")
        print("========================================")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()