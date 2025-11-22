from pathlib import Path
from tqdm import tqdm
import torch
import os
import re
from torch.utils.data import Dataset


@torch.no_grad()
def save_selected_fused_img_feats_by_sequence(
    model,
    dataloader,
    out_dir,
    device="cuda",
    cast_fp16=True,
):
    """
    Caches the 4 specified fused image feature layers for EVERY (sample, timestep).

    For each tXX.pt we store:
      {
        "layers": {
            "l0":   Tensor([S?, D]),
            "l2q":  Tensor([S,  D]),   # [:,1:] applied upstream if desired
            "l3q":  Tensor([S,  D]),   # [:,1:] applied upstream if desired
            "lfin": Tensor([S?, D]),
        },
        "meta": {"t": int, "seq_name": str, "start_stem": str},
        "view": {
            "depthmap": Tensor([H, W]) or ([S])   # optional, saved if present in views[t]
            "camera_pose": Tensor([4, 4])         # optional, saved if present in views[t]
        }
      }
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval().to(device)

    def _cpu_pack(x):
        """Move to CPU and (optionally) cast to fp16.

        NOTE: We intentionally keep small tensors like 4x4 poses in fp32 for
        numerical stability. Threshold chosen conservatively (<=32 elements).
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu()
            if cast_fp16 and x.dtype.is_floating_point:
                if x.numel() > 32:  # keep tiny matrices like 4x4 in fp32
                    x = x.to(torch.float16)
        elif isinstance(x, (list, tuple)):
            x = type(x)(_cpu_pack(xx) for xx in x)
        elif isinstance(x, dict):
            x = {k: _cpu_pack(v) for k, v in x.items()}
        return x

    for views in tqdm(dataloader):
        if isinstance(views, dict):
            views = [views]
        if len(views) == 0:
            continue

        # Infer batch size B
        v0 = views[0]
        if "camera_pose" in v0 and isinstance(v0["camera_pose"], torch.Tensor):
            B = v0["camera_pose"].shape[0]
        else:
            # pick any tensor/list field
            for k, v in v0.items():
                if isinstance(v, (list, tuple)):
                    B = len(v); break
                if isinstance(v, torch.Tensor) and v.ndim >= 1:
                    B = int(v.shape[0]); break

        # Move to device (only necessary inputs; keep simple)
        ignore = {"depthmap", "dataset", "label", "instance", "idx", "rng"}
        for view in views:
            for k, v in list(view.items()):
                if k in ignore:
                    continue
                if isinstance(v, (list, tuple)):
                    view[k] = [vv.to(device, non_blocking=True) if isinstance(vv, torch.Tensor) else vv for vv in v]
                else:
                    view[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v

        # Encode once
        shapes, feat_levels, pos = model._encode_views(views)
        # feat_levels[-1] is per-timestep image tokens
        feat = feat_levels[-1]

        # Init model state exactly like rollout does
        state_feat, state_pos = model._init_state(feat[0], pos[0])
        mem = model.pose_retriever.mem.expand(feat[0].shape[0], -1, -1)
        init_state_feat = state_feat.clone()
        init_mem = mem.clone()

        labels = v0.get("label", [f"seq_{i}" for i in range(B)])
        instances = v0.get("instance", [f"sample_{i}" for i in range(B)])

        # T loop
        for t in range(len(views)):
            feat_t, pos_t, shape_t = feat[t], pos[t], shapes[t]

            # Build pose tokens same as model (kept minimal)
            if getattr(model, "pose_head_flag", False):
                global_img_feat_t = model._get_img_level_feat(feat_t)
                pose_feat_t = model.pose_token.expand(feat_t.shape[0], -1, -1) if t == 0 \
                              else model.pose_retriever.inquire(global_img_feat_t, mem)
                pose_pos_t = -torch.ones(feat_t.shape[0], 1, 2, device=feat_t.device, dtype=pos_t.dtype)
            else:
                global_img_feat_t = None
                pose_feat_t = None
                pose_pos_t = None

            # One recurrent step -> list(dec_l) with length dec_depth+1
            new_state_feat, dec = model._recurrent_rollout(
                state_feat, state_pos, feat_t, pos_t,
                pose_feat_t, pose_pos_t,
                init_state_feat,
                img_mask=views[t]["img_mask"],
                reset_mask=views[t]["reset"],
                update=views[t].get("update", None),
            )

            # Update mem as model does
            out_pose_feat_t = dec[-1][:, 0:1]
            if getattr(model, "pose_head_flag", False):
                new_mem = model.pose_retriever.update_mem(mem, global_img_feat_t, out_pose_feat_t)
            else:
                new_mem = mem

            # Pick the 4 layers
            ddepth = model.dec_depth  # assumed available in your model
            l0   = dec[0].float()
            l2q  = dec[(ddepth * 2) // 4]  # [:, 1:] drop pose if you want, we normalize later
            l3q  = dec[(ddepth * 3) // 4]
            lfin = dec[ddepth]

            # --- NEW: gather per-(b,t) depth & pose directly from views (left on CPU)
            has_depth = ("depthmap" in views[t]) and isinstance(views[t]["depthmap"], torch.Tensor)
            has_pose  = ("camera_pose" in views[t]) and isinstance(views[t]["camera_pose"], torch.Tensor)
            depth_t = views[t]["depthmap"] if has_depth else None    # [B, H, W] or [B, S]
            pose_t  = views[t]["camera_pose"] if has_pose else None  # [B, 4, 4] or similar

            for b in range(B):
                seq_name = str(labels[b]) if isinstance(labels, (list, tuple)) else "unknown"
                start_stem = Path(str(instances[b]) if isinstance(instances, (list, tuple)) else f"sample_{b}").stem

                payload = {
                    "layers": {
                        "l0":   l0[b],
                        "l2q":  l2q[b],
                        "l3q":  l3q[b],
                        "lfin": lfin[b],
                    },
                    "meta": {"t": int(t), "seq_name": seq_name, "start_stem": start_stem},
                }

                # Attach view data if available
                view_block = {}
                if depth_t is not None:
                    view_block["depthmap"] = depth_t[b]
                if pose_t is not None:
                    view_block["camera_pose"] = pose_t[b]
                if view_block:
                    payload["view"] = view_block

                subdir = out_dir / seq_name / start_stem
                subdir.mkdir(parents=True, exist_ok=True)
                torch.save(_cpu_pack(payload), subdir / f"t{t:02d}.pt")

            # Roll state/mem forward (exactly like the model)
            img_mask = views[t]["img_mask"]
            update = views[t].get("update", None)
            update_mask = (img_mask & update) if update is not None else img_mask
            update_mask = update_mask[:, None, None].float()
            state_feat = new_state_feat * update_mask + state_feat * (1 - update_mask)
            mem = new_mem * update_mask + mem * (1 - update_mask)

            reset_mask = views[t]["reset"]
            if reset_mask is not None:
                reset_mask = reset_mask[:, None, None].float()
                state_feat = init_state_feat * reset_mask + state_feat * (1 - reset_mask)
                mem = init_mem * reset_mask + mem * (1 - reset_mask)


if __name__ == "__main__":
    import os
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    # ----- multi-GPU setup -----
    # When launched with torchrun, these are set automatically.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)

    # ----- paths -----
    waymo_dir = "/workspace/raid/jevers/cut3r_processed_waymo/validation"
    feature_dir = "/workspace/raid/jevers/cut3r_features/waymo/fused_img_tokens_224/validation"

    # ----- dataset & sampler -----
    from dust3r.datasets.waymo import Waymo_Multi_TStride
    waymo_ds = Waymo_Multi_TStride(
        num_views=7,
        resolution=224,
        ROOT=waymo_dir,
        temporal_stride=2,
        overlap_step=5,
    )

    # This splits the dataset over world_size processes
    sampler = DistributedSampler(
        waymo_ds,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=False,
    )

    loader = DataLoader(
        waymo_ds,
        batch_size=16,
        sampler=sampler,          # <- important: no shuffle when using sampler
        num_workers=4,            # you can bump this up to better feed the GPUs
        pin_memory=True,
        drop_last=False,
    )

    if local_rank == 0:
        print("World size:", world_size)
        print("Number of sequences in waymo dataset: ", len(waymo_ds))

    # ----- model on the per-rank GPU -----
    from dust3r.model import ARCroco3DStereo
    cut3r_model = ARCroco3DStereo.from_pretrained(
        "/workspace/CUT3R/cut3r_512_dpt_4_64.pth"
    ).to(device)
    cut3r_model.eval()

    # ----- run caching on this rank's shard -----
    save_selected_fused_img_feats_by_sequence(
        cut3r_model,
        loader,
        out_dir=feature_dir,
        device=device,            # e.g. "cuda:3" on rank 3
        cast_fp16=True,
    )

    print("All features cached. :)")
