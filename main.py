import os, sys
sys.path.append("/workspace/CUT3R/src")
from dust3r.utils.device import to_cpu
import numpy as np
import torch
import time
import glob
import random
import cv2
import argparse
import tempfile
import shutil
from copy import deepcopy
from add_ckpt_path import add_path_to_dust3r
import imageio.v2 as iio

from dust3r.model import ARCroco3DStereo



def prepare_output(outputs, outdir, revisit=1, use_pose=True):
    """
    Process inference outputs to generate point clouds and camera parameters for visualization.

    Args:
        outputs (dict): Inference outputs.
        revisit (int): Number of revisits per view.
        use_pose (bool): Whether to transform points using camera pose.

    Returns:
        tuple: (points, colors, confidence, camera parameters dictionary)
    """
    from dust3r.utils.camera import pose_encoding_to_camera
    from dust3r.post_process import estimate_focal_knowing_depth
    from dust3r.utils.geometry import geotrf

    # Only keep the outputs corresponding to one full pass.
    valid_length = len(outputs["pred"]) // revisit
    outputs["pred"] = outputs["pred"][-valid_length:]
    outputs["views"] = outputs["views"][-valid_length:]

    pts3ds_self_ls = [output["pts3d_in_self_view"].cpu() for output in outputs["pred"]]
    pts3ds_other = [output["pts3d_in_other_view"].cpu() for output in outputs["pred"]]
    conf_self = [output["conf_self"].cpu() for output in outputs["pred"]]
    conf_other = [output["conf"].cpu() for output in outputs["pred"]]
    pts3ds_self = torch.cat(pts3ds_self_ls, 0)

    # Recover camera poses.
    pr_poses = [
        pose_encoding_to_camera(pred["camera_pose"].clone()).cpu()
        for pred in outputs["pred"]
    ]
    R_c2w = torch.cat([pr_pose[:, :3, :3] for pr_pose in pr_poses], 0)
    t_c2w = torch.cat([pr_pose[:, :3, 3] for pr_pose in pr_poses], 0)

    if use_pose:
        transformed_pts3ds_other = []
        for pose, pself in zip(pr_poses, pts3ds_self):
            transformed_pts3ds_other.append(geotrf(pose, pself.unsqueeze(0)))
        pts3ds_other = transformed_pts3ds_other
        conf_other = conf_self

    # Estimate focal length based on depth.
    B, H, W, _ = pts3ds_self.shape
    pp = torch.tensor([W // 2, H // 2], device=pts3ds_self.device).float().repeat(B, 1)
    focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")

    colors = [
        0.5 * (output["img"].permute(0, 2, 3, 1) + 1.0) for output in outputs["views"]
    ]

    cam_dict = {
        "focal": focal.cpu().numpy(),
        "pp": pp.cpu().numpy(),
        "R": R_c2w.cpu().numpy(),
        "t": t_c2w.cpu().numpy(),
    }

    pts3ds_self_tosave = pts3ds_self  # B, H, W, 3
    depths_tosave = pts3ds_self_tosave[..., 2]
    pts3ds_other_tosave = torch.cat(pts3ds_other)  # B, H, W, 3
    conf_self_tosave = torch.cat(conf_self)  # B, H, W
    conf_other_tosave = torch.cat(conf_other)  # B, H, W
    colors_tosave = torch.cat(
        [
            0.5 * (output["img"].permute(0, 2, 3, 1).cpu() + 1.0)
            for output in outputs["views"]
        ]
    )  # [B, H, W, 3]
    cam2world_tosave = torch.cat(pr_poses)  # B, 4, 4
    intrinsics_tosave = (
        torch.eye(3).unsqueeze(0).repeat(cam2world_tosave.shape[0], 1, 1)
    )  # B, 3, 3
    intrinsics_tosave[:, 0, 0] = focal.detach().cpu()
    intrinsics_tosave[:, 1, 1] = focal.detach().cpu()
    intrinsics_tosave[:, 0, 2] = pp[:, 0]
    intrinsics_tosave[:, 1, 2] = pp[:, 1]

    os.makedirs(os.path.join(outdir, "depth"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "conf"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "color"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "camera"), exist_ok=True)
    for f_id in range(len(pts3ds_self)):
        depth = depths_tosave[f_id].cpu().numpy()
        conf = conf_self_tosave[f_id].cpu().numpy()
        color = colors_tosave[f_id].cpu().numpy()
        c2w = cam2world_tosave[f_id].cpu().numpy()
        intrins = intrinsics_tosave[f_id].cpu().numpy()
        np.save(os.path.join(outdir, "depth", f"{f_id:06d}.npy"), depth)
        np.save(os.path.join(outdir, "conf", f"{f_id:06d}.npy"), conf)
        iio.imwrite(
            os.path.join(outdir, "color", f"{f_id:06d}.png"),
            (color * 255).astype(np.uint8),
        )
        np.savez(
            os.path.join(outdir, "camera", f"{f_id:06d}.npz"),
            pose=c2w,
            intrinsics=intrins,
        )

    return pts3ds_other, colors, conf_other, cam_dict#

@torch.no_grad()
def run_inference_with_forecast(cut3r_model: ARCroco3DStereo, forecaster, views, output_dir, size):
    device = "cuda"

    from viser_utils import PointCloudViewer

    # Run inference.
    print("Running inference...")
    start_time = time.time()

    # standard cut3r    
    # outputs, state_args = inference(views, model, device)
    # Move views to cuda
    ignore_keys = set(
        ["depthmap", "dataset", "label", "instance", "idx", "rng"] # removed "true_shape"
    )
    for view in views:
        for name in view.keys():  # pseudo_focal
            if name in ignore_keys:
                continue
            if isinstance(view[name], tuple) or isinstance(view[name], list):
                view[name] = [x.to(device, non_blocking=True) for x in view[name]]
            else:
                view[name] = view[name].to(device, non_blocking=True)
    
    # Inference
    output, state_args = model.forward_with_forecaster(views, forecaster, context_len=4)
    preds, batch = output.ress, output.views
    outputs = dict(views=batch, pred=preds)

    total_time = time.time() - start_time
    print(
        f"Inference completed in {total_time:.2f} seconds."
    )

    # Process outputs for visualization.
    print("Preparing output for visualization...")
    pts3ds_other, colors, conf, cam_dict = prepare_output(
        outputs, output_dir, 1, True
    )

    # Convert tensors to numpy arrays for visualization.
    pts3ds_to_vis = [p.cpu().numpy() for p in pts3ds_other]
    colors_to_vis = [c.cpu().numpy() for c in colors]
    edge_colors = [None] * len(pts3ds_to_vis)

    import pickle
    with open(f"pts3d_quant_onestage.pkl", "wb") as f:
        pickle.dump(pts3ds_to_vis, f)

    # Create and run the point cloud viewer.
    print("Launching point cloud viewer...")
    viewer = PointCloudViewer(
        model,
        state_args,
        pts3ds_to_vis,
        colors_to_vis,
        conf,
        cam_dict,
        device=device,
        edge_color_list=edge_colors,
        show_camera=True,
        vis_threshold=1.0,
        size = size
    )
    
    viewer.run()

if __name__ == "__main__":
    from dust3r.datasets.waymo import Waymo_Multi_TStride
    from src.helpers import batchify_sample_for_cut3r

    waymo_val_ds = Waymo_Multi_TStride(
        num_views=7,
        resolution=224,
        ROOT="/workspace/raid/jevers/cut3r_processed_waymo/validation",
        temporal_stride=2,
        overlap_step=5,
    )
    
    sample = waymo_val_ds[479]
    batch = batchify_per_view_like_you_want(sample)

    cut3r_model = ARCroco3DStereo.from_pretrained("/workspace/cut3r-forecasting/cut3r_512_dpt_4_64.pth").to("cuda")
    cut3r_model.eval()

    sys.path.append("/workspace/DINO-Foresight")
    from dino_f import Dino_f
    args = torch.load("/workspace/DINO-Foresight/args.pt")
    dinof_model = Dino_f.load_from_checkpoint(ckpt_path, args=args, strict=False, map_location="cpu")
    dinof_model.to(args.device).eval()


    run_inference_with_forecast(cut3r_model, dinof_model, batch, "output", 224)










