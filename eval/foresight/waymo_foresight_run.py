import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Make sure repo root is on the path (adjust depth if needed)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append("/workspace/CUT3R/src")
sys.path.append("/workspace/CUT3R/eval")

from dust3r.utils.image import load_images_for_eval as load_images
from dust3r.model import ARCroco3DStereo

from eval.foresight.waymo_adapter import (
    list_segments,
    list_front_camera_frames,
)

sys.path.append("/workspace/DINO-Foresight")
from src.dino_f import Dino_f


def parse_args():
    parser = argparse.ArgumentParser()

    # Dataset & model
    parser.add_argument(
        "--waymo_root",
        type=str,
        default="/workspace/raid/jevers/cut3r_processed_waymo/validation_full_res_depth/validation",
        help="Waymo root directory containing segment subfolders",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/workspace/raid/jevers/waymo_outputs",
        help="Directory where predicted sequences + metadata will be saved",
    )

    # Sequence config
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Image size for model input (as in your normal pipeline)",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=7,
        help="Total number of time steps per sequence",
    )
    parser.add_argument(
        "--context_len",
        type=int,
        default=4,
        help="Number of context frames (first part of each sequence)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=2,
        help="Temporal stride between frames in a sequence",
    )

    return parser.parse_args()


def make_sequences(frame_list, seq_len, stride):
    """
    Given a list of (frame_idx, path) for a segment, generate sequences.

    Sequence defined as frames at indices:
        start_frame_idx + 0*stride,
        start_frame_idx + 1*stride,
        ...
        start_frame_idx + (seq_len-1)*stride

    Only keep sequences where all those indices exist.
    """
    sequences = []
    if len(frame_list) == 0:
        return sequences

    idx_to_pos = {frame_idx: pos for pos, (frame_idx, _) in enumerate(frame_list)}

    min_frame = frame_list[0][0]
    max_frame = frame_list[-1][0]

    for start_frame_idx in range(min_frame, max_frame + 1):
        if start_frame_idx not in idx_to_pos:
            continue

        abs_indices = [start_frame_idx + k * stride for k in range(seq_len)]
        if any(fi not in idx_to_pos for fi in abs_indices):
            continue

        seq = [frame_list[idx_to_pos[fi]] for fi in abs_indices]
        sequences.append(seq)

    return sequences


def prepare_views(img_paths, size, device):
    """
    Prepare CUT3R-style 'views' for dust3r.inference / ARCroco3DStereo.

    This matches the format they use in their eval code:
    each view has img, ray_map, true_shape, etc.
    """
    images = load_images(img_paths, size=size, crop=True)
    views = []

    for i, img_dict in enumerate(images):
        img = img_dict["img"].to(device)
        H, W = img.shape[-2], img.shape[-1]

        view = {
            "img": img,
            "ray_map": torch.full(
                (img.shape[0], 6, H, W),
                torch.nan,
                device=device,
            ),
            "true_shape": torch.from_numpy(img_dict["true_shape"]).to(device),
            "idx": i,
            "instance": str(i),
            "camera_pose": torch.from_numpy(
                np.eye(4, dtype=np.float32)
            ).unsqueeze(0).to(device),
            "img_mask": torch.tensor(True, device=device).unsqueeze(0),
            "ray_mask": torch.tensor(False, device=device).unsqueeze(0),
            "update": torch.tensor(True, device=device).unsqueeze(0),
            "reset": torch.tensor(False, device=device).unsqueeze(0),
        }
        views.append(view)

    return views


def extract_depth_from_outputs(outputs):
    """
    Given ARCroco3DStereo outputs, extract depth tensor [T, H, W].
    Assumes outputs['pred'] is a list of dicts with 'pts3d_in_self_view'.
    """
    pts3ds_self = [pred["pts3d_in_self_view"].cpu() for pred in outputs["pred"]]
    pts3ds_self = torch.cat(pts3ds_self, dim=0)  # [T, H, W, 3]
    depth = pts3ds_self[..., -1]  # z coordinate
    return depth  # [T, H, W]


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Simple single-device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load models
    print("Loading CUT3R model...")
    cut3r_model = ARCroco3DStereo.from_pretrained(
        "/workspace/CUT3R/cut3r_512_dpt_4_64.pth"
    ).to(device)
    cut3r_model.eval()

    print("Loading Dino_f model...")
    args_dinof = torch.load("/workspace/DINO-Foresight/args.pt")
    dinof_model = Dino_f.load_from_checkpoint(
        "/workspace/epoch=87-step=59136-val_loss=0.00000.ckpt",
        args=args_dinof,
        strict=False,
        map_location=device,
    )
    dinof_model.to(device).eval()

    # List all segments (single process)
    segments = list_segments(args.waymo_root)
    print(f"Found {len(segments)} segments.")

    for seg_dir in tqdm(segments, desc="Segments"):
        seg_name = os.path.basename(seg_dir)
        frames = list_front_camera_frames(seg_dir)
        if len(frames) == 0:
            print(f"Warning: no frames found for segment {seg_name}")
            continue

        sequences = make_sequences(
            frame_list=frames, seq_len=args.seq_len, stride=args.stride
        )
        if len(sequences) == 0:
            print(f"Warning: no valid sequences in segment {seg_name}")
            continue

        seg_out_dir = Path(args.output_dir) / seg_name
        seg_out_dir.mkdir(parents=True, exist_ok=True)

        for seq in tqdm(sequences, desc=f"Sequences in {seg_name}", leave=False):
            # seq: list of (frame_idx, rgb_path)
            start_frame_idx = seq[0][0]
            img_paths = [str(p) for (_, p) in seq]

            # Prepare views for CUT3R
            views = prepare_views(img_paths, size=args.img_size, device=device)

            print(views[0]["img"].shape)
            assert False

            with torch.no_grad():
                output = cut3r_model.forward_with_forecaster(
                    views, dinof_model, context_len=args.context_len
                )

            preds, batch = output.ress, output.views
            outputs = dict(views=batch, pred=preds)

            # Build a dict that includes metadata + full outputs
            save_dict = {
                "segment": seg_name,
                "start_frame_idx": start_frame_idx,
                "stride": args.stride,
                "seq_len": args.seq_len,
                "context_len": args.context_len,
                "img_paths": img_paths,
                "outputs": outputs,  # full inference result
            }

            out_path = seg_out_dir / f"seq_{start_frame_idx:05d}.pt"
            torch.save(save_dict, out_path)

    print("Done.")


if __name__ == "__main__":
    main()