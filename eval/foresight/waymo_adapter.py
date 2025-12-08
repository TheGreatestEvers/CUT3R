import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import cv2


def list_segments(root: str) -> List[Path]:
    """
    Return a sorted list of segment directories under the Waymo root.
    Each segment dir contains ~200 frames and associated depth / camera files.
    """
    root_path = Path(root)
    return sorted([d for d in root_path.iterdir() if d.is_dir()])


def list_front_camera_frames(segment_dir: Path) -> List[Tuple[int, Path]]:
    """
    List all front-camera frames (*_1.png) in numeric order.

    Returns:
        List of (frame_idx, rgb_path)
        where frame_idx is the integer prefix (e.g. '00023_1.png' -> 23).
    """
    segment_dir = Path(segment_dir)
    paths = sorted(segment_dir.glob("*_1.jpg"))

    frames = []
    for p in paths:
        stem = p.stem  # e.g. '00023_1'
        base_idx_str = stem.split("_")[0]
        try:
            frame_idx = int(base_idx_str)
        except ValueError:
            continue
        frames.append((frame_idx, p))

    frames.sort(key=lambda x: x[0])
    return frames


def rgb_to_depth_path(rgb_path: Path) -> Path:
    """
    Map an RGB PNG path to its depth EXR path.
    Adjust this if your naming convention differs.
    """
    rgb_path = Path(rgb_path)
    return rgb_path.with_suffix(".exr")


def rgb_to_cam_params_path(rgb_path: Path) -> Path:
    """
    Map an RGB PNG path to its camera-params NPZ path.
    Assumes same basename + '.npz'. Adjust if needed.
    """
    rgb_path = Path(rgb_path)
    return rgb_path.with_suffix(".npz")


def read_depth_exr(depth_path: Path) -> np.ndarray:
    """
    Read a Waymo EXR depth map into a numpy array [H, W] float32.
    Assumes depth in meters stored as floats.

    Any depth <= 0 is marked invalid with -1.0.
    """
    depth_path = Path(depth_path)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

    if depth is None:
        raise FileNotFoundError(f"Cannot read depth EXR: {depth_path}")

    if depth.ndim == 3:
        depth = depth[..., 0]

    depth = depth.astype(np.float32)
    depth[depth <= 0] = -1.0
    return depth


def read_camera_params_npz(cam_path: Path):
    """
    Read intrinsics and cam2world from NPZ file with keys
    'intrinsics' and 'cam2world'. For future pose eval.
    """
    cam_path = Path(cam_path)
    data = np.load(cam_path)
    intrinsics = data["intrinsics"]
    cam2world = data["cam2world"]
    return intrinsics, cam2world