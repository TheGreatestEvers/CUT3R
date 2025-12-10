# eval/foresight/kitti_adapter.py

from pathlib import Path
from typing import List, Tuple

import numpy as np
import cv2


# Subdir names inside the KITTI val_selection_cropped root
RGB_SUBDIR = "image_gathered"
DEPTH_SUBDIR = "groundtruth_depth_gathered"
POSE_SUBDIR = "cam_poses_gathered"


def list_segments(root: str) -> List[Path]:
    """
    Return a sorted list of 'segment' directories for KITTI.

    Here a segment is one of your gathered folders, e.g.
      <root>/image_gathered/2011_09_26_drive_0002_sync_02
    """
    root_path = Path(root) / RGB_SUBDIR
    if not root_path.exists():
        raise FileNotFoundError(f"RGB root not found: {root_path}")

    return sorted([d for d in root_path.iterdir() if d.is_dir()])


def list_front_camera_frames(segment_dir: Path) -> List[Tuple[int, Path]]:
    """
    List all frames (KITTI image_02) in numeric order.

    segment_dir example:
      <root>/image_gathered/2011_09_26_drive_0002_sync_02

    Returns:
        List of (frame_idx, rgb_path)
        where frame_idx is the integer filename (e.g. '0000000005.png' -> 5).
    """
    segment_dir = Path(segment_dir)
    paths = sorted(segment_dir.glob("*.png"))

    frames = []
    for p in paths:
        stem = p.stem  # e.g. '0000000005'
        try:
            frame_idx = int(stem)
        except ValueError:
            continue
        frames.append((frame_idx, p))

    frames.sort(key=lambda x: x[0])
    return frames


def rgb_to_depth_path(rgb_path: Path, kitti_root: Path = None) -> Path:
    """
    Map an RGB PNG path under image_gathered/ to its depth PNG path
    under groundtruth_depth_gathered/ with the same relative segment/filename.
    """
    rgb_path = Path(rgb_path)

    # If user passes a dataset root, use it to rebuild the path:
    if kitti_root is not None:
        kitti_root = Path(kitti_root)
        # relative path from image_gathered/ to the file
        rel = rgb_path.relative_to(kitti_root / RGB_SUBDIR)
        return (kitti_root / DEPTH_SUBDIR / rel).with_suffix(".png")

    # Otherwise, do a simple string replacement
    depth_str = str(rgb_path).replace(f"/{RGB_SUBDIR}/", f"/{DEPTH_SUBDIR}/")
    return Path(depth_str).with_suffix(".png")


def rgb_to_cam_pose_path(rgb_path: Path, kitti_root: Path = None) -> Path:
    """
    Map an RGB PNG path to its cam2world pose .npy file
    under cam_poses_gathered/.

    Not used in your depth-only eval yet, but handy for pose eval later.
    """
    rgb_path = Path(rgb_path)

    if kitti_root is not None:
        kitti_root = Path(kitti_root)
        rel = rgb_path.relative_to(kitti_root / RGB_SUBDIR)
        return (kitti_root / POSE_SUBDIR / rel).with_suffix(".npy")

    pose_str = str(rgb_path).replace(f"/{RGB_SUBDIR}/", f"/{POSE_SUBDIR}/")
    return Path(pose_str).with_suffix(".npy")


def read_depth_kitti_png(depth_path: Path) -> np.ndarray:
    """
    Read KITTI 16-bit depth PNG into [H, W] float32 in meters.

    This matches your earlier depth_read():
        depth = depth_png.astype(np.float) / 256.
        depth[depth_png == 0] = -1.

    and matches depth_evaluation's expectation that invalid depth = -1.
    """
    depth_path = Path(depth_path)
    depth_png = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

    if depth_png is None:
        raise FileNotFoundError(f"Cannot read depth PNG: {depth_path}")

    if depth_png.ndim == 3:
        depth_png = depth_png[..., 0]

    depth = depth_png.astype(np.float32) / 256.0
    depth[depth_png == 0] = -1.0
    return depth


def read_cam_pose_npy(pose_path: Path) -> np.ndarray:
    """
    Read a 4x4 cam2world pose from .npy (as saved before).
    """
    pose_path = Path(pose_path)
    pose = np.load(pose_path)
    if pose.shape != (4, 4):
        raise ValueError(f"Expected (4,4) pose in {pose_path}, got {pose.shape}")
    return pose