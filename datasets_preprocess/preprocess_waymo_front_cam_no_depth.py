#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Preprocessing code for the WayMo Open dataset
# dataset at https://github.com/waymo-research/waymo-open-dataset
# 1) Accept the license
# 2) download all training/*.tfrecord files from Perception Dataset, version 1.4.2
# 3) put all .tfrecord files in '/path/to/waymo_dir'
# 4) install the waymo_open_dataset package with
#    python3 -m pip install gcsfs waymo-open-dataset-tf-2-12-0==1.6.4
# 5) execute this script as:
#    python preprocess_waymo.py --waymo_dir /path/to/waymo_dir \
#       [--front_only] [--dummy_depth] [--workers N]
# --------------------------------------------------------
import sys
import os
import os.path as osp
import shutil
import json
from tqdm import tqdm
import PIL.Image
import numpy as np

# Match original env usage (just enable EXR for OpenCV)
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

import tensorflow.compat.v1 as tf
tf.enable_eager_execution()

# (Optional but recommended) limit TF threads per process to avoid oversubscription
try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
except Exception:
    # Older TF versions may not have threading config; ignore
    pass

import path_to_root  # noqa
from src.dust3r.utils.geometry import geotrf, inv
from src.dust3r.utils.image import imread_cv2
from src.dust3r.utils.parallel import parallel_processes as parallel_map
from datasets_preprocess.utils import cropping
from src.dust3r.viz import show_raw_pointcloud


def get_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--waymo_dir", required=True)
    parser.add_argument("--output_dir", default="data/waymo_processed")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--front_only",
        action="store_true",
        help="Only process the FRONT camera (CameraName.FRONT / id 1)",
    )
    parser.add_argument(
        "--dummy_depth",
        action="store_true",
        help="Write zero depthmaps instead of real LiDAR-derived depth "
             "and SKIP LiDAR/3D processing in the extraction stage.",
    )
    return parser


def main(waymo_root, output_dir, workers=1, front_only=False, dummy_depth=False):
    # First extract raw frames (and optionally 3D) to tmp/
    extract_frames(
        waymo_root,
        output_dir,
        workers=workers,
        front_only=front_only,
        dummy_depth=dummy_depth,
    )
    # Then crop + generate final RGB/EXR/NPZ
    make_crops(output_dir, workers=workers, dummy_depth=dummy_depth)
    # No precomputed_pairs check (by request)
    shutil.rmtree(osp.join(output_dir, "tmp"))
    print("Done! all data generated at", output_dir)


def _list_sequences(db_root):
    print(">> Looking for sequences in", db_root)
    res = sorted(f for f in os.listdir(db_root) if f.endswith(".tfrecord"))
    print(f"    found {len(res)} sequences")
    return res


def extract_frames(db_root, output_dir, workers=8, front_only=False, dummy_depth=False):
    sequences = _list_sequences(db_root)
    output_tmp = osp.join(output_dir, "tmp")
    print(">> outputing result to", output_tmp)
    args = [(db_root, output_tmp, seq, front_only, dummy_depth) for seq in sequences]
    parallel_map(process_one_seq, args, star_args=True, workers=workers)


def process_one_seq(db_root, output_dir, seq, front_only=False, dummy_depth=False):
    out_dir = osp.join(output_dir, seq)
    os.makedirs(out_dir, exist_ok=True)
    calib_path = osp.join(out_dir, "calib.json")
    if osp.isfile(calib_path):
        return

    try:
        with tf.device("/CPU:0"):
            calib = extract_frames_one_seq(
                osp.join(db_root, seq),
                out_dir,
                front_only=front_only,
                dummy_depth=dummy_depth,
            )
    except RuntimeError:
        print(f"/!\\ Error with sequence {seq} /!\\", file=sys.stderr)
        return  # nothing is saved

    with open(calib_path, "w") as f:
        json.dump(calib, f)


def extract_frames_one_seq(filename, out_dir, front_only=False, dummy_depth=False):
    """
    Stream frames to disk (no big in-RAM 'frames' list).
    Matches original per-file naming/contents.

    When dummy_depth=True:
      - we SKIP LiDAR / 3D processing completely
      - we only save pose + timestamp per image (no pixels / pts3d)
      - depthmaps later are dummy zeros
    """
    from waymo_open_dataset import dataset_pb2 as open_dataset
    from waymo_open_dataset.utils import frame_utils

    print(">> Opening", filename)
    dataset = tf.data.TFRecordDataset(filename, compression_type="")

    calib = None

    for f_idx, data in enumerate(tqdm(dataset, leave=False)):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        # once in a sequence, read camera calibration info
        if calib is None:
            calib = []
            for cam in frame.context.camera_calibrations:
                calib.append(
                    (
                        cam.name,
                        dict(
                            width=cam.width,
                            height=cam.height,
                            intrinsics=list(cam.intrinsic),
                            extrinsics=list(cam.extrinsic.transform),
                        ),
                    )
                )

        # Only compute LiDAR->pointcloud if we actually need 3D
        if not dummy_depth:
            content = frame_utils.parse_range_image_and_camera_projection(frame)
            range_images, camera_projections, _, range_image_top_pose = content

            # convert LIDAR to pointcloud
            points, cp_points = frame_utils.convert_range_image_to_point_cloud(
                frame, range_images, camera_projections, range_image_top_pose
            )

            # 3d points in vehicle frame.
            points_all = np.concatenate(points, axis=0)
            cp_points_all = np.concatenate(cp_points, axis=0)

        # Save each camera view immediately (match original filenames)
        for image in frame.images:
            # Only keep FRONT camera if requested
            # FRONT is CameraName.FRONT == 1 in Waymo
            if front_only and image.name != open_dataset.CameraName.FRONT:
                continue

            pose = np.asarray(image.pose.transform).reshape(4, 4)
            timestamp = image.pose_timestamp
            rgb = tf.image.decode_jpeg(image.image).numpy()

            stem = f"{f_idx:05d}_{image.name}"  # e.g., 00012_1
            PIL.Image.fromarray(rgb).save(osp.join(out_dir, f"{stem}.jpg"))

            if dummy_depth:
                # No 3D: only save what cropping really needs later
                np.savez(
                    osp.join(out_dir, f"{stem}.npz"),
                    pose=pose,
                    timestamp=timestamp,
                )
            else:
                # select relevant 3D points for this view using NumPy mask
                cam_id = image.name
                mask = (cp_points_all[:, 0] == cam_id)  # boolean mask

                cam_points = cp_points_all[mask]
                pix = cam_points[:, 1:3].round().astype(np.int16)
                pts3d = points_all[mask]

                np.savez(
                    osp.join(out_dir, f"{stem}.npz"),
                    pose=pose,
                    pixels=pix,
                    pts3d=pts3d,
                    timestamp=timestamp,
                )

        # This block never runs (kept for compatibility with original)
        if not "show full point cloud":
            show_raw_pointcloud(
                [v for v in [points_all]] if not dummy_depth else [],
                [tf.image.decode_jpeg(img.image).numpy() for img in frame.images],
            )

        # free large per-frame variables promptly
        if not dummy_depth:
            del points, cp_points, points_all, cp_points_all

    return calib


def make_crops(output_dir, workers=16, dummy_depth=False, **kw):
    tmp_dir = osp.join(output_dir, "tmp")
    sequences = _list_sequences(tmp_dir)
    args = [(tmp_dir, output_dir, seq, dummy_depth) for seq in sequences]
    parallel_map(crop_one_seq, args, star_args=True, workers=workers, front_num=0)


def crop_one_seq(input_dir, output_dir, seq, dummy_depth=False, resolution=512):
    seq_dir = osp.join(input_dir, seq)
    out_dir = osp.join(output_dir, seq)
    if osp.isfile(osp.join(out_dir, "00100_1.jpg")):
        return
    os.makedirs(out_dir, exist_ok=True)

    # load calibration file
    try:
        with open(osp.join(seq_dir, "calib.json")) as f:
            calib = json.load(f)
    except IOError:
        print(f"/!\\ Error: Missing calib.json in sequence {seq} /!\\", file=sys.stderr)
        return

    axes_transformation = np.array(
        [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
    )

    cam_K = {}
    cam_distortion = {}
    cam_res = {}
    cam_to_car = {}
    for cam_idx, cam_info in calib:
        cam_idx = str(cam_idx)
        cam_res[cam_idx] = (W, H) = (cam_info["width"], cam_info["height"])
        f1, f2, cx, cy, k1, k2, p1, p2, k3 = cam_info["intrinsics"]
        cam_K[cam_idx] = np.asarray([(f1, 0, cx), (0, f2, cy), (0, 0, 1)])
        cam_distortion[cam_idx] = np.asarray([k1, k2, p1, p2, k3])
        cam_to_car[cam_idx] = np.asarray(cam_info["extrinsics"]).reshape(4, 4)

    # IMPORTANT: match original stem behavior (keep trailing '.')
    frames = sorted(f[:-3] for f in os.listdir(seq_dir) if f.endswith(".jpg"))

    # from dust3r.viz import SceneViz
    # viz = SceneViz()

    for frame in tqdm(frames, leave=False):
        cam_idx = frame[-2]  # expects "..._<digit>."
        assert cam_idx in "12345", f"bad {cam_idx=} in {frame=}"
        data = np.load(osp.join(seq_dir, frame + "npz"))
        car_to_world = data["pose"]
        W, H = cam_res[cam_idx]

        # load image
        image = imread_cv2(osp.join(seq_dir, frame + "jpg"))

        # downscale image
        output_resolution = (resolution, 1) if W > H else (1, resolution)
        image, _, intrinsics2 = cropping.rescale_image_depthmap(
            image, None, cam_K[cam_idx], output_resolution
        )
        image.save(osp.join(out_dir, frame + "jpg"), quality=80)

        # save depth as EXR (match original behavior)
        W2, H2 = image.size

        if dummy_depth:
            # Just write zeros in the right shape / dtype
            depthmap = np.zeros((H2, W2), dtype=np.float32)
        else:
            # Real depth computation from pre-saved 3D points
            pos2d = data["pixels"].round().astype(np.uint16)
            x, y = pos2d.T
            pts3d = data["pts3d"]
            pts3d = geotrf(axes_transformation @ inv(cam_to_car[cam_idx]), pts3d)
            # X=LEFT_RIGHT y=ALTITUDE z=DEPTH

            depthmap = np.zeros((H2, W2), dtype=np.float32)
            pos2d_scaled = (
                geotrf(intrinsics2 @ inv(cam_K[cam_idx]), pos2d)
                .round()
                .astype(np.int16)
            )
            x2, y2 = pos2d_scaled.T
            depthmap[
                y2.clip(min=0, max=H2 - 1),
                x2.clip(min=0, max=W2 - 1),
            ] = pts3d[:, 2]

        cv2.imwrite(osp.join(out_dir, frame + "exr"), depthmap)

        # save camera parameters
        cam2world = car_to_world @ cam_to_car[cam_idx] @ inv(axes_transformation)
        np.savez(
            osp.join(out_dir, frame + "npz"),
            intrinsics=intrinsics2,
            cam2world=cam2world,
            distortion=cam_distortion[cam_idx],
        )

        # viz.add_rgbd(np.asarray(image), depthmap, intrinsics2, cam2world)
    # viz.show()


if __name__ == "__main__":
    # keep spawn to avoid TF/OpenCV fork issues in some environments (harmless otherwise)
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = get_parser()
    args = parser.parse_args()
    main(
        args.waymo_dir,
        args.output_dir,
        workers=args.workers,
        front_only=args.front_only,
        dummy_depth=args.dummy_depth,
    )