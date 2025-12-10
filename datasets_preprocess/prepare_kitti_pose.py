#!/usr/bin/env python

import os
import glob
import numpy as np
import pykitti

# ==== adjust these if needed ====
BASE_KITTI_DIR = '/workspace/raid/jevers/kitti'  # where your date folders (2011_09_26, ...) live
GATHERED_IMG_ROOT = '/workspace/raid/jevers/kitti/depth_selection/val_selection_cropped/image_gathered'
OUTPUT_ROOT = '/workspace/raid/jevers/kitti/depth_selection/val_selection_cropped/cam_poses_gathered'
# ================================

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Example gathered subdir name: 2011_09_26_drive_0002_sync_02
seq_dirs = glob.glob(os.path.join(GATHERED_IMG_ROOT, '*_02'))

for seq_dir in seq_dirs:
    seq_name = os.path.basename(seq_dir)
    print(f'Processing sequence: {seq_name}')

    parts = seq_name.split('_')
    # '2011_09_26_drive_0002_sync_02' -> date='2011_09_26', drive='0002'
    date = '_'.join(parts[:3])   # 2011_09_26
    drive = parts[4]             # 0002

    # Load raw data for this sequence
    raw = pykitti.raw(BASE_KITTI_DIR, date, drive)
    calib = raw.calib

    # IMU -> cam2 transform (pykitti provides this)
    T_cam2_imu = calib.T_cam2_imu
    T_cam2_imu_inv = np.linalg.inv(T_cam2_imu)

    # Output dir for this sequence, mirroring image_gathered
    out_seq_dir = os.path.join(OUTPUT_ROOT, seq_name)
    os.makedirs(out_seq_dir, exist_ok=True)

    img_files = sorted(glob.glob(os.path.join(seq_dir, '*.png')))
    for img_path in img_files:
        fname = os.path.basename(img_path)          # e.g. 0000000005.png
        frame_id = int(os.path.splitext(fname)[0])  # 5

        if frame_id >= len(raw.oxts):
            print(f'  [WARN] frame {frame_id} out of range for {date} {drive}')
            continue

        # 4x4 pose of IMU in world frame
        T_w_imu = raw.oxts[frame_id].T_w_imu

        # 4x4 pose of cam2 in world frame: T_w_cam2 = T_w_imu * inv(T_cam2_imu)
        T_w_cam2 = T_w_imu @ T_cam2_imu_inv

        # Save as .npy next to the image (but under cam_poses_gathered)
        pose_path = os.path.join(out_seq_dir, os.path.splitext(fname)[0] + '.npy')
        np.save(pose_path, T_w_cam2)

        # Optional: print once per sequence
        # print('  saved pose to', pose_path)

print('All poses saved into', OUTPUT_ROOT)