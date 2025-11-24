import os
import os.path as osp
import numpy as np
from glob import glob
from PIL import Image

from .base_multi_view_dataset import BaseMultiViewDataset  # adjust import to your project


class Cityscapes_Multi_TStride(BaseMultiViewDataset):
    """
    Cityscapes video dataset with fixed temporal stride and optional overlapping windows,
    similar to Waymo_Multi_TStride.

    - ROOT: path to the Cityscapes video RGB root, the same data_path you used before,
            i.e. something like ".../leftImg8bit_sequence"
            and then ROOT/<split>/<city>/*leftImg8bit.png

    - split: "train" | "val" | "test"

    - temporal_stride: number of raw frames between views inside a sample (e.g. 2)
                       -> logical step between views = temporal_stride

    - overlap_step: how much we slide the starting index for overlapping windows.
                    If None, defaults to temporal_stride (like in the Waymo class).

    - non_overlapping: if True, windows are disjoint: start indices jump by block_len.

    - val_fix_19: if True and split == "val":
        For each (city, seq_id) sequence we build one window whose last view is
        the frame with local index 19 (20th frame) in that sequence, mimicking
        your original CityScapesRGBDataset eval behavior (gt at frames_path[19]).
        This ignores overlap_step / non_overlapping for val.
    """

    def __init__(
        self,
        *args,
        ROOT,
        split="train",
        temporal_stride=2,
        overlap_step=None,
        non_overlapping=False,
        val_fix_19=False,
        **kwargs,
    ):
        self.ROOT = ROOT
        self.split = split  # "train", "val", "test"
        self.temporal_stride = int(temporal_stride)
        assert self.temporal_stride >= 1

        self.overlap_step = (
            int(overlap_step) if overlap_step is not None else self.temporal_stride
        )
        self.non_overlapping = bool(non_overlapping)
        self.val_fix_19 = bool(val_fix_19)

        # optional flags similar to your other datasets
        self.video = True
        self.is_metric = True  # keep if your downstream expects it

        super()._init_(*args, **kwargs)
        self._load_data()

    def __len__(self):
        return len(self.start_img_ids)

    # ------------- index building -------------

    def _load_data(self):
        """
        Build a flat index of all frames and a list of starting positions for windows,
        similar to Waymo_Multi_TStride.
        """
        split_root = osp.join(self.ROOT, self.split)
        assert osp.isdir(split_root), f"Split root not found: {split_root}"

        city_dirs = sorted(
            [d for d in os.listdir(split_root) if osp.isdir(osp.join(split_root, d))]
        )

        offset = 0
        scenes = []          # list[(city_name, seq_id)] per sequence
        sceneids = []        # scene index for each frame (global index)
        images = []          # basename (without extension) for each frame (global)
        start_img_ids = []   # global frame indices used as window starts
        scene_img_list = []  # per scene, list of global frame indices in order

        j = 0  # scene index

        for city_name in city_dirs:
            city_dir = osp.join(split_root, city_name)

            # Gather all PNG frames in this city
            frame_paths = sorted(
                glob(osp.join(city_dir, "*.png"))
            )

            # Group by (city, seq_id) where seq_id is the second token
            # Example filename: aachen_000001_000019_leftImg8bit.png
            seq2frames = {}
            for fpath in frame_paths:
                fname = osp.basename(fpath)[:-4]  # strip ".png"
                tokens = fname.split("_")
                if len(tokens) < 3:
                    # Not a standard cityscapes video name, skip
                    continue

                seq_id = tokens[1]       # e.g., "000001"
                frame_id = tokens[2]     # e.g., "000019"

                key = (city_name, seq_id)
                seq2frames.setdefault(key, []).append((int(frame_id), fname))

            # For each (city, seq_id) sequence, sort frames and create windows
            for (city, seq_id), frame_list in seq2frames.items():
                # frame_list: list[(frame_id_int, fname)]
                frame_list.sort(key=lambda x: x[0])
                if len(frame_list) == 0:
                    continue

                num_imgs = len(frame_list)
                # block_len is the number of raw frames consumed by one sample window
                block_len = (self.num_views - 1) * self.temporal_stride + 1
                if num_imgs < block_len:
                    # too short to form even a single sample
                    continue

                # global frame ids for this sequence
                img_ids = list(np.arange(num_imgs) + offset)

                # register frames globally
                scenes.append((city, seq_id))
                sceneids.extend([j] * num_imgs)
                images.extend([fname for (_, fname) in frame_list])
                scene_img_list.append(img_ids)

                # -------------------
                # START INDICES LOGIC
                # -------------------
                if self.split == "val" and self.val_fix_19:
                    # We want exactly one window per sequence such that
                    # the LAST view corresponds to the local index 19
                    # (20th frame) of this sequence, like frames_path[19].
                    # last_view_idx = start + (num_views-1)*temporal_stride
                    # => start = 19 - (num_views-1)*temporal_stride
                    last_local_idx = 19  # 0-based index -> 20th frame
                    start_local_idx = last_local_idx - (self.num_views - 1) * self.temporal_stride

                    if (
                        start_local_idx >= 0
                        and start_local_idx + block_len - 1 < num_imgs
                    ):
                        start_img_ids.append(img_ids[start_local_idx])
                    # else: sequence too short or not enough frames before index 19
                else:
                    # TRAIN / TEST or val without special 19th-frame behavior
                    if self.non_overlapping:
                        # disjoint windows: 0, block_len, 2*block_len, ...
                        starts_local = list(
                            range(0, num_imgs - block_len + 1, block_len)
                        )
                    else:
                        # overlapping sliding windows
                        step = max(1, self.overlap_step)
                        starts_local = list(
                            range(0, num_imgs - block_len + 1, step)
                        )

                    if len(starts_local) > 0:
                        start_img_ids.extend([img_ids[s] for s in starts_local])

                offset += num_imgs
                j += 1  # next scene

        self.scenes = scenes
        self.sceneids = sceneids
        self.images = images
        self.start_img_ids = start_img_ids
        self.scene_img_list = scene_img_list
        # self.is_video is already set; keep if downstream uses it

    # ------------- sampling -------------

    def _get_views(self, idx, resolution, rng, num_views):
        """
        Build one multi-view sample, similar to Waymo_Multi_TStride._get_views.

        - idx indexes self.start_img_ids.
        - We compute a fixed-stride block starting at that frame.
        - We return a list of num_views dicts with at least img and some metadata.
        """
        # Start from global frame index
        start_id = self.start_img_ids[idx]

        # Scene index from that frame
        scene_idx = self.sceneids[start_id]
        all_image_ids = self.scene_img_list[scene_idx]  # global ids for this scene
        city_name, seq_id = self.scenes[scene_idx]

        # fixed-stride block from the start index
        pos = []
        pos_ref = all_image_ids.index(start_id)
        for k in range(num_views):
            p = pos_ref + k * self.temporal_stride
            assert p < len(all_image_ids), "Block overflows sequence length"
            pos.append(p)

        ordered_video = True

        image_idxs = np.array(all_image_ids)[pos]
        views = []

        split_root = osp.join(self.ROOT, self.split)
        city_dir = osp.join(split_root, city_name)

        for v, view_idx in enumerate(image_idxs):
            scene_id = self.sceneids[view_idx]
            city_name, seq_id = self.scenes[scene_id]
            city_dir = osp.join(split_root, city_name)

            fname = self.images[view_idx]  # e.g. "aachen_000001_000019_leftImg8bit"
            impath = osp.join(city_dir, fname + ".png")

            # ---- load RGB image ----
            img = np.array(Image.open(impath).convert("RGB"))

            # ---- If you want to add depth/segm/normals, you can mirror your old paths here ----
            # e.g. depth_path = impath.replace("leftImg8bit_sequence", "leftImg8bit_sequence_depthv2")...
            depthmap = None  # placeholder; set real depth if you have it

            # Placeholder intrinsics / poses if you don't have them
            # (adjust to your setup; e.g. load from a .npz next to the image)
            intrinsics = np.eye(3, dtype=np.float32)
            camera_pose = np.eye(4, dtype=np.float32)

            # Reuse your existing cropping/resizing logic if defined in BaseMultiViewDataset
            img, depthmap, intrinsics = self._crop_resize_if_necessary(
                img, depthmap, intrinsics, resolution, rng, info=(city_dir, fname)
            )

            img_mask, ray_mask = self.get_img_and_ray_masks(
                self.is_metric, v, rng, p=[0.85, 0.10, 0.05]
            )

            views.append(
                dict(
                    img=img,
                    depthmap=depthmap,
                    camera_pose=camera_pose,
                    camera_intrinsics=intrinsics,
                    dataset="Cityscapes",
                    label=osp.relpath(city_dir, self.ROOT),
                    is_metric=self.is_metric,
                    instance=impath,
                    is_video=ordered_video,
                    quantile=np.array(0.98, dtype=np.float32),
                    img_mask=img_mask,
                    ray_mask=ray_mask,
                    camera_only=False,
                    depth_only=False,
                    single_view=False,
                    reset=False,
                    # you can also mark the reference/19th frame like this:
                    is_ref=(v == num_views - 1),
                )
            )

        return views
