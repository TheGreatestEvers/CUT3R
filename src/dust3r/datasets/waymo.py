import os.path as osp
import os
import numpy as np
import sys

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))
import h5py
from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from dust3r.utils.image import imread_cv2


class Waymo_Multi(BaseMultiViewDataset):
    """Dataset of outdoor street scenes, 5 images each time"""

    def __init__(self, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        self.max_interval = 8
        self.video = True
        self.is_metric = True
        super().__init__(*args, **kwargs)
        assert self.split is None
        self._load_data()

    def load_invalid_dict(self, h5_file_path):
        invalid_dict = {}
        with h5py.File(h5_file_path, "r") as h5f:
            for scene in h5f:
                data = h5f[scene]["invalid_pairs"][:]
                invalid_pairs = set(
                    tuple(pair.decode("utf-8").split("_")) for pair in data
                )
                invalid_dict[scene] = invalid_pairs
        return invalid_dict

    def _load_data(self):
        invalid_dict = self.load_invalid_dict(
            os.path.join(self.ROOT, "invalid_files.h5")
        )
        scene_dirs = sorted(
            [
                d
                for d in os.listdir(self.ROOT)
                if os.path.isdir(os.path.join(self.ROOT, d))
            ]
        )
        offset = 0
        scenes = []
        sceneids = []
        images = []
        start_img_ids = []
        scene_img_list = []
        is_video = []
        j = 0

        for scene in scene_dirs:
            scene_dir = osp.join(self.ROOT, scene)
            invalid_pairs = invalid_dict.get(scene, set())
            seq2frames = {}
            for f in os.listdir(scene_dir):
                if not f.endswith(".jpg"):
                    continue
                basename = f[:-4]
                frame_id = basename.split("_")[0]
                seq_id = basename.split("_")[1]
                if seq_id == "5":
                    continue
                if (seq_id, frame_id) in invalid_pairs:
                    continue  # Skip invalid files
                if seq_id not in seq2frames:
                    seq2frames[seq_id] = []
                seq2frames[seq_id].append(frame_id)

            for seq_id, frame_ids in seq2frames.items():
                frame_ids = sorted(frame_ids)
                num_imgs = len(frame_ids)
                img_ids = list(np.arange(num_imgs) + offset)
                cut_off = (
                    self.num_views
                    if not self.allow_repeat
                    else max(self.num_views // 3, 3)
                )
                start_img_ids_ = img_ids[: num_imgs - cut_off + 1]

                if num_imgs < cut_off:
                    print(f"Skipping {scene}_{seq_id}")
                    continue

                scenes.append((scene, seq_id))
                sceneids.extend([j] * num_imgs)
                images.extend(frame_ids)
                start_img_ids.extend(start_img_ids_)
                scene_img_list.append(img_ids)

                offset += num_imgs
                j += 1

        self.scenes = scenes
        self.sceneids = sceneids
        self.images = images
        self.start_img_ids = start_img_ids
        self.scene_img_list = scene_img_list
        self.is_video = is_video

    def __len__(self):
        return len(self.start_img_ids)

    def get_image_num(self):
        return len(self.images)

    def get_stats(self):
        return f"{len(self)} groups of views"

    def _get_views(self, idx, resolution, rng, num_views):
        start_id = self.start_img_ids[idx]
        all_image_ids = self.scene_img_list[self.sceneids[start_id]]
        _, seq_id = self.scenes[self.sceneids[start_id]]
        max_interval = self.max_interval // 2 if seq_id == "4" else self.max_interval
        pos, ordered_video = self.get_seq_from_start_id(
            num_views,
            start_id,
            all_image_ids,
            rng,
            max_interval=max_interval,
            video_prob=0.9,
            fix_interval_prob=0.9,
            block_shuffle=16,
        )
        image_idxs = np.array(all_image_ids)[pos]
        views = []
        ordered_video = True

        views = []

        for v, view_idx in enumerate(image_idxs):
            scene_id = self.sceneids[view_idx]
            scene_dir, seq_id = self.scenes[scene_id]
            scene_dir = osp.join(self.ROOT, scene_dir)
            frame_id = self.images[view_idx]

            impath = f"{frame_id}_{seq_id}"
            image = imread_cv2(osp.join(scene_dir, impath + ".jpg"))
            depthmap = imread_cv2(osp.join(scene_dir, impath + ".exr"))
            camera_params = np.load(osp.join(scene_dir, impath + ".npz"))

            intrinsics = np.float32(camera_params["intrinsics"])
            camera_pose = np.float32(camera_params["cam2world"])

            image, depthmap, intrinsics = self._crop_resize_if_necessary(
                image, depthmap, intrinsics, resolution, rng, info=(scene_dir, impath)
            )

            # generate img mask and raymap mask
            img_mask, ray_mask = self.get_img_and_ray_masks(
                self.is_metric, v, rng, p=[0.85, 0.10, 0.05]
            )

            views.append(
                dict(
                    img=image,
                    depthmap=depthmap,
                    camera_pose=camera_pose,  # cam2world
                    camera_intrinsics=intrinsics,
                    dataset="Waymo",
                    label=osp.relpath(scene_dir, self.ROOT),
                    is_metric=self.is_metric,
                    instance=osp.join(scene_dir, impath + ".jpg"),
                    is_video=ordered_video,
                    quantile=np.array(0.98, dtype=np.float32),
                    img_mask=img_mask,
                    ray_mask=ray_mask,
                    camera_only=False,
                    depth_only=False,
                    single_view=False,
                    reset=False,
                )
            )

        return views



class Waymo_Multi_TStride(BaseMultiViewDataset):
    """
    Waymo dataset with fixed temporal stride and optional overlapping windows.

    - temporal_stride: number of raw frames between views inside a sample (e.g., 2)
    - allow_overlap: if True, we generate sliding windows with partial overlap
                     (start positions step by overlap_step, default=temporal_stride)
    - non_overlapping: if True, start positions jump by block_len so samples do not overlap
    - Keeps the same output contract as your original class (jpg/exr/npz)
    """

    def __init__(
        self,
        *args,
        ROOT,
        ignore_invalid=True,
        temporal_stride=2,
        overlap_step=None,          # if None, defaults to temporal_stride
        non_overlapping=False,      # set True to force disjoint windows
        **kwargs
    ):
        self.ROOT = ROOT
        self.ignore_invalid = ignore_invalid
        self.temporal_stride = int(temporal_stride)
        assert self.temporal_stride >= 1
        self.overlap_step = int(overlap_step) if overlap_step is not None else self.temporal_stride
        self.non_overlapping = bool(non_overlapping)

        # keep flags from your original
        self.max_interval = 8
        self.video = True
        self.is_metric = True

        super().__init__(*args, **kwargs)
        assert self.split is None
        self._load_data()

    def __len__(self):
        return len(self.start_img_ids)

    # ---- invalid list handling ----
    def load_invalid_dict(self, h5_file_path):
        if self.ignore_invalid or (not osp.isfile(h5_file_path)):
            return {}  # nothing filtered
        import h5py
        invalid_dict = {}
        with h5py.File(h5_file_path, "r") as h5f:
            for scene in h5f:
                data = h5f[scene]["invalid_pairs"][:]
                invalid_pairs = set(tuple(pair.decode("utf-8").split("_")) for pair in data)
                invalid_dict[scene] = invalid_pairs
        return invalid_dict

    # ---- index building ----
    def _load_data(self):
        invalid_dict = self.load_invalid_dict(osp.join(self.ROOT, "invalid_files.h5"))
        scene_dirs = sorted([d for d in os.listdir(self.ROOT) if osp.isdir(osp.join(self.ROOT, d))])

        offset = 0
        scenes, sceneids, images, start_img_ids, scene_img_list = [], [], [], [], []
        is_video = []
        j = 0

        for scene in scene_dirs:
            scene_dir = osp.join(self.ROOT, scene)
            invalid_pairs = invalid_dict.get(scene, set())
            seq2frames = {}

            # gather frames by camera id; keep only seq_id == "1" to mirror your filter
            for f in os.listdir(scene_dir):
                if not f.endswith(".jpg"):
                    continue
                basename = f[:-4]  # '00012_1'
                frame_id, seq_id = basename.split("_")
                if seq_id != "1":
                    continue
                if (seq_id, frame_id) in invalid_pairs:
                    continue
                seq2frames.setdefault(seq_id, []).append(frame_id)

            for seq_id, frame_ids in seq2frames.items():
                frame_ids = sorted(frame_ids, key=lambda x: int(x))  # numeric sort
                num_imgs = len(frame_ids)
                if num_imgs == 0:
                    continue

                # total raw frames consumed by one sample window
                block_len = (self.num_views - 1) * self.temporal_stride + 1
                if num_imgs < block_len:
                    # too short to form one window
                    continue

                img_ids = list(np.arange(num_imgs) + offset)

                if self.non_overlapping:
                    # disjoint windows: 0, block_len, 2*block_len, ...
                    starts = list(range(0, num_imgs - block_len + 1, block_len))
                else:
                    # overlapping sliding windows
                    # step controls how much the windows slide; default equals temporal_stride
                    step = max(1, self.overlap_step)
                    starts = list(range(0, num_imgs - block_len + 1, step))

                if len(starts) == 0:
                    continue

                scenes.append((scene, seq_id))
                sceneids.extend([j] * num_imgs)
                images.extend(frame_ids)
                start_img_ids.extend([img_ids[s] for s in starts])
                scene_img_list.append(img_ids)

                offset += num_imgs
                j += 1

        self.scenes = scenes
        self.sceneids = sceneids
        self.images = images
        self.start_img_ids = start_img_ids
        self.scene_img_list = scene_img_list
        self.is_video = is_video  # unused downstream but kept for parity

    # ---- sampling ----
    def _get_views(self, idx, resolution, rng, num_views):
        start_id = self.start_img_ids[idx]
        scene_idx = self.sceneids[start_id]
        all_image_ids = self.scene_img_list[scene_idx]
        scene_dir_name, seq_id = self.scenes[scene_idx]

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

        for v, view_idx in enumerate(image_idxs):
            scene_id = self.sceneids[view_idx]
            scene_dir_name, seq_id = self.scenes[scene_id]
            scene_dir = osp.join(self.ROOT, scene_dir_name)
            frame_id = self.images[view_idx]

            impath = f"{frame_id}_{seq_id}"
            image = imread_cv2(osp.join(scene_dir, impath + ".jpg"))
            depthmap = imread_cv2(osp.join(scene_dir, impath + ".exr"))  # zeros, created in preprocessing
            camera_params = np.load(osp.join(scene_dir, impath + ".npz"))
            intrinsics = np.float32(camera_params["intrinsics"])
            camera_pose = np.float32(camera_params["cam2world"])

            image, depthmap, intrinsics = self._crop_resize_if_necessary(
                image, depthmap, intrinsics, resolution, rng, info=(scene_dir, impath)
            )

            img_mask, ray_mask = self.get_img_and_ray_masks(self.is_metric, v, rng, p=[0.85, 0.10, 0.05])

            views.append(
                dict(
                    img=image,
                    depthmap=depthmap,
                    camera_pose=camera_pose,
                    camera_intrinsics=intrinsics,
                    dataset="Waymo",
                    label=osp.relpath(scene_dir, self.ROOT),
                    is_metric=self.is_metric,
                    instance=osp.join(scene_dir, impath + ".jpg"),
                    is_video=ordered_video,
                    quantile=np.array(0.98, dtype=np.float32),
                    img_mask=img_mask,
                    ray_mask=ray_mask,
                    camera_only=False,
                    depth_only=False,
                    single_view=False,
                    reset=False,
                )
            )

        return views
