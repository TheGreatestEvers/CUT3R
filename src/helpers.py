import os.path as osp
from PIL import Image, ImageDraw
import numpy as np
import torch
from collections.abc import Mapping

def save_waymo_dataset_sample_gif(dataset, idx, out_path="waymo_sample.gif", max_side=512, fps=4, annotate=True):
    """
    Builds a GIF from dataset[idx] using the raw JPGs at view['instance'].
    This verifies frame order + camera selection without touching your transforms.

    Args:
        dataset: your Waymo dataset instance
        idx: sample index to visualize
        out_path: where to save the GIF
        max_side: resize longer side to this (keeps aspect)
        fps: frames per second for the GIF
        annotate: draw filename (e.g., '00012_1.jpg') on each frame
    """
    views = dataset[idx]           # uses your sampling (non-overlap or not)
    frames = []

    for v in views:
        img_path = v["instance"]   # full path to the original JPG (set by your loader)
        im = Image.open(img_path).convert("RGB")

        # resize to a manageable size (preserve aspect)
        w, h = im.size
        scale = max_side / max(w, h)
        if scale < 1.0:
            im = im.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

        if annotate:
            draw = ImageDraw.Draw(im)
            label = osp.basename(img_path)  # e.g. '00012_1.jpg'
            # simple text box
            pad = 4
            tw, th = draw.textlength(label), 12  # basic estimate; good enough for debug
            draw.rectangle([0, 0, tw + 2*pad, th + 2*pad], fill=(0, 0, 0))
            draw.text((pad, pad), label, fill=(255, 255, 255))

        frames.append(im)

    # duration per frame in ms
    duration_ms = int(1000 / max(fps, 1))
    if len(frames) == 1:
        frames[0].save(out_path)
    else:
        frames[0].save(out_path, save_all=True, append_images=frames[1:], loop=0, duration=duration_ms)

    print(f"✅ Saved GIF with {len(frames)} frames to: {out_path}")




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
            return [ _to_torch_with_batch(e) for e in x ]
        # Otherwise elementwise recurse, keep tuple
        return tuple(_to_torch_with_batch(e) for e in x)
    if isinstance(x, list):
        # Keep list structure but batchify elements
        if len(x) > 0 and isinstance(x[0], str):
            # list of strings: assume already per-batch; keep as is
            return x
        return [ _to_torch_with_batch(e) for e in x ]
    # 5) dicts: recurse
    if isinstance(x, Mapping):
        return {k: _to_torch_with_batch(v) for k, v in x.items()}
    # 6) fallback: wrap as single-item list
    return [x]

def batchify_sample_for_cut3r(views, inplace=False):
    """
    Input:  views = List[Dict] from your dataset.
    Output: List[Dict] with the SAME length/structure, but:
        - torch.Tensor -> (1, …)
        - np.ndarray   -> torch.Tensor with (1, …)
        - scalars/bools -> torch.Size([1])
        - strings/paths -> ['...']
        - tuples of scalars (e.g., idx) -> [tensor([..]), tensor([..]), ...]
        - complex tuples/lists (e.g., corres) -> elementwise torch + batch dim
    """
    assert isinstance(views, (list, tuple)) and len(views) > 0 and isinstance(views[0], Mapping)

    if inplace:
        for view in views:
            for k, v in list(view.items()):
                view[k] = _to_torch_with_batch(v)
        return views

    new_views = []
    for view in views:
        new_views.append({k: _to_torch_with_batch(v) for k, v in view.items()})
    return new_views