import os.path as osp
from PIL import Image, ImageDraw
import numpy as np

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