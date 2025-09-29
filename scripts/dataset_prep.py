# scripts/dataset_prep.py
import argparse, os, subprocess, shutil
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

def run(cmd):
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)

def variance_of_laplacian(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def downscale_to_width(img, width=1600):
    h, w = img.shape[:2]
    if w <= width: return img
    scale = width / w
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target_frames", type=int, default=150)
    ap.add_argument("--min_sharpness", type=float, default=80.0)  # increase if too lenient
    ap.add_argument("--width", type=int, default=1600)
    args = ap.parse_args()

    out = Path(args.out)
    frames_dir = out / "frames_raw"
    clean_dir = out / "frames"
    split_dir  = out / "splits"
    for d in [frames_dir, clean_dir, split_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1) Extract frames with ffmpeg at a rate that yields ~target_frames total
    # (use fps filter; see references for ffmpeg frame extraction basics). 
    # We'll probe duration to estimate fps.
    import imageio
    import math
    reader = imageio.get_reader(args.video)
    meta = reader.get_meta_data()
    dur = float(meta.get("duration", 0)) or 1.0
    fps = max(1.0, math.ceil(args.target_frames / dur))
    reader.close()

    # Write PNG/JPG frames
    # Tip: PNG for lossless; JPG q=2 for speed/size. Here we’ll do jpg.
    pattern = str(frames_dir / "frame_%05d.jpg")
    run(["ffmpeg", "-y", "-i", args.video, "-vf", f"fps={fps}", "-q:v", "2", pattern])

    # 2) Blur cull + downscale
    kept = []
    for p in tqdm(sorted(frames_dir.glob("*.jpg"))):
        img = cv2.imread(str(p))
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharp = variance_of_laplacian(gray)
        if sharp < args.min_sharpness:
            continue
        img2 = downscale_to_width(img, args.width)
        out_name = clean_dir / p.name
        cv2.imwrite(str(out_name), img2, [cv2.IMWRITE_JPEG_QUALITY, 95])
        kept.append(out_name)

    # 3) Enforce max ~target_frames by evenly subsampling if too many survived
    if len(kept) > args.target_frames:
        step = len(kept) / args.target_frames
        keep_idxs = {int(i*step) for i in range(args.target_frames)}
        selected = [f for i, f in enumerate(kept) if i in keep_idxs]
        for f in kept:
            if f not in selected:
                f.unlink(missing_ok=True)
        kept = selected

    # 4) Train/test split (every 10th frame to test)
    train_txt = split_dir / "train.txt"
    test_txt  = split_dir / "test.txt"
    with open(train_txt, "w") as tr, open(test_txt, "w") as te:
        for i, f in enumerate(sorted(clean_dir.glob("*.jpg"))):
            (te if (i % 10 == 0) else tr).write(f.name + "\n")

    print(f"Done. Kept {len(kept)} frames at {clean_dir}")
    print(f"Splits at {split_dir}/train.txt and test.txt")

if __name__ == "__main__":
    main()
