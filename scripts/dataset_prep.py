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
    clean_dir = out / "images"
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

    # Write JPG frames with reasonable quality
    # -q:v 2 is very compressed; use 1 for better quality (1 is highest for JPEG)
    pattern = str(frames_dir / "frame_%05d.jpg")
    run(["ffmpeg", "-y", "-i", args.video, "-vf", f"fps={fps}", "-q:v", "1", pattern])

    # 2) Blur cull + downscale
    kept = []
    total_frames = len(list(frames_dir.glob("*.jpg")))
    print(f"Processing {total_frames} extracted frames...")

    for p in tqdm(sorted(frames_dir.glob("*.jpg")), desc="Filtering frames"):
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

    print(f"Kept {len(kept)}/{total_frames} frames after blur filtering (threshold: {args.min_sharpness})")

    if len(kept) == 0:
        raise ValueError(f"No frames survived blur filtering! Try lowering --min_sharpness (current: {args.min_sharpness})")

    if len(kept) < args.target_frames * 0.5:
        print(f"WARNING: Only {len(kept)} frames survived, much less than target {args.target_frames}. Consider lowering --min_sharpness")

    # 3) Enforce max ~target_frames by evenly subsampling if too many survived
    # Use np.linspace for better temporal distribution
    if len(kept) > args.target_frames:
        original_count = len(kept)
        indices = np.linspace(0, len(kept) - 1, args.target_frames, dtype=int)
        selected = [kept[i] for i in indices]
        for f in kept:
            if f not in selected:
                f.unlink(missing_ok=True)
        kept = selected
        print(f"Subsampled from {original_count} to {len(selected)} frames with even temporal spacing")

    # 4) Train/test split (every 10th frame to test)
    train_txt = split_dir / "train.txt"
    test_txt  = split_dir / "test.txt"
    with open(train_txt, "w") as tr, open(test_txt, "w") as te:
        for i, f in enumerate(sorted(clean_dir.glob("*.jpg"))):
            (te if (i % 10 == 0) else tr).write(f.name + "\n")

    # Count train/test split
    num_train = len([line for line in open(train_txt)])
    num_test = len([line for line in open(test_txt)])

    print(f"\n{'='*60}")
    print(f"✅ Dataset preparation complete!")
    print(f"{'='*60}")
    print(f"Total frames:     {len(kept)}")
    print(f"Training frames:  {num_train}")
    print(f"Test frames:      {num_test}")
    print(f"Output directory: {clean_dir}")
    print(f"Splits at:        {split_dir}/train.txt and test.txt")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
