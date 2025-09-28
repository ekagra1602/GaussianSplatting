#!/usr/bin/env python3
import argparse, os
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # TODO: wire up real VGGT inference here and write COLMAP binaries
    # cameras.bin, images.bin, points3D.bin
    print("VGGT stub ran. Write COLMAP files to:", args.out_dir)

if __name__ == "__main__":
    main()
