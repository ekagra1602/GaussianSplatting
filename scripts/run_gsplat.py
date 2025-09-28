#!/usr/bin/env python3
import argparse
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to COLMAP-format dataset")
    ap.add_argument("--iters", type=int, default=2000)
    args = ap.parse_args()
    # TODO: integrate gsplat / nerfstudio training call
    print(f"[Day1] Would train gsplat on {args.data} for {args.iters} iters")
if __name__ == "__main__":
    main()
