"""
src/baseline/train_gsplat.py
Wrapper for gsplat training with preset configs
"""
import subprocess
import argparse
from pathlib import Path

def train_gsplat(
    data_dir: str,
    result_dir: str,
    data_factor: int = 1,
    max_steps: int = 10000,
    sh_degree: int = 3,
):
    """Train Gaussian Splatting model using gsplat"""
    
    # Validate paths
    data_dir = Path(data_dir)
    assert (data_dir / "images").exists(), f"No images/ in {data_dir}"
    assert (data_dir / "sparse" / "0").exists(), f"No sparse/0/ in {data_dir}"
    
    # Build command
    cmd = [
        "python", "examples/simple_trainer.py", "default",
        "--data_dir", str(data_dir),
        "--result_dir", str(result_dir),
        "--data_factor", str(data_factor),
        "--max_steps", str(max_steps),
        "--sh_degree", str(sh_degree),
        "--save_ply",
        "--ssim_lambda", "0.2",
    ]
    
    print(f"🚀 Starting training...")
    print(f"   Data: {data_dir}")
    print(f"   Output: {result_dir}")
    print(f"   Steps: {max_steps}")
    
    subprocess.run(cmd, check=True)
    print(f"\n✅ Training complete!")
    print(f"   Results: {result_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--data_factor", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=10000)
    args = parser.parse_args()
    
    train_gsplat(**vars(args))