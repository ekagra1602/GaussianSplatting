#!/usr/bin/env python3
"""
Fix PLY export from checkpoint - converts to proper SH format for viewers
"""

import torch
import torch.nn.functional as F
import struct
from pathlib import Path

def save_ply_fixed(checkpoint_path: str, output_path: str):
    """Load checkpoint and save properly formatted PLY."""

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract parameters
    means = checkpoint["means"].numpy()
    scales = torch.exp(checkpoint["scales"]).numpy()
    quats = F.normalize(checkpoint["quats"], dim=-1).numpy()
    opacities = torch.sigmoid(checkpoint["opacities"]).numpy().flatten()

    # Convert RGB to SH DC coefficients (this is the fix!)
    rgb = torch.sigmoid(checkpoint["sh0"]).squeeze(1)  # [N, 3] in [0, 1]
    C0 = 0.28209479177387814  # 0.5 / sqrt(pi)
    sh_dc = ((rgb - 0.5) / C0).numpy()  # Convert to SH space

    N = means.shape[0]
    print(f"Exporting {N} Gaussians...")

    # Create PLY header
    header = f"""ply
format binary_little_endian 1.0
element vertex {N}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""

    # Write PLY file
    with open(output_path, 'wb') as f:
        f.write(header.encode('utf-8'))

        # Write binary data
        for i in range(N):
            # Position
            f.write(struct.pack('fff', means[i, 0], means[i, 1], means[i, 2]))

            # Normals (set to 0)
            f.write(struct.pack('fff', 0.0, 0.0, 0.0))

            # SH DC coefficients (in SH space, not RGB!)
            f.write(struct.pack('fff', sh_dc[i, 0], sh_dc[i, 1], sh_dc[i, 2]))

            # Opacity
            f.write(struct.pack('f', opacities[i]))

            # Scales
            f.write(struct.pack('fff', scales[i, 0], scales[i, 1], scales[i, 2]))

            # Rotation (quaternion)
            f.write(struct.pack('ffff', quats[i, 0], quats[i, 1], quats[i, 2], quats[i, 3]))

    file_size = Path(output_path).stat().st_size / (1024*1024)
    print(f"✅ Saved PLY with {N} Gaussians to {output_path}")
    print(f"   File size: {file_size:.1f} MB")
    print(f"\n📌 View at: https://antimatter15.com/splat/")
    print(f"   Just drag and drop the PLY file!")


if __name__ == "__main__":
    checkpoint_path = "final (2).pt"
    output_path = "final_fixed.ply"

    save_ply_fixed(checkpoint_path, output_path)
