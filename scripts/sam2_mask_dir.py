# scripts/sam2_mask_dir.py
import os
import cv2
import numpy as np
from glob import glob
import torch

# --- SAM2 imports ---
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2
from sam2.modeling import build_sam2_model

# Hydra bits for direct-config fallback
from hydra.errors import MissingConfigException
from hydra import initialize_config_dir, compose
import pathlib, sam2

# ========= USER SETTINGS =========
IM_DIR   = "lantern_ds/frames"     # images produced by dataset_prep.py
MASK_DIR = "lantern_ds/masks"      # will be created if missing
CONFIG_NAME = "sam2_hiera_base_plus"      # SAM2 config (YAML stem)
CKPT_PATH   = "sam2_hiera_base_plus.pt"   # checkpoint file in CWD
# A generous box around the lantern + some grass. Tweak to your frames:
BOX = (520, 220, 1120, 980)  # (x0, y0, x1, y1)
# =================================

os.makedirs(MASK_DIR, exist_ok=True)

if not os.path.isfile(CKPT_PATH):
    raise FileNotFoundError(
        f"Checkpoint not found: {CKPT_PATH}\n"
        "Place the SAM2 .pt file in the working directory (same name), "
        "or download via Hugging Face before running this script."
    )

def build_model_robust(config_name: str, ckpt_path: str):
    """
    Try the standard SAM2 builder; if Hydra can't find the config,
    fall back to an explicit config_dir pointing at the installed package YAMLs.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Try standard build first
    try:
        sam = build_sam2(
            config_name=config_name,
            ckpt_path=ckpt_path,
            device=device
        )
        return sam
    except MissingConfigException:
        pass  # fall through to direct-config mode

    # Fallback: point Hydra at the package's configs directory explicitly
    cfg_dir = pathlib.Path(sam2.__file__).parent / "configs"
    if not cfg_dir.exists():
        raise RuntimeError(
            f"SAM2 configs directory not found at {cfg_dir}. "
            "Try reinstalling SAM2 from GitHub."
        )

    with initialize_config_dir(version_base=None, config_dir=str(cfg_dir)):
        cfg = compose(config_name=config_name)

    sam = build_sam2_model(
        cfg=cfg,
        ckpt_path=ckpt_path,
        device=device
    )
    return sam

print("Building SAM2 model…")
sam_model = build_model_robust(CONFIG_NAME, CKPT_PATH)
predictor = SAM2ImagePredictor(sam_model)
print("SAM2 ready.")

# Collect frames (supports .jpg and .png)
frames = sorted([*glob(f"{IM_DIR}/*.jpg"), *glob(f"{IM_DIR}/*.png")])
if not frames:
    raise RuntimeError(f"No images found in {IM_DIR}. Did dataset_prep.py run?")

# Convert BOX to np.array once
BOX_ARR = np.array(BOX)

for p in frames:
    img = cv2.imread(p)
    if img is None:
        print(f"[WARN] Could not read {p}, skipping.")
        continue

    # BGR->RGB for predictor
    predictor.set_image(img[..., ::-1])

    masks, scores, _ = predictor.predict(box=BOX_ARR)
    if masks is None or len(masks) == 0:
        print(f"[WARN] No mask predicted for {os.path.basename(p)}, skipping.")
        continue

    m = masks[int(np.argmax(scores))].astype(np.uint8) * 255

    # Light post-process: keep grass base + smooth edges
    m = cv2.medianBlur(m, 7)
    m = cv2.dilate(m, np.ones((7, 7), np.uint8), 1)

    out_mask = os.path.join(
        MASK_DIR,
        os.path.splitext(os.path.basename(p))[0] + "_mask.png"
    )
    cv2.imwrite(out_mask, m)

print(f"Done. Wrote masks to: {MASK_DIR}")
