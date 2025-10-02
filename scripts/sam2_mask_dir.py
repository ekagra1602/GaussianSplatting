# scripts/sam2_mask_dir.py
import os, cv2, numpy as np
from glob import glob
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2
from hydra.errors import MissingConfigException

# ====== SETTINGS ======
IM_DIR    = "lantern_ds/frames"         # from dataset_prep.py
MASK_DIR  = "lantern_ds/masks"
CONFIG    = "sam2_hiera_base_plus"      # YAML name (no .pt)
CKPT_FILE = "sam2_hiera_base_plus.pt"   # must exist in CWD
BOX       = (520, 220, 1120, 980)       # (x0,y0,x1,y1) — adjust once if needed
# ======================

os.makedirs(MASK_DIR, exist_ok=True)

if not os.path.isfile(CKPT_FILE):
    raise FileNotFoundError(
        f"Checkpoint not found: {CKPT_FILE}. "
        "Download it first (same filename) into the working directory."
    )

print("Building SAM2 model…")
try:
    sam = build_sam2(
        config_name=CONFIG,
        ckpt_path=CKPT_FILE,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
except MissingConfigException as e:
    raise RuntimeError(
        "SAM2 config not found by Hydra. Reinstall SAM2 from GitHub:\n"
        "  pip uninstall -y sam-2 && pip install -q git+https://github.com/facebookresearch/segment-anything-2.git@main\n"
        "Then rerun this script."
    ) from e

predictor = SAM2ImagePredictor(sam)
print("SAM2 ready.")

frames = sorted([*glob(f"{IM_DIR}/*.jpg"), *glob(f"{IM_DIR}/*.png")])
if not frames:
    raise RuntimeError(f"No images found in {IM_DIR}. Did dataset_prep.py run?")

BOX_ARR = np.array(BOX)

for p in frames:
    img = cv2.imread(p)
    if img is None:
        print(f"[WARN] Could not read {p}, skipping.")
        continue

    predictor.set_image(img[..., ::-1])   # BGR->RGB
    masks, scores, _ = predictor.predict(box=BOX_ARR)
    if masks is None or len(masks) == 0:
        print(f"[WARN] No mask predicted for {os.path.basename(p)}, skipping.")
        continue

    m = masks[int(np.argmax(scores))].astype(np.uint8) * 255
    m = cv2.medianBlur(m, 7)
    m = cv2.dilate(m, np.ones((7,7), np.uint8), 1)

    out_mask = os.path.join(
        MASK_DIR, os.path.splitext(os.path.basename(p))[0] + "_mask.png"
    )
    cv2.imwrite(out_mask, m)

print(f"Done. Wrote masks to: {MASK_DIR}")
