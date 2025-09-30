import os, cv2, numpy as np
from glob import glob
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

IM_DIR = "lantern_ds/frames"
MASK_DIR = "lantern_ds/masks"
os.makedirs(MASK_DIR, exist_ok=True)

sam = build_sam2("sam2_hiera_base_plus.pt")
pred = SAM2ImagePredictor(sam)

# TODO: adjust after opening one image: (x0, y0, x1, y1)
BOX = (520, 220, 1120, 980)

imgs = sorted(glob(f"{IM_DIR}/*.jpg"))
for p in imgs:
    img = cv2.imread(p)
    pred.set_image(img[..., ::-1])  # BGR->RGB
    masks, scores, _ = pred.predict(box=np.array(BOX))
    m = masks[int(np.argmax(scores))].astype(np.uint8) * 255
    # keep grass and smooth edges
    m = cv2.medianBlur(m, 7)
    m = cv2.dilate(m, np.ones((7,7),np.uint8), 1)
    out_mask = os.path.join(MASK_DIR, os.path.basename(p).replace(".jpg","_mask.png"))
    cv2.imwrite(out_mask, m)

print("Masks written to", MASK_DIR)