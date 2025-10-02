import os, cv2, numpy as np, torch, sam2, urllib.request
from glob import glob
from pathlib import Path
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2

# ====== SETTINGS ======
IM_DIR    = "lantern_ds/frames"
MASK_DIR  = "lantern_ds/masks"
CKPT_FILE = "sam2_hiera_base_plus.pt"
WANTED    = "sam2_hiera_base_plus.yaml"
BOX       = (520, 220, 1120, 980)  # (x0,y0,x1,y1)
# ======================

os.makedirs(MASK_DIR, exist_ok=True)

if not os.path.isfile(CKPT_FILE):
    raise FileNotFoundError(f"Checkpoint not found: {CKPT_FILE}")

def find_or_fetch_yaml() -> Path:
    local_yaml = Path(WANTED)
    
    # 1) Check if already downloaded locally
    if local_yaml.exists():
        print(f"[INFO] Found existing YAML: {local_yaml}")
        return local_yaml
    
    # 2) Search inside installed package
    pkg_dir = Path(sam2.__file__).parent
    for y in pkg_dir.rglob("*.yaml"):
        if y.name.lower() == WANTED.lower():
            print(f"[INFO] Found YAML in package: {y}")
            return y
    
    # 3) Try updated GitHub URLs
    gh_candidates = [
        "https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/sam2_configs/sam2_hiera_b%2B.yaml",
        "https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/sam2/configs/sam2.1/sam2.1_hiera_b%2B.yaml",
        "https://raw.githubusercontent.com/facebookresearch/segment-anything-2/refs/heads/main/sam2_configs/sam2_hiera_b%2B.yaml",
    ]
    
    for url in gh_candidates:
        try:
            print(f"[INFO] Trying GitHub: {url}")
            urllib.request.urlretrieve(url, local_yaml)
            print(f"[INFO] Downloaded YAML to: {local_yaml}")
            return local_yaml
        except Exception as e:
            print(f"[WARN] Failed: {e}")
    
    # 4) Try Hugging Face with correct repo ID
    try:
        from huggingface_hub import hf_hub_download
        print("[INFO] Trying Hugging Face...")
        
        # Correct repo ID for the model checkpoint
        try:
            cfg_in_hf = hf_hub_download(
                repo_id="facebook/sam2-hiera-base-plus",
                filename="sam2_hiera_base_plus.yaml"
            )
            local_yaml.write_bytes(Path(cfg_in_hf).read_bytes())
            print(f"[INFO] Got YAML from Hugging Face -> {local_yaml}")
            return local_yaml
        except Exception as e:
            print(f"[WARN] HF fetch failed: {e}")
    except ImportError:
        print("[WARN] huggingface_hub not available")
    
    # 5) Create YAML from template as last resort
    print("[INFO] Creating YAML from template...")
    yaml_content = """# SAM 2 Hiera Base Plus Configuration
model:
  _target_: sam2.modeling.sam2_base.SAM2Base
  image_encoder:
    _target_: sam2.modeling.backbones.image_encoder.ImageEncoder
    scalp: 1
    trunk:
      _target_: sam2.modeling.backbones.hieradet.Hiera
      embed_dim: 112
      num_heads: 2
      stages: [1, 2, 11, 2]
      global_att_blocks: [7, 10, 13]
      window_pos_embed_bkg_spatial_size: [7, 7]
    neck:
      _target_: sam2.modeling.backbones.image_encoder.FpnNeck
      position_encoding:
        _target_: sam2.modeling.position_encoding.PositionEmbeddingSine
        num_pos_feats: 256
        normalize: true
        scale: null
        temperature: 10000
      d_model: 256
      backbone_channel_list: [896, 448, 224, 112]
      fpn_top_down_levels: [2, 3]
      fpn_interp_model: nearest

  memory_attention:
    _target_: sam2.modeling.memory_attention.MemoryAttention
    d_model: 256
    pos_enc_at_input: true
    layer:
      _target_: sam2.modeling.memory_attention.MemoryAttentionLayer
      activation: relu
      dim_feedforward: 2048
      dropout: 0.1
      pos_enc_at_attn: false
      self_attention:
        _target_: sam2.modeling.sam.transformer.RoPEAttention
        rope_theta: 10000.0
        feat_sizes: [32, 32]
        embedding_dim: 256
        num_heads: 1
        downsample_rate: 1
        dropout: 0.1
      d_model: 256
      pos_enc_at_cross_attn_keys: true
      pos_enc_at_cross_attn_queries: false
      cross_attention:
        _target_: sam2.modeling.sam.transformer.RoPEAttention
        rope_theta: 10000.0
        feat_sizes: [32, 32]
        rope_k_repeat: True
        embedding_dim: 256
        num_heads: 1
        downsample_rate: 1
        dropout: 0.1
        kv_in_dim: 64
    num_layers: 4

  memory_encoder:
    _target_: sam2.modeling.memory_encoder.MemoryEncoder
    out_dim: 64
    position_encoding:
      _target_: sam2.modeling.position_encoding.PositionEmbeddingSine
      num_pos_feats: 64
      normalize: true
      scale: null
      temperature: 10000
    mask_downsampler:
      _target_: sam2.modeling.memory_encoder.MaskDownSampler
      kernel_size: 3
      stride: 2
      padding: 1
    fuser:
      _target_: sam2.modeling.memory_encoder.Fuser
      layer:
        _target_: sam2.modeling.memory_encoder.CXBlock
        dim: 256
        kernel_size: 7
        padding: 3
        layer_scale_init_value: 1e-6
        use_dwconv: True
      num_layers: 2

  num_maskmem: 7
  image_size: 1024
  backbone_stride: 16
  sigmoid_scale_for_mem_enc: 20.0
  sigmoid_bias_for_mem_enc: -10.0
  use_mask_input_as_output_without_sam: true
  directly_add_no_mem_embed: true
  use_high_res_features_in_sam: true
  multimask_output_in_sam: true
  iou_prediction_use_sigmoid: True
  use_obj_ptrs_in_encoder: true
  add_tpos_enc_to_obj_ptrs: false
  only_obj_ptrs_in_the_past_for_eval: true
  pred_obj_scores: true
  pred_obj_scores_mlp: true
  fixed_no_obj_ptr: true
  soft_no_obj_ptr: false
  use_mlp_for_obj_ptr_proj: true
  no_obj_embed_spatial: false

  sam_mask_decoder_extra_args:
    dynamic_multimask_via_stability: true
    dynamic_multimask_stability_delta: 0.05
    dynamic_multimask_stability_thresh: 0.98

  compile_image_encoder: false
"""
    
    local_yaml.write_text(yaml_content)
    print(f"[INFO] Created template YAML: {local_yaml}")
    return local_yaml

# Get config path
cfg_path = find_or_fetch_yaml()
print(f"[INFO] Using SAM2 config: {cfg_path}")

# Build model
print("Building SAM2 model…")
sam = build_sam2(
    config_file=str(cfg_path),
    ckpt_path=CKPT_FILE,
    device="cuda" if torch.cuda.is_available() else "cpu"
)
predictor = SAM2ImagePredictor(sam)
print("SAM2 ready.")

# Process frames
frames = sorted([*glob(f"{IM_DIR}/*.jpg"), *glob(f"{IM_DIR}/*.png")])
if not frames:
    raise RuntimeError(f"No images found in {IM_DIR}. Did dataset_prep.py run?")

BOX_ARR = np.array(BOX)

for p in frames:
    img = cv2.imread(p)
    if img is None:
        print(f"[WARN] Could not read {p}, skipping.")
        continue

    predictor.set_image(img[..., ::-1])  # BGR->RGB
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