"""
Hand tracking and ASL sign prediction for the SignFlow overlay.

Supports two models running simultaneously:
- SigLIP2 vision model (asl_alphabet_siglip/) — 99.96% accuracy on ASL A-Z static alphabet
- 3D Landmark Transformer (best_model.pth) — 256 ASL motion signs (Hello, Thank you, etc.)

The screen capture feeds frames here. MediaPipe extracts landmarks and hand crops.
Both models run every frame and the highest-confidence prediction is shown.
"""

import collections
import json
import os
import threading
import time
from pathlib import Path

import numpy as np

# IMPORTANT: torch must be imported BEFORE PyQt5 and cv2 on Windows to avoid DLL conflicts
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    torch = None
    nn = None
    F = None

from PyQt5.QtCore import QThread, pyqtSignal

from overlay_constants import SIGN_PREDICTION_MIN_CONFIDENCE

try:
    import cv2
except Exception:
    cv2 = None

# SigLIP2 ASL alphabet model (static A-Z classification)
try:
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    from PIL import Image as PILImage
    _SIGLIP_AVAILABLE = True
except Exception:
    _SIGLIP_AVAILABLE = False

# MediaPipe Tasks API (for both models)
try:
    import mediapipe as mp_lib
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        FaceLandmarker, FaceLandmarkerOptions,
        HandLandmarker, HandLandmarkerOptions,
        PoseLandmarker, PoseLandmarkerOptions,
        RunningMode,
    )
    _MP_TASKS_AVAILABLE = True
except Exception:
    _MP_TASKS_AVAILABLE = False

from overlay_constants import (
    DETECTION_MAX_DIM,
    DETECTION_MIN_DIM,
    ENABLE_DETECTION_RESIZE,
    ENABLE_DETECTION_SQUARE,
)

# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------
# Transformer model constants
MAX_FRAMES = 64
NUM_LANDMARKS = 92
NUM_COORDS = 3
PREDICT_INTERVAL = 3
SMOOTH_ALPHA = 0.6
MIN_BUFFER_FRAMES = 6
LANDMARK_BUFFER_SIZE = 64

LIPS_FACE_IDXS = np.array([
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
], dtype=np.int32)

POSE_UPPER_IDXS = np.array([0, 11, 12, 13, 14, 15, 16, 23, 24, 25], dtype=np.int32)

HAND_CONNS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17),
]


# ===============================================================
# HAND CROP HELPER (for SigLIP2 image-based alphabet model)
# ===============================================================
HAND_CROP_PADDING = 0.45  # 45% padding — hand stays prominent, enough context
SIGLIP_SMOOTH_ALPHA = 0.4  # EMA smoothing for SigLIP predictions (lower = more responsive)
MIN_CROP_PX = 50  # minimum crop size in pixels (skip tiny crops)
SIGLIP_CONF_THRESHOLD = 0.40  # minimum raw confidence to accept SigLIP prediction


def crop_hand_region(image, lm_92, offset, h, w):
    """Crop a square hand region from image using landmarks at lm_92[offset:offset+21].
    Returns PIL Image or None if no hand visible."""
    hand_lm = lm_92[offset:offset + 21]
    valid = np.any(hand_lm != 0, axis=1)
    if not np.any(valid):
        return None
    pts = hand_lm[valid, :2]  # (N, 2) in [0..1]
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    # Add padding
    dx = (x_max - x_min) * HAND_CROP_PADDING
    dy = (y_max - y_min) * HAND_CROP_PADDING
    x_min = max(0.0, x_min - dx)
    y_min = max(0.0, y_min - dy)
    x_max = min(1.0, x_max + dx)
    y_max = min(1.0, y_max + dy)
    # Make square (SigLIP expects square input)
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    side = max(x_max - x_min, y_max - y_min)
    x_min = max(0.0, cx - side / 2)
    y_min = max(0.0, cy - side / 2)
    x_max = min(1.0, cx + side / 2)
    y_max = min(1.0, cy + side / 2)
    # Convert to pixel coords
    px1, py1 = int(x_min * w), int(y_min * h)
    px2, py2 = int(x_max * w), int(y_max * h)
    if px2 - px1 < MIN_CROP_PX or py2 - py1 < MIN_CROP_PX:
        return None
    crop = image[py1:py2, px1:px2]
    if not _SIGLIP_AVAILABLE:
        return None
    return PILImage.fromarray(crop)


# ===============================================================
# TRANSFORMER MODEL DEFINITIONS ("Local Medium" - best_model.pth)
# ===============================================================
if torch is not None and nn is not None:
    class LandmarkEmbedding(nn.Module):
        def __init__(self, in_features, units):
            super().__init__()
            self.empty_embedding = nn.Parameter(torch.zeros(units))
            self.proj = nn.Sequential(
                nn.Linear(in_features, units, bias=False),
                nn.GELU(),
                nn.Linear(units, units, bias=False),
            )

        def forward(self, x):
            out = self.proj(x)
            mask = (x.abs().sum(dim=-1, keepdim=True) == 0)
            return torch.where(mask, self.empty_embedding, out)

    class LandmarkTransformerEmbedding(nn.Module):
        def __init__(self, max_frames, units, lips_units, hands_units, pose_units):
            super().__init__()
            self.positional_embedding = nn.Embedding(max_frames + 1, units)
            nn.init.zeros_(self.positional_embedding.weight)
            self.lips_embedding = LandmarkEmbedding(40 * 3, lips_units)
            self.lh_embedding = LandmarkEmbedding(21 * 3, hands_units)
            self.rh_embedding = LandmarkEmbedding(21 * 3, hands_units)
            self.pose_embedding = LandmarkEmbedding(10 * 3, pose_units)
            self.landmark_weights = nn.Parameter(torch.zeros(4))
            self.fc = nn.Sequential(
                nn.Linear(max(lips_units, hands_units, pose_units), units, bias=False),
                nn.GELU(),
                nn.Linear(units, units, bias=False),
            )
            self.max_frames = max_frames

        def forward(self, frames, non_empty_frame_idxs):
            x = frames
            lips = x[:, :, 0:40, :].reshape(x.shape[0], x.shape[1], 40 * 3)
            lh = x[:, :, 40:61, :].reshape(x.shape[0], x.shape[1], 21 * 3)
            rh = x[:, :, 61:82, :].reshape(x.shape[0], x.shape[1], 21 * 3)
            pose = x[:, :, 82:92, :].reshape(x.shape[0], x.shape[1], 10 * 3)
            lips_emb = self.lips_embedding(lips)
            lh_emb = self.lh_embedding(lh)
            rh_emb = self.rh_embedding(rh)
            pose_emb = self.pose_embedding(pose)
            mu = max(lips_emb.shape[-1], lh_emb.shape[-1],
                     rh_emb.shape[-1], pose_emb.shape[-1])
            if lips_emb.shape[-1] < mu:
                lips_emb = F.pad(lips_emb, (0, mu - lips_emb.shape[-1]))
            if lh_emb.shape[-1] < mu:
                lh_emb = F.pad(lh_emb, (0, mu - lh_emb.shape[-1]))
            if rh_emb.shape[-1] < mu:
                rh_emb = F.pad(rh_emb, (0, mu - rh_emb.shape[-1]))
            if pose_emb.shape[-1] < mu:
                pose_emb = F.pad(pose_emb, (0, mu - pose_emb.shape[-1]))
            stacked = torch.stack([lips_emb, lh_emb, rh_emb, pose_emb], dim=-1)
            weights = torch.softmax(self.landmark_weights, dim=0)
            fused = (stacked * weights).sum(dim=-1)
            fused = self.fc(fused)
            max_idx = non_empty_frame_idxs.max(dim=1, keepdim=True).values.clamp(min=1)
            pos_indices = torch.where(
                non_empty_frame_idxs == -1.0,
                torch.tensor(self.max_frames, device=frames.device, dtype=torch.long),
                (non_empty_frame_idxs / max_idx * self.max_frames).long().clamp(0, self.max_frames - 1),
            )
            return fused + self.positional_embedding(pos_indices)

    class TransformerBlock(nn.Module):
        def __init__(self, units, num_heads, mlp_ratio=4, dropout=0.1):
            super().__init__()
            self.norm1 = nn.LayerNorm(units)
            self.attn = nn.MultiheadAttention(units, num_heads, dropout=dropout, batch_first=True)
            self.norm2 = nn.LayerNorm(units)
            self.mlp = nn.Sequential(
                nn.Linear(units, units * mlp_ratio), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(units * mlp_ratio, units), nn.Dropout(dropout),
            )

        def forward(self, x, key_padding_mask=None):
            normed = self.norm1(x)
            attn_out, _ = self.attn(normed, normed, normed, key_padding_mask=key_padding_mask)
            x = x + attn_out
            return x + self.mlp(self.norm2(x))

    class LandmarkTransformer(nn.Module):
        def __init__(self, num_classes, max_frames=64, units=512, num_blocks=4,
                     num_heads=8, mlp_ratio=4, dropout=0.2, emb_dropout=0.1,
                     lips_units=384, hands_units=384, pose_units=256):
            super().__init__()
            self.embedding = LandmarkTransformerEmbedding(
                max_frames, units, lips_units, hands_units, pose_units)
            self.emb_dropout = nn.Dropout(emb_dropout)
            self.blocks = nn.ModuleList([
                TransformerBlock(units, num_heads, mlp_ratio, dropout)
                for _ in range(num_blocks)])
            self.norm = nn.LayerNorm(units)
            self.head_dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(units, num_classes)
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

        def forward(self, frames, non_empty_frame_idxs):
            x = self.embedding(frames, non_empty_frame_idxs)
            x = self.emb_dropout(x)
            kpm = (non_empty_frame_idxs == -1.0)
            for block in self.blocks:
                x = block(x, key_padding_mask=kpm)
            x = self.norm(x)
            mask = (~kpm).unsqueeze(-1).float()
            denom = mask.sum(dim=1).clamp(min=1e-6)
            x = (x * mask).sum(dim=1) / denom
            x = self.head_dropout(x)
            return self.classifier(x)
else:
    LandmarkTransformer = None


# ===============================================================
# MediaPipe Tasks Landmark Extractor (for transformer model)
# ===============================================================
class TasksLandmarkExtractor:
    """Extracts 92 landmarks using MediaPipe Tasks API."""

    def __init__(self, model_dir):
        self.face_available = True
        try:
            self.face_lm = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=os.path.join(model_dir, "face_landmarker.task")),
                running_mode=RunningMode.IMAGE, num_faces=1,
                min_face_detection_confidence=0.1, min_face_presence_confidence=0.1,
            ))
        except Exception:
            self.face_available = False

        self.hand_lm = HandLandmarker.create_from_options(HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=os.path.join(model_dir, "hand_landmarker.task")),
            running_mode=RunningMode.IMAGE, num_hands=2,
            min_hand_detection_confidence=0.2, min_hand_presence_confidence=0.2,
        ))
        self.pose_lm = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=os.path.join(model_dir, "pose_landmarker_heavy.task")),
            running_mode=RunningMode.IMAGE, num_poses=1,
            min_pose_detection_confidence=0.2, min_pose_presence_confidence=0.2,
        ))

    def extract(self, rgb_array):
        """Extract 92 landmarks. Returns (landmarks_92x3, face_ok, hands_ok, pose_ok)."""
        mp_img = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB,
                              data=np.ascontiguousarray(rgb_array))
        lm = np.zeros((NUM_LANDMARKS, NUM_COORDS), dtype=np.float32)

        face_ok = False
        if self.face_available:
            fr = self.face_lm.detect(mp_img)
            if fr.face_landmarks and len(fr.face_landmarks) > 0:
                face = fr.face_landmarks[0]
                for i, fi in enumerate(LIPS_FACE_IDXS):
                    if fi < len(face):
                        lm[i] = [face[fi].x, face[fi].y, face[fi].z]
                face_ok = True

        hands_ok = False
        hr = self.hand_lm.detect(mp_img)
        if hr.hand_landmarks:
            for hi, (hm, hn) in enumerate(zip(hr.hand_landmarks, hr.handedness)):
                if hi >= 2:
                    break
                label = hn[0].category_name.lower() if hn else "left"
                offset = 40 if label == "left" else 61
                for j in range(min(21, len(hm))):
                    lm[offset + j] = [hm[j].x, hm[j].y, hm[j].z]
                hands_ok = True

        pose_ok = False
        pr = self.pose_lm.detect(mp_img)
        if pr.pose_landmarks and len(pr.pose_landmarks) > 0:
            pose = pr.pose_landmarks[0]
            for k, pidx in enumerate(POSE_UPPER_IDXS):
                if pidx < len(pose):
                    lm[82 + k] = [pose[pidx].x, pose[pidx].y, pose[pidx].z]
            pose_ok = True

        return lm, face_ok, hands_ok, pose_ok

    def close(self):
        if self.face_available:
            self.face_lm.close()
        self.hand_lm.close()
        self.pose_lm.close()


# ---------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------
def _draw_hand_lm(image, lm, offset, color, w, h):
    for i in range(21):
        x, y = lm[offset + i, 0], lm[offset + i, 1]
        if x > 0 or y > 0:
            cv2.circle(image, (int(x * w), int(y * h)), 3, color, -1)
    for a, b in HAND_CONNS:
        x1, y1 = lm[offset + a, 0], lm[offset + a, 1]
        x2, y2 = lm[offset + b, 0], lm[offset + b, 1]
        if (x1 > 0 or y1 > 0) and (x2 > 0 or y2 > 0):
            cv2.line(image, (int(x1 * w), int(y1 * h)),
                     (int(x2 * w), int(y2 * h)), color, 2)


def draw_92_landmarks(image, lm):
    """Draw all 92 landmarks on a BGR image."""
    if cv2 is None:
        return
    h, w = image.shape[:2]
    for i in range(40):
        x, y = lm[i, 0], lm[i, 1]
        if x > 0 or y > 0:
            cv2.circle(image, (int(x * w), int(y * h)), 2, (0, 255, 0), -1)
    _draw_hand_lm(image, lm, 40, (255, 150, 0), w, h)
    _draw_hand_lm(image, lm, 61, (0, 130, 255), w, h)
    for i in range(82, 92):
        x, y = lm[i, 0], lm[i, 1]
        if x > 0 or y > 0:
            cv2.circle(image, (int(x * w), int(y * h)), 4, (0, 255, 255), -1)


# ---------------------------------------------------------------
# Transformer inference
# ---------------------------------------------------------------
def run_transformer_inference(model, frames_list, device):
    n = len(frames_list)
    if n < MIN_BUFFER_FRAMES:
        return None
    if n > MAX_FRAMES:
        indices = np.linspace(0, n - 1, MAX_FRAMES, dtype=int)
        frames_list = [frames_list[i] for i in indices]
        n = MAX_FRAMES
    arr = np.stack(frames_list, axis=0).astype(np.float32)
    if n < MAX_FRAMES:
        pad = np.zeros((MAX_FRAMES - n, NUM_LANDMARKS, NUM_COORDS), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
    non_empty = np.any(arr != 0, axis=(1, 2))
    ne_idxs = np.where(non_empty, np.arange(MAX_FRAMES, dtype=np.float32), -1.0)
    frames_t = torch.from_numpy(arr).unsqueeze(0).to(device)
    idxs_t = torch.from_numpy(ne_idxs).unsqueeze(0).to(device)
    with torch.no_grad():
        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(frames_t, idxs_t)
        else:
            logits = model(frames_t, idxs_t)
        return torch.softmax(logits, dim=1)[0].cpu().numpy()


# ===============================================================
# HandTracker - supports BOTH models
# ===============================================================
class HandTracker:
    def __init__(self):
        self.available = _MP_TASKS_AVAILABLE

        # --- SigLIP2 alphabet model (static A-Z) ---
        self._siglip_model = None
        self._siglip_processor = None
        self._siglip_device = None
        self._siglip_avg_label = None  # EMA-smoothed label tracking
        self._siglip_avg_conf = 0.0    # EMA-smoothed confidence

        # --- transformer model (Local Medium) ---
        self._transformer_model = None
        self._transformer_device = None
        self._transformer_class_names = {}
        self._tasks_extractor = None
        self._landmark_buffer = collections.deque(maxlen=LANDMARK_BUFFER_SIZE)
        self._avg_probs = None
        self._frame_count = 0

        self.last_status = {
            "hands_detected": 0,
            "left_conf": 0.0,
            "right_conf": 0.0,
            "prediction": "No Hand",
            "prediction_conf": 0.0,
        }

        if not self.available:
            return

        mp_model_dir = Path(__file__).resolve().parent / "mediapipe_models"

        # --- Load SigLIP2 ASL alphabet model ---
        if _SIGLIP_AVAILABLE and torch is not None:
            siglip_dir = Path(__file__).resolve().parent / "models" / "asl_alphabet_siglip"
            if siglip_dir.exists():
                try:
                    self._siglip_device = torch.device(
                        "cuda" if torch.cuda.is_available() else "cpu")
                    self._siglip_processor = AutoImageProcessor.from_pretrained(
                        str(siglip_dir))
                    self._siglip_model = AutoModelForImageClassification.from_pretrained(
                        str(siglip_dir)).to(self._siglip_device).eval()
                    # Use half-precision on CUDA for faster inference
                    if self._siglip_device.type == "cuda":
                        self._siglip_model = self._siglip_model.half()
                    print(f"[SignFlow] SigLIP2 alphabet model loaded ({self._siglip_model.config.num_labels} classes)")
                except Exception as e:
                    print(f"[SignFlow] SigLIP2 model load error: {e}")
                    self._siglip_model = None

        # --- Load transformer model ---
        if torch is not None and LandmarkTransformer is not None:
            model_dir = Path(__file__).resolve().parent / "models"
            model_path = model_dir / "best_model.pth"
            class_map_path = model_dir / "class_map.json"

            if model_path.exists():
                try:
                    self._transformer_device = torch.device(
                        "cuda" if torch.cuda.is_available() else "cpu")
                    ckpt = torch.load(str(model_path),
                                      map_location=self._transformer_device,
                                      weights_only=False)
                    config = ckpt.get("config", {})
                    num_classes = ckpt.get("num_classes", 256)

                    if class_map_path.exists():
                        with open(class_map_path) as f:
                            self._transformer_class_names = {
                                int(k): v for k, v in json.load(f).items()}
                    else:
                        self._transformer_class_names = {
                            i: n for i, n in enumerate(ckpt.get("class_names", []))}

                    self._transformer_model = LandmarkTransformer(
                        num_classes=num_classes,
                        max_frames=config.get("max_frames", MAX_FRAMES),
                        units=config.get("units", 512),
                        num_blocks=config.get("num_blocks", 4),
                        num_heads=config.get("num_heads", 8),
                        dropout=config.get("dropout", 0.15),
                    ).to(self._transformer_device)
                    self._transformer_model.load_state_dict(ckpt["model_state_dict"])
                    self._transformer_model.eval()

                    # GPU warmup
                    if self._transformer_device.type == "cuda":
                        d_f = torch.zeros(1, MAX_FRAMES, NUM_LANDMARKS, NUM_COORDS,
                                          device=self._transformer_device)
                        d_i = torch.full((1, MAX_FRAMES), -1.0,
                                         device=self._transformer_device)
                        d_i[0, 0] = 0.0
                        with torch.no_grad(), torch.amp.autocast("cuda"):
                            self._transformer_model(d_f, d_i)
                        del d_f, d_i
                except Exception as e:
                    print(f"[SignFlow] Transformer model load error: {e}")
                    self._transformer_model = None

        # --- Init MediaPipe Tasks (for transformer model) ---
        if _MP_TASKS_AVAILABLE:
            mp_model_dir = Path(__file__).resolve().parent / "mediapipe_models"
            if mp_model_dir.exists():
                try:
                    self._tasks_extractor = TasksLandmarkExtractor(str(mp_model_dir))
                except Exception as e:
                    print(f"[SignFlow] MediaPipe Tasks init error: {e}")

    def process(self, frame: dict, flip_horizontal: bool = False):
        """Process a frame with BOTH models, pick the best prediction.

        1. Extract 92 landmarks (face+hands+pose) via Tasks API — one pass
        2. Crop hand region and run SigLIP2 (static signs A-Z, 99.96% accuracy)
        3. Buffer landmarks, run transformer every N frames (motion signs like Hello)
        4. Show whichever prediction has higher confidence
        """
        if not self.available or frame is None or self._tasks_extractor is None:
            return frame, self.last_status

        rgb = frame.get("rgb")
        width = int(frame.get("width", 0) or 0)
        height = int(frame.get("height", 0) or 0)
        if rgb is None or width <= 0 or height <= 0:
            return frame, self.last_status

        start_time = time.perf_counter()
        image = np.frombuffer(rgb, dtype=np.uint8).reshape(height, width, 3).copy()

        # Extract landmarks from the original image orientation
        if flip_horizontal:
            extract_image = image[:, ::-1, :].copy()
        else:
            extract_image = image

        # --- Step 1: Extract all 92 landmarks (single MediaPipe pass) ---
        lm, face_ok, hands_ok, pose_ok = self._tasks_extractor.extract(extract_image)

        # Draw landmarks on display image
        if cv2 is not None:
            if flip_horizontal:
                display_lm = lm.copy()
                mask = np.any(lm != 0, axis=1)
                display_lm[mask, 0] = 1.0 - display_lm[mask, 0]
                draw_92_landmarks(image, display_lm)
            else:
                draw_92_landmarks(image, lm)

        hands_visible = np.any(lm[40:82] != 0)
        lh_visible = np.any(lm[40:61] != 0)
        rh_visible = np.any(lm[61:82] != 0)

        # --- Step 2: SigLIP2 alphabet prediction (static signs A-Z) ---
        siglip_text = "No Hand"
        siglip_conf = 0.0

        if hands_visible and self._siglip_model is not None and self._siglip_processor is not None:
            # Collect all crop images for batch inference (original + flipped)
            crop_images = []
            for offset in [61, 40]:  # right hand first, then left
                crop = crop_hand_region(extract_image, lm, offset, height, width)
                if crop is not None:
                    crop_images.append(crop)
                    crop_images.append(crop.transpose(PILImage.FLIP_LEFT_RIGHT))

            best_raw_conf = 0.0
            best_raw_label = "No Hand"

            if crop_images:
                try:
                    # Batch all crops into single forward pass for speed
                    inputs = self._siglip_processor(images=crop_images, return_tensors="pt")
                    inputs = {k: v.to(self._siglip_device) for k, v in inputs.items()}
                    with torch.no_grad():
                        if self._siglip_device.type == "cuda":
                            with torch.amp.autocast("cuda"):
                                logits = self._siglip_model(**inputs).logits
                        else:
                            logits = self._siglip_model(**inputs).logits
                        all_probs = torch.softmax(logits.float(), dim=1)  # [N, 26]

                    # Pick the crop with highest confidence
                    for i in range(all_probs.shape[0]):
                        probs = all_probs[i]
                        top_conf = float(probs.max())
                        top_idx = int(probs.argmax())
                        if top_conf > best_raw_conf:
                            best_raw_conf = top_conf
                            best_raw_label = self._siglip_model.config.id2label.get(
                                top_idx, str(top_idx))
                except Exception:
                    pass

            # EMA smoothing for stable predictions
            if best_raw_conf >= SIGLIP_CONF_THRESHOLD:
                if self._siglip_avg_label == best_raw_label:
                    # Same label — smooth confidence upward
                    self._siglip_avg_conf = (
                        SIGLIP_SMOOTH_ALPHA * best_raw_conf
                        + (1 - SIGLIP_SMOOTH_ALPHA) * self._siglip_avg_conf)
                else:
                    # New label — switch quickly if confident
                    if best_raw_conf > 0.4 or best_raw_conf > self._siglip_avg_conf:
                        self._siglip_avg_label = best_raw_label
                        self._siglip_avg_conf = best_raw_conf * 0.85
                siglip_text = self._siglip_avg_label
                siglip_conf = self._siglip_avg_conf
            else:
                # Decay confidence when no good detection
                self._siglip_avg_conf *= 0.85
                if self._siglip_avg_conf > 0.15 and self._siglip_avg_label is not None:
                    siglip_text = self._siglip_avg_label
                    siglip_conf = self._siglip_avg_conf
                else:
                    self._siglip_avg_label = None
                    self._siglip_avg_conf = 0.0

        # --- Step 3: Transformer sequence prediction (motion signs) ---
        transformer_text = "No Hand"
        transformer_conf = 0.0

        if hands_visible:
            self._landmark_buffer.append(lm.copy())

        self._frame_count += 1

        if (self._frame_count % PREDICT_INTERVAL == 0
                and len(self._landmark_buffer) >= MIN_BUFFER_FRAMES
                and self._transformer_model is not None):
            probs = run_transformer_inference(
                self._transformer_model, list(self._landmark_buffer),
                self._transformer_device)
            if probs is not None:
                if self._avg_probs is None:
                    self._avg_probs = probs
                else:
                    self._avg_probs = SMOOTH_ALPHA * probs + (1 - SMOOTH_ALPHA) * self._avg_probs

        if self._avg_probs is not None:
            top_idx = int(np.argmax(self._avg_probs))
            transformer_conf = float(self._avg_probs[top_idx])
            if transformer_conf >= SIGN_PREDICTION_MIN_CONFIDENCE:
                transformer_text = self._transformer_class_names.get(top_idx, str(top_idx))
            else:
                transformer_text = "Uncertain"

        # --- Step 4: Pick the best prediction ---
        # Strategy: SigLIP is better for static signs (A-Z), transformer for motion.
        # If SigLIP is highly confident (>0.6), it's likely a real static sign — prefer it.
        # Otherwise fall back to transformer which tracks motion patterns.
        prediction_text = "No Hand"
        prediction_conf = 0.0

        if not hands_visible:
            prediction_text = "No Hand"
            prediction_conf = 0.0
        elif siglip_conf >= 0.6:
            # SigLIP very confident — this is a clear static alphabet sign
            prediction_text = siglip_text
            prediction_conf = siglip_conf
        elif transformer_conf >= siglip_conf and transformer_text not in ("No Hand", "Uncertain"):
            prediction_text = transformer_text
            prediction_conf = transformer_conf
        elif siglip_text not in ("No Hand", "Uncertain"):
            prediction_text = siglip_text
            prediction_conf = siglip_conf
        elif transformer_text not in ("No Hand", "Uncertain"):
            prediction_text = transformer_text
            prediction_conf = transformer_conf

        # If still uncertain, show whatever has any confidence
        if prediction_text in ("No Hand", "Uncertain") and hands_visible:
            if transformer_conf > siglip_conf and transformer_text not in ("No Hand", "Uncertain"):
                prediction_text = transformer_text
                prediction_conf = transformer_conf
            elif siglip_text not in ("No Hand", "Uncertain"):
                prediction_text = siglip_text
                prediction_conf = siglip_conf

        lh_conf = 1.0 if lh_visible else 0.0
        rh_conf = 1.0 if rh_visible else 0.0
        hands_detected = int(lh_conf > 0) + int(rh_conf > 0)
        processing_ms = (time.perf_counter() - start_time) * 1000.0

        self.last_status = {
            "hands_detected": hands_detected,
            "left_conf": lh_conf,
            "right_conf": rh_conf,
            "prediction": prediction_text,
            "prediction_conf": prediction_conf,
            "input_w": width,
            "input_h": height,
            "det_w": width,
            "det_h": height,
            "det_scale": 1.0,
            "pad_x": 0,
            "pad_y": 0,
            "flip": bool(flip_horizontal),
            "model_loaded": (self._siglip_model is not None or
                             self._transformer_model is not None),
            "hand_label": ("Both" if (lh_conf > 0 and rh_conf > 0)
                           else "Left" if lh_conf > 0
                           else "Right" if rh_conf > 0
                           else "None"),
            "processing_ms": processing_ms,
            "face_ok": face_ok,
            "pose_ok": pose_ok,
            "buffer_size": len(self._landmark_buffer),
            "siglip_pred": siglip_text,
            "siglip_conf": siglip_conf,
            "transformer_pred": transformer_text,
            "transformer_conf": transformer_conf,
        }

        out_frame = dict(frame)
        out_frame["rgb"] = image.tobytes()
        return out_frame, self.last_status

    def close(self):
        if self._tasks_extractor is not None:
            self._tasks_extractor.close()


# ===============================================================
# Worker thread
# ===============================================================
class HandTrackingWorker(QThread):
    status_updated = pyqtSignal(dict)
    frame_processed = pyqtSignal(object)
    fps_updated = pyqtSignal(float)
    prediction_updated = pyqtSignal(str)

    def __init__(self, flip_horizontal: bool = False):
        super().__init__()
        self._tracker = HandTracker()
        self._flip_horizontal = bool(flip_horizontal)
        self._queue = collections.deque(maxlen=1)
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._running = True
        self._fps = 0.0
        self._last_time = None

    @property
    def available(self):
        return self._tracker.available

    def submit(self, frame: dict):
        if not self.available or frame is None:
            return
        with self._lock:
            self._queue.clear()
            self._queue.append(frame)
        self._event.set()

    def run(self):
        while self._running:
            if not self._event.wait(0.5):
                continue
            self._event.clear()
            with self._lock:
                frame = self._queue.pop() if self._queue else None
            if frame is None:
                continue
            processed, status = self._tracker.process(frame, flip_horizontal=self._flip_horizontal)

            now = time.perf_counter()
            if self._last_time is not None:
                delta = now - self._last_time
                if delta > 1e-6:
                    instant = 1.0 / delta
                    self._fps = (self._fps * 0.85) + (instant * 0.15)
                    self.fps_updated.emit(self._fps)
            self._last_time = now

            if status is not None:
                self.status_updated.emit(status)
                prediction = status.get("prediction") if isinstance(status, dict) else None
                if prediction is not None:
                    self.prediction_updated.emit(str(prediction))
            if processed is not None:
                self.frame_processed.emit(processed)

    def stop(self):
        self._running = False
        self._event.set()
        self.wait(500)
        if self._tracker is not None:
            self._tracker.close()
