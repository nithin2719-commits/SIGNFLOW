"""
Real-time ASL Sign Language Recognition - Live Continuous Prediction.

Shows hands -> instant prediction. No buttons, no recording, no waiting.
Predicts continuously every ~0.3 seconds on the last ~2 seconds of frames.

Handles missing face/lip landmarks gracefully (common on webcams).

Controls:  Q = Quit  |  C = Clear

Usage:
  python sign_inference.py
  python sign_inference.py --camera 1
"""

import argparse
import collections
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import mediapipe as mp_lib
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker, FaceLandmarkerOptions,
    HandLandmarker, HandLandmarkerOptions,
    PoseLandmarker, PoseLandmarkerOptions,
    RunningMode,
)

# ---------------------------------------------------------------
# Constants (must match training)
# ---------------------------------------------------------------
MAX_FRAMES = 64
NUM_LANDMARKS = 92
NUM_COORDS = 3

LIPS_FACE_IDXS = np.array([
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
], dtype=np.int32)

POSE_UPPER_IDXS = np.array([0, 11, 12, 13, 14, 15, 16, 23, 24, 25], dtype=np.int32)

# MediaPipe pose: face-related landmarks (nose, eyes, ears, mouth corners)
POSE_FACE_IDXS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

HAND_CONNS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
]


# ---------------------------------------------------------------
# Model (identical to training script)
# ---------------------------------------------------------------
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


# ---------------------------------------------------------------
# Landmark extraction - ALL landmarks EVERY frame
# ---------------------------------------------------------------
class LandmarkExtractor:
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
        self.face_detected_ever = False

    def extract(self, rgb):
        """Extract ALL 92 landmarks from an RGB frame."""
        mp_img = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB,
                              data=np.ascontiguousarray(rgb))
        lm = np.zeros((NUM_LANDMARKS, NUM_COORDS), dtype=np.float32)

        # --- Face / Lips (40 landmarks) ---
        face_ok = False
        if self.face_available:
            fr = self.face_lm.detect(mp_img)
            if fr.face_landmarks and len(fr.face_landmarks) > 0:
                face = fr.face_landmarks[0]
                for i, fi in enumerate(LIPS_FACE_IDXS):
                    if fi < len(face):
                        lm[i] = [face[fi].x, face[fi].y, face[fi].z]
                face_ok = True
                self.face_detected_ever = True

        # --- Hands (21 + 21 landmarks) ---
        hr = self.hand_lm.detect(mp_img)
        if hr.hand_landmarks:
            for hi, (hm, hn) in enumerate(zip(hr.hand_landmarks, hr.handedness)):
                if hi >= 2:
                    break
                label = hn[0].category_name.lower() if hn else "left"
                offset = 40 if label == "left" else 61
                for j in range(min(21, len(hm))):
                    lm[offset + j] = [hm[j].x, hm[j].y, hm[j].z]

        # --- Pose (10 landmarks) ---
        pr = self.pose_lm.detect(mp_img)
        if pr.pose_landmarks and len(pr.pose_landmarks) > 0:
            pose = pr.pose_landmarks[0]
            for k, pidx in enumerate(POSE_UPPER_IDXS):
                if pidx < len(pose):
                    lm[82 + k] = [pose[pidx].x, pose[pidx].y, pose[pidx].z]

            # If face landmarker failed, use pose face landmarks as approximate lips
            # This gives the model SOMETHING in the lip slots instead of all zeros
            if not face_ok and len(pose) > 10:
                nose = pose[0]
                mouth_l = pose[9] if 9 < len(pose) else nose
                mouth_r = pose[10] if 10 < len(pose) else nose
                # Fill lip slots with approximate positions from pose
                # Use nose, mouth corners, and interpolate to create 40 approximate points
                # This won't be perfect but gives the model face-region signal
                lip_center_x = (mouth_l.x + mouth_r.x) / 2
                lip_center_y = (mouth_l.y + mouth_r.y) / 2
                lip_center_z = (mouth_l.z + mouth_r.z) / 2
                lip_width = abs(mouth_r.x - mouth_l.x) * 0.5
                lip_height = lip_width * 0.4

                for i in range(40):
                    # Distribute points in an ellipse around lip center
                    angle = 2 * np.pi * i / 40
                    if i < 20:
                        # Outer lip
                        rx, ry = lip_width, lip_height
                    else:
                        # Inner lip (smaller)
                        rx, ry = lip_width * 0.6, lip_height * 0.5
                    lm[i, 0] = lip_center_x + rx * np.cos(angle)
                    lm[i, 1] = lip_center_y + ry * np.sin(angle)
                    lm[i, 2] = lip_center_z

        return lm, face_ok

    def close(self):
        if self.face_available:
            self.face_lm.close()
        self.hand_lm.close()
        self.pose_lm.close()


# ---------------------------------------------------------------
# Inference
# ---------------------------------------------------------------
def run_inference(model, frames_list, device):
    n = len(frames_list)
    if n < 5:
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


# ---------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------
def draw_hand(frame, lm, offset, color, w, h):
    for i in range(21):
        x, y = lm[offset + i, 0], lm[offset + i, 1]
        if x > 0 or y > 0:
            cv2.circle(frame, (int(x * w), int(y * h)), 4, color, -1)
    for a, b in HAND_CONNS:
        x1, y1 = lm[offset + a, 0], lm[offset + a, 1]
        x2, y2 = lm[offset + b, 0], lm[offset + b, 1]
        if (x1 > 0 or y1 > 0) and (x2 > 0 or y2 > 0):
            cv2.line(frame, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), color, 2)


def draw_landmarks(frame, lm):
    h, w = frame.shape[:2]
    # Lips - green
    for i in range(40):
        x, y = lm[i, 0], lm[i, 1]
        if x > 0 or y > 0:
            cv2.circle(frame, (int(x * w), int(y * h)), 2, (0, 255, 0), -1)
    # Hands
    draw_hand(frame, lm, 40, (255, 150, 0), w, h)
    draw_hand(frame, lm, 61, (0, 130, 255), w, h)
    # Pose
    for i in range(82, 92):
        x, y = lm[i, 0], lm[i, 1]
        if x > 0 or y > 0:
            cv2.circle(frame, (int(x * w), int(y * h)), 5, (0, 255, 255), -1)
    pose_conns = [(1,2), (1,3), (2,4), (3,5), (4,6), (1,7), (2,8)]
    for a, b in pose_conns:
        x1, y1 = lm[82+a, 0], lm[82+a, 1]
        x2, y2 = lm[82+b, 0], lm[82+b, 1]
        if (x1 > 0 or y1 > 0) and (x2 > 0 or y2 > 0):
            cv2.line(frame, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), (0, 255, 255), 2)


def flip_lm_for_display(lm):
    f = lm.copy()
    mask = np.any(lm != 0, axis=1)
    f[mask, 0] = 1.0 - f[mask, 0]
    return f


def draw_ui(frame, sign, conf, top5, face_ok, hands_ok, pose_ok, buf_sz, fps):
    h, w = frame.shape[:2]

    # Top bar
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 55), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, "SIGNFLOW - ASL Recognition", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # Status dots
    x = 10
    for label, ok, c in [("LIPS", face_ok, (0,255,0)),
                          ("HANDS", hands_ok, (255,150,0)),
                          ("POSE", pose_ok, (0,255,255))]:
        dc = c if ok else (50, 50, 50)
        cv2.circle(frame, (x+5, 42), 5, dc, -1)
        cv2.putText(frame, label, (x+14, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.35, dc, 1)
        x += 80

    cv2.putText(frame, f"FPS:{fps:.0f} Buf:{buf_sz}", (w-140, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)

    if not face_ok:
        cv2.putText(frame, "(lips from pose)", (x+10, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,200,0), 1)

    # Prediction
    if sign and conf > 0.05:
        ov2 = frame.copy()
        cv2.rectangle(ov2, (0, h-120), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(ov2, 0.75, frame, 0.25, 0, frame)

        color = (0,255,100) if conf > 0.4 else (0,220,255) if conf > 0.2 else (150,200,255)

        cv2.putText(frame, sign.upper(), (12, h-85),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
        cv2.putText(frame, f"{conf:.0%}", (w-75, h-85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        for i, (s, c) in enumerate(top5):
            y = h - 60 + i * 14
            bw = max(1, int(c * (w - 20)))
            bc = color if i == 0 else (50, 50, 50)
            cv2.rectangle(frame, (8, y-2), (8+bw, y+9), bc, -1)
            cv2.putText(frame, f"{s} {c:.0%}", (12, y+7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255,255,255), 1)

    cv2.putText(frame, "Q:Quit  C:Clear", (8, h-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80,80,80), 1)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Real-time ASL Sign Recognition")
    parser.add_argument("--model", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "outputs", "landmark_transformer_3d", "best_model.pth"))
    parser.add_argument("--mediapipe-dir", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "mediapipe_models"))
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    # Load model
    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    num_classes = ckpt.get("num_classes", 256)

    class_map_path = os.path.join(os.path.dirname(args.model), "class_map.json")
    if os.path.exists(class_map_path):
        with open(class_map_path) as f:
            class_names = {int(k): v for k, v in json.load(f).items()}
    else:
        class_names = {i: n for i, n in enumerate(ckpt.get("class_names", []))}

    model = LandmarkTransformer(
        num_classes=num_classes,
        max_frames=config.get("max_frames", MAX_FRAMES),
        units=config.get("units", 512),
        num_blocks=config.get("num_blocks", 4),
        num_heads=config.get("num_heads", 8),
        dropout=config.get("dropout", 0.15),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded: {num_classes} classes, {ckpt.get('best_val_acc',0)*100:.1f}% val acc")

    # GPU warmup
    if device.type == "cuda":
        d_f = torch.zeros(1, MAX_FRAMES, NUM_LANDMARKS, NUM_COORDS, device=device)
        d_i = torch.full((1, MAX_FRAMES), -1.0, device=device)
        d_i[0, 0] = 0.0
        with torch.no_grad(), torch.amp.autocast("cuda"):
            model(d_f, d_i)
        del d_f, d_i
        print("GPU warm")

    # MediaPipe
    print("Loading MediaPipe...")
    extractor = LandmarkExtractor(args.mediapipe_dir)

    # Camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("ERROR: camera failed")
        extractor.close()
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Wait for camera warmup
    print("Camera warming up...")
    for _ in range(10):
        cap.read()

    # Test face detection
    ret, test_bgr = cap.read()
    if ret:
        test_rgb = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2RGB)
        _, face_ok = extractor.extract(test_rgb)
        if not face_ok:
            print("NOTE: Face landmarks not detected - using pose estimation for lip region")
            print("      This is normal for some webcams. Predictions will still work.")
        else:
            print("Face landmarks: OK")

    print("\n  READY - Show your hands and do a sign!\n")

    # State
    hand_buffer = collections.deque(maxlen=MAX_FRAMES)

    cur_sign = None
    cur_conf = 0.0
    cur_top5 = []

    avg_probs = None
    smooth_alpha = 0.6

    fps = 0.0
    fps_t = time.time()
    fps_c = 0
    frame_idx = 0
    predict_interval = 8  # predict every N frames (~0.27s at 30fps)

    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        # Extract from ORIGINAL image (not flipped) to match training data
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # FPS
        fps_c += 1
        now = time.time()
        if now - fps_t >= 1.0:
            fps = fps_c / (now - fps_t)
            fps_c = 0
            fps_t = now

        # Extract ALL landmarks every frame
        lm, face_ok = extractor.extract(rgb)

        hands_visible = np.any(lm[40:82] != 0)
        lips_visible = np.any(lm[0:40] != 0)
        pose_visible = np.any(lm[82:92] != 0)

        # Buffer when hands are visible
        if hands_visible:
            hand_buffer.append(lm.copy())

        # Predict continuously
        if frame_idx % predict_interval == 0 and len(hand_buffer) >= 8:
            probs = run_inference(model, list(hand_buffer), device)
            if probs is not None:
                if avg_probs is None:
                    avg_probs = probs
                else:
                    avg_probs = smooth_alpha * probs + (1 - smooth_alpha) * avg_probs

                top_idx = np.argsort(avg_probs)[::-1][:5]
                cur_sign = class_names.get(top_idx[0], "?")
                cur_conf = float(avg_probs[top_idx[0]])
                cur_top5 = [(class_names.get(i, "?"), float(avg_probs[i])) for i in top_idx]

                print(f"\r  >> {cur_sign.upper():15s} {cur_conf:5.0%}  buf={len(hand_buffer):2d}  "
                      f"face={'Y' if face_ok else 'N(pose)'}", end="", flush=True)

        # Flip for display only
        display = cv2.flip(bgr, 1)
        display_lm = flip_lm_for_display(lm)

        draw_landmarks(display, display_lm)
        draw_ui(display, cur_sign, cur_conf, cur_top5,
                face_ok or lips_visible, hands_visible, pose_visible,
                len(hand_buffer), fps)

        cv2.imshow("SIGNFLOW", display)
        frame_idx += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('c') or key == ord('C'):
            hand_buffer.clear()
            avg_probs = None
            cur_sign = None
            cur_conf = 0.0
            cur_top5 = []
            print("\r  [Cleared]" + " "*60, end="", flush=True)

    print()
    cap.release()
    cv2.destroyAllWindows()
    extractor.close()
    print("Done.")


if __name__ == "__main__":
    main()
