"""
Real-time ASL Sign Language Recognition - Live Continuous Prediction.

Shows hands -> instant prediction. No buttons, no recording, no waiting.
Predicts continuously every 0.5 seconds on the last ~2 seconds of frames.

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
# Fast landmark extraction
# ---------------------------------------------------------------
class LandmarkExtractor:
    def __init__(self, model_dir):
        self.face_lm = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=os.path.join(model_dir, "face_landmarker.task")),
            running_mode=RunningMode.IMAGE, num_faces=1,
            min_face_detection_confidence=0.25, min_face_presence_confidence=0.25,
        ))
        self.hand_lm = HandLandmarker.create_from_options(HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=os.path.join(model_dir, "hand_landmarker.task")),
            running_mode=RunningMode.IMAGE, num_hands=2,
            min_hand_detection_confidence=0.25, min_hand_presence_confidence=0.25,
        ))
        self.pose_lm = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=os.path.join(model_dir, "pose_landmarker_heavy.task")),
            running_mode=RunningMode.IMAGE, num_poses=1,
            min_pose_detection_confidence=0.25, min_pose_presence_confidence=0.25,
        ))
        self._cached_face_pose = np.zeros((NUM_LANDMARKS, NUM_COORDS), dtype=np.float32)

    def extract_all(self, rgb):
        """Full extraction: face + hands + pose."""
        mp_img = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=rgb)
        lm = np.zeros((NUM_LANDMARKS, NUM_COORDS), dtype=np.float32)

        fr = self.face_lm.detect(mp_img)
        if fr.face_landmarks and len(fr.face_landmarks) > 0:
            face = fr.face_landmarks[0]
            for i, fi in enumerate(LIPS_FACE_IDXS):
                if fi < len(face):
                    lm[i] = [face[fi].x, face[fi].y, face[fi].z]

        hr = self.hand_lm.detect(mp_img)
        if hr.hand_landmarks:
            for hi, (hm, hn) in enumerate(zip(hr.hand_landmarks, hr.handedness)):
                if hi >= 2:
                    break
                label = hn[0].category_name.lower() if hn else "left"
                offset = 40 if label == "left" else 61
                for j in range(min(21, len(hm))):
                    lm[offset + j] = [hm[j].x, hm[j].y, hm[j].z]

        pr = self.pose_lm.detect(mp_img)
        if pr.pose_landmarks and len(pr.pose_landmarks) > 0:
            pose = pr.pose_landmarks[0]
            for k, pidx in enumerate(POSE_UPPER_IDXS):
                if pidx < len(pose):
                    lm[82 + k] = [pose[pidx].x, pose[pidx].y, pose[pidx].z]

        # Cache face+pose for fast frames
        self._cached_face_pose[:40] = lm[:40]
        self._cached_face_pose[82:] = lm[82:]
        return lm

    def extract_hands_fast(self, rgb):
        """Fast path: only hands (reuse cached face+pose)."""
        mp_img = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=rgb)
        lm = self._cached_face_pose.copy()
        lm[40:82] = 0

        hr = self.hand_lm.detect(mp_img)
        if hr.hand_landmarks:
            for hi, (hm, hn) in enumerate(zip(hr.hand_landmarks, hr.handedness)):
                if hi >= 2:
                    break
                label = hn[0].category_name.lower() if hn else "left"
                offset = 40 if label == "left" else 61
                for j in range(min(21, len(hm))):
                    lm[offset + j] = [hm[j].x, hm[j].y, hm[j].z]
        return lm

    def close(self):
        self.face_lm.close()
        self.hand_lm.close()
        self.pose_lm.close()


# ---------------------------------------------------------------
# Inference
# ---------------------------------------------------------------
def run_inference(model, frames_list, device):
    """Run model on a list of [92, 3] landmark arrays. Returns probabilities."""
    n = len(frames_list)
    if n < 5:
        return None

    # Subsample to MAX_FRAMES if we have more
    if n > MAX_FRAMES:
        indices = np.linspace(0, n - 1, MAX_FRAMES, dtype=int)
        frames_list = [frames_list[i] for i in indices]
        n = MAX_FRAMES

    arr = np.stack(frames_list, axis=0).astype(np.float32)

    # Pad to MAX_FRAMES
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
def draw_hand(frame, landmarks, offset, color, w, h):
    for i in range(21):
        x, y = landmarks[offset + i, 0], landmarks[offset + i, 1]
        if x > 0 or y > 0:
            cv2.circle(frame, (int(x * w), int(y * h)), 3, color, -1)
    for a, b in HAND_CONNS:
        x1, y1 = landmarks[offset + a, 0], landmarks[offset + a, 1]
        x2, y2 = landmarks[offset + b, 0], landmarks[offset + b, 1]
        if (x1 > 0 or y1 > 0) and (x2 > 0 or y2 > 0):
            cv2.line(frame, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), color, 2)


def draw_all(frame, lm, sign, conf, top3, hands_visible, buf_size, fps):
    h, w = frame.shape[:2]

    # Draw landmarks
    draw_hand(frame, lm, 40, (255, 150, 0), w, h)   # left hand blue
    draw_hand(frame, lm, 61, (0, 130, 255), w, h)    # right hand orange

    # Top bar
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, "ASL Recognition", (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    status = f"FPS:{fps:.0f} | Buf:{buf_size}"
    cv2.putText(frame, status, (w - 160, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    if hands_visible:
        cv2.circle(frame, (w - 15, 20), 6, (0, 255, 0), -1)
    else:
        cv2.circle(frame, (w - 15, 20), 6, (0, 0, 150), -1)

    # Prediction
    if sign and conf > 0.08:
        ov2 = frame.copy()
        cv2.rectangle(ov2, (0, h - 100), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(ov2, 0.7, frame, 0.3, 0, frame)

        color = (0, 255, 100) if conf > 0.4 else (0, 220, 255) if conf > 0.2 else (150, 200, 255)

        # Big sign name
        cv2.putText(frame, sign.upper(), (12, h - 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(frame, f"{conf:.0%}", (w - 70, h - 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Top 3 bars
        for i, (s, c) in enumerate(top3):
            y = h - 42 + i * 16
            bw = int(c * (w - 20))
            bc = color if i == 0 else (50, 50, 50)
            cv2.rectangle(frame, (8, y - 2), (8 + bw, y + 10), bc, -1)
            cv2.putText(frame, f"{s} {c:.0%}", (12, y + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    cv2.putText(frame, "Q:Quit  C:Clear", (8, h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 80), 1)


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

    # ---- Load model ----
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

    # Warmup
    if device.type == "cuda":
        d_f = torch.zeros(1, MAX_FRAMES, NUM_LANDMARKS, NUM_COORDS, device=device)
        d_i = torch.full((1, MAX_FRAMES), -1.0, device=device)
        d_i[0, 0] = 0.0
        with torch.no_grad(), torch.amp.autocast("cuda"):
            model(d_f, d_i)
        del d_f, d_i
        print("GPU warm")

    # ---- MediaPipe ----
    print("Loading MediaPipe...")
    extractor = LandmarkExtractor(args.mediapipe_dir)

    # ---- Camera ----
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("ERROR: camera failed")
        extractor.close()
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("\n  READY - Just show your hands and do a sign!\n")

    # ---- State ----
    # Sliding window: keep last ~2 sec of frames where hands were visible
    hand_buffer = collections.deque(maxlen=MAX_FRAMES)

    # Prediction state
    cur_sign = None
    cur_conf = 0.0
    cur_top3 = []

    # Smoothing: accumulate probability over recent predictions
    avg_probs = None
    smooth_alpha = 0.4  # weight for new prediction (higher = more responsive)

    # Timing
    fps = 0.0
    fps_t = time.time()
    fps_c = 0
    frame_idx = 0
    predict_interval = 12  # predict every N frames (~0.4s at 30fps)

    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        bgr = cv2.flip(bgr, 1)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # FPS
        fps_c += 1
        now = time.time()
        if now - fps_t >= 1.0:
            fps = fps_c / (now - fps_t)
            fps_c = 0
            fps_t = now

        # Extract landmarks (full every 3 frames, hands-only otherwise)
        if frame_idx % 3 == 0:
            lm = extractor.extract_all(rgb)
        else:
            lm = extractor.extract_hands_fast(rgb)

        # Check if hands visible
        hands_visible = np.any(lm[40:82] != 0)

        # Add to buffer only if hands detected
        if hands_visible:
            hand_buffer.append(lm.copy())

        # Run prediction every predict_interval frames (if we have enough data)
        if frame_idx % predict_interval == 0 and len(hand_buffer) >= 8:
            probs = run_inference(model, list(hand_buffer), device)
            if probs is not None:
                # Exponential moving average for smoothing
                if avg_probs is None:
                    avg_probs = probs
                else:
                    avg_probs = smooth_alpha * probs + (1 - smooth_alpha) * avg_probs

                top_idx = np.argsort(avg_probs)[::-1][:3]
                cur_sign = class_names.get(top_idx[0], "?")
                cur_conf = avg_probs[top_idx[0]]
                cur_top3 = [(class_names.get(i, "?"), float(avg_probs[i])) for i in top_idx]

                print(f"\r  >> {cur_sign.upper():15s} {cur_conf:5.0%}  "
                      f"(buf={len(hand_buffer):2d})", end="", flush=True)

        # If no hands for a while, decay the buffer
        if not hands_visible and len(hand_buffer) > 0:
            # After hands disappear, keep the buffer for a bit to show last prediction
            # But don't add empty frames
            pass

        # Draw
        draw_all(bgr, lm, cur_sign, cur_conf, cur_top3,
                 hands_visible, len(hand_buffer), fps)
        cv2.imshow("ASL Sign Recognition", bgr)

        frame_idx += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('c') or key == ord('C'):
            hand_buffer.clear()
            avg_probs = None
            cur_sign = None
            cur_conf = 0.0
            cur_top3 = []
            print("\r  [Cleared]" + " " * 40, end="", flush=True)

    print()
    cap.release()
    cv2.destroyAllWindows()
    extractor.close()
    print("Done.")


if __name__ == "__main__":
    main()
