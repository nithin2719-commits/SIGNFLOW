"""
Real-time ASL Sign Language Recognition from Webcam.

Uses MediaPipe Tasks API for landmark extraction and a trained
3D Landmark Transformer for sign classification (256 classes).

Runs continuous prediction with a sliding window - no button pressing needed.
Just perform a sign in front of the camera and see the result.

Controls:
  Q - Quit
  C - Clear current prediction
  SPACE - Force predict now on current buffer

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
import threading

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

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


# ---------------------------------------------------------------
# Model architecture (identical to train_landmark_transformer.py)
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
        out = torch.where(mask, self.empty_embedding, out)
        return out


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
        max_units = max(lips_emb.shape[-1], lh_emb.shape[-1],
                        rh_emb.shape[-1], pose_emb.shape[-1])
        if lips_emb.shape[-1] < max_units:
            lips_emb = F.pad(lips_emb, (0, max_units - lips_emb.shape[-1]))
        if lh_emb.shape[-1] < max_units:
            lh_emb = F.pad(lh_emb, (0, max_units - lh_emb.shape[-1]))
        if rh_emb.shape[-1] < max_units:
            rh_emb = F.pad(rh_emb, (0, max_units - rh_emb.shape[-1]))
        if pose_emb.shape[-1] < max_units:
            pose_emb = F.pad(pose_emb, (0, max_units - pose_emb.shape[-1]))
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
        fused = fused + self.positional_embedding(pos_indices)
        return fused


class TransformerBlock(nn.Module):
    def __init__(self, units, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(units)
        self.attn = nn.MultiheadAttention(units, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(units)
        self.mlp = nn.Sequential(
            nn.Linear(units, units * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(units * mlp_ratio, units),
            nn.Dropout(dropout),
        )

    def forward(self, x, key_padding_mask=None):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, key_padding_mask=key_padding_mask)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


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
        key_padding_mask = (non_empty_frame_idxs == -1.0)
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        mask = (~key_padding_mask).unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp(min=1e-6)
        x = (x * mask).sum(dim=1) / denom
        x = self.head_dropout(x)
        return self.classifier(x)


# ---------------------------------------------------------------
# Landmark extraction (optimized for real-time)
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
        self._last_lm = np.zeros((NUM_LANDMARKS, NUM_COORDS), dtype=np.float32)

    def extract(self, rgb_frame):
        mp_img = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=rgb_frame)
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

        self._last_lm = lm
        return lm

    def extract_hands_only(self, rgb_frame):
        """Fast path: only detect hands (skip face/pose for speed)."""
        mp_img = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=rgb_frame)
        lm = self._last_lm.copy()  # reuse last face+pose

        # Clear hands and re-detect
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

        self._last_lm = lm
        return lm

    def close(self):
        self.face_lm.close()
        self.hand_lm.close()
        self.pose_lm.close()


# ---------------------------------------------------------------
# Sign state machine - detects sign boundaries
# ---------------------------------------------------------------
class SignDetector:
    """Detects when a sign starts and ends based on hand motion."""

    def __init__(self):
        self.state = "idle"  # idle -> signing -> cooldown -> idle
        self.frames = []
        self.no_hand_count = 0
        self.hand_count = 0
        self.prev_hand_center = None
        self.motion_energy = 0.0
        self.still_count = 0

    def _hand_center(self, landmarks):
        """Get average hand position from landmarks."""
        hands = landmarks[40:82]  # both hands
        nonzero = hands[np.any(hands != 0, axis=1)]
        if len(nonzero) == 0:
            return None
        return nonzero[:, :2].mean(axis=0)

    def _has_hands(self, landmarks):
        return np.any(landmarks[40:82] != 0)

    def update(self, landmarks):
        """Update state machine. Returns (event, frames) where event is
        None, 'recording', or 'predict'."""
        has_hands = self._has_hands(landmarks)
        center = self._hand_center(landmarks)

        # Compute motion
        if center is not None and self.prev_hand_center is not None:
            motion = np.linalg.norm(center - self.prev_hand_center)
            self.motion_energy = 0.7 * self.motion_energy + 0.3 * motion
        else:
            self.motion_energy *= 0.8
        self.prev_hand_center = center

        if self.state == "idle":
            if has_hands:
                self.hand_count += 1
                if self.hand_count >= 3:  # hands visible for 3 frames
                    self.state = "signing"
                    self.frames = []
                    self.still_count = 0
                    self.no_hand_count = 0
            else:
                self.hand_count = 0
            return None, []

        elif self.state == "signing":
            if has_hands:
                self.frames.append(landmarks.copy())
                self.no_hand_count = 0

                # Check if hands stopped moving (sign ended)
                if self.motion_energy < 0.003 and len(self.frames) > 15:
                    self.still_count += 1
                else:
                    self.still_count = 0

                # Sign ended: hands still for a while OR buffer full
                if self.still_count >= 12 or len(self.frames) >= MAX_FRAMES * 2:
                    self.state = "cooldown"
                    result_frames = self.frames.copy()
                    self.frames = []
                    self.still_count = 0
                    self.no_hand_count = 0
                    self.hand_count = 0
                    return "predict", result_frames

                return "recording", []
            else:
                self.no_hand_count += 1
                # Hands gone for a few frames - sign ended
                if self.no_hand_count >= 5 and len(self.frames) >= 5:
                    self.state = "cooldown"
                    result_frames = self.frames.copy()
                    self.frames = []
                    self.no_hand_count = 0
                    self.hand_count = 0
                    return "predict", result_frames
                elif self.no_hand_count >= 15:
                    # Too long without hands, discard
                    self.state = "idle"
                    self.frames = []
                    self.hand_count = 0
                return "recording", []

        elif self.state == "cooldown":
            if not has_hands:
                self.no_hand_count += 1
                if self.no_hand_count >= 8:
                    self.state = "idle"
                    self.no_hand_count = 0
                    self.hand_count = 0
            else:
                # Hands still visible during cooldown, go back to signing
                self.no_hand_count = 0
                self.hand_count += 1
                if self.hand_count >= 5:
                    self.state = "signing"
                    self.frames = []
                    self.still_count = 0
            return None, []

        return None, []


# ---------------------------------------------------------------
# Inference
# ---------------------------------------------------------------
def subsample_frames(frames_list, target_count=MAX_FRAMES):
    """Subsample frames to target_count, matching training pipeline."""
    n = len(frames_list)
    if n <= target_count:
        return frames_list
    indices = np.linspace(0, n - 1, target_count, dtype=int)
    return [frames_list[i] for i in indices]


def predict_sign(model, frames_list, device, class_names, use_amp=True):
    """Run model inference on a list of landmark frames."""
    if len(frames_list) < 5:
        return None, 0.0, []

    # Subsample to MAX_FRAMES (matching training data pipeline)
    sampled = subsample_frames(frames_list, MAX_FRAMES)
    landmarks = np.stack(sampled, axis=0).astype(np.float32)
    num_frames = landmarks.shape[0]

    # Pad to MAX_FRAMES
    if num_frames < MAX_FRAMES:
        pad = np.zeros((MAX_FRAMES - num_frames, NUM_LANDMARKS, NUM_COORDS), dtype=np.float32)
        landmarks = np.concatenate([landmarks, pad], axis=0)

    # Non-empty frame indices
    non_empty = np.any(landmarks != 0, axis=(1, 2))
    non_empty_idxs = np.where(non_empty, np.arange(MAX_FRAMES, dtype=np.float32), -1.0)

    frames_t = torch.from_numpy(landmarks).unsqueeze(0).to(device)
    idxs_t = torch.from_numpy(non_empty_idxs).unsqueeze(0).to(device)

    with torch.no_grad():
        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(frames_t, idxs_t)
        else:
            logits = model(frames_t, idxs_t)
        probs = torch.softmax(logits, dim=1)[0].cpu()

    top5_probs, top5_indices = probs.topk(5)
    top5 = [(class_names[idx.item()], prob.item()) for idx, prob in zip(top5_indices, top5_probs)]
    return top5[0][0], top5[0][1], top5


# ---------------------------------------------------------------
# Prediction smoother
# ---------------------------------------------------------------
class PredictionSmoother:
    """Smooth predictions using a short history of recent results."""

    def __init__(self, history_size=3):
        self.history = collections.deque(maxlen=history_size)
        self.current_sign = None
        self.current_conf = 0.0
        self.current_top5 = []

    def update(self, sign, conf, top5):
        if sign is None:
            return
        self.history.append((sign, conf, top5))

        # Vote across recent predictions
        votes = {}
        for s, c, _ in self.history:
            votes[s] = votes.get(s, 0) + c
        best_sign = max(votes, key=votes.get)

        # Find the most recent top5 for the best sign
        for s, c, t5 in reversed(self.history):
            if s == best_sign:
                self.current_sign = s
                self.current_conf = c
                self.current_top5 = t5
                break

    def clear(self):
        self.history.clear()
        self.current_sign = None
        self.current_conf = 0.0
        self.current_top5 = []


# ---------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------
def draw_landmarks(frame, landmarks):
    h, w = frame.shape[:2]

    # Lips (green dots)
    for i in range(40):
        x, y = landmarks[i, 0], landmarks[i, 1]
        if x > 0 or y > 0:
            cv2.circle(frame, (int(x * w), int(y * h)), 1, (0, 255, 0), -1)

    # Left hand (blue)
    for i in range(40, 61):
        x, y = landmarks[i, 0], landmarks[i, 1]
        if x > 0 or y > 0:
            cv2.circle(frame, (int(x * w), int(y * h)), 3, (255, 150, 0), -1)
    for a, b in HAND_CONNECTIONS:
        x1, y1 = landmarks[40 + a, 0], landmarks[40 + a, 1]
        x2, y2 = landmarks[40 + b, 0], landmarks[40 + b, 1]
        if (x1 > 0 or y1 > 0) and (x2 > 0 or y2 > 0):
            cv2.line(frame, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), (255, 150, 0), 2)

    # Right hand (orange-red)
    for i in range(61, 82):
        x, y = landmarks[i, 0], landmarks[i, 1]
        if x > 0 or y > 0:
            cv2.circle(frame, (int(x * w), int(y * h)), 3, (0, 130, 255), -1)
    for a, b in HAND_CONNECTIONS:
        x1, y1 = landmarks[61 + a, 0], landmarks[61 + a, 1]
        x2, y2 = landmarks[61 + b, 0], landmarks[61 + b, 1]
        if (x1 > 0 or y1 > 0) and (x2 > 0 or y2 > 0):
            cv2.line(frame, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), (0, 130, 255), 2)

    # Pose (yellow)
    for i in range(82, 92):
        x, y = landmarks[i, 0], landmarks[i, 1]
        if x > 0 or y > 0:
            cv2.circle(frame, (int(x * w), int(y * h)), 4, (0, 255, 255), -1)


def draw_ui(frame, smoother, state, frame_count, fps, threshold):
    h, w = frame.shape[:2]

    # Top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, "ASL Sign Recognition", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 100, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # State indicator
    if state == "signing":
        # Recording state
        if int(time.time() * 3) % 2 == 0:
            cv2.circle(frame, (15, 45), 6, (0, 0, 255), -1)
        cv2.putText(frame, f"Recording... ({frame_count} frames)", (28, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    elif state == "cooldown":
        cv2.putText(frame, "Processing...", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1)
    else:
        cv2.putText(frame, "Show your hands to start", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

    # Prediction result (bottom)
    sign = smoother.current_sign
    conf = smoother.current_conf
    top5 = smoother.current_top5

    if sign and conf >= threshold:
        bar_h = 30 + len(top5) * 22
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, h - bar_h - 10), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay2, 0.75, frame, 0.25, 0, frame)

        # Main prediction - large text
        if conf > 0.5:
            color = (0, 255, 100)
        elif conf > 0.3:
            color = (0, 220, 255)
        else:
            color = (100, 180, 255)

        cv2.putText(frame, sign.upper(), (10, h - bar_h + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(frame, f"{conf:.0%}", (w - 60, h - bar_h + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Top-5 bars
        for i, (s, c) in enumerate(top5):
            y = h - bar_h + 35 + i * 22
            bar_len = int(c * (w - 130))
            bar_color = color if i == 0 else (60, 60, 60)
            cv2.rectangle(frame, (10, y - 2), (10 + bar_len, y + 14), bar_color, -1)
            cv2.putText(frame, f"{s} ({c:.0%})", (15, y + 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Controls
    cv2.putText(frame, "SPACE: Predict | C: Clear | Q: Quit", (5, h - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)


# ---------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Real-time ASL Sign Recognition")
    parser.add_argument("--model", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "outputs", "landmark_transformer_3d", "best_model.pth"))
    parser.add_argument("--mediapipe-dir", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "mediapipe_models"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.10,
                        help="Min confidence to display prediction")
    args = parser.parse_args()

    # Load model
    print("Loading model...")
    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)

    config = checkpoint.get("config", {})
    num_classes = checkpoint.get("num_classes", 256)

    class_map_path = os.path.join(os.path.dirname(args.model), "class_map.json")
    if os.path.exists(class_map_path):
        with open(class_map_path) as f:
            class_names = {int(k): v for k, v in json.load(f).items()}
    else:
        class_names = {i: name for i, name in enumerate(checkpoint.get("class_names", []))}

    model = LandmarkTransformer(
        num_classes=num_classes,
        max_frames=config.get("max_frames", MAX_FRAMES),
        units=config.get("units", 512),
        num_blocks=config.get("num_blocks", 4),
        num_heads=config.get("num_heads", 8),
        dropout=config.get("dropout", 0.15),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Model: {num_classes} classes, val acc: {checkpoint.get('best_val_acc', 0)*100:.1f}%")

    # Warmup inference (first run is slow due to CUDA compilation)
    if device.type == "cuda":
        dummy_f = torch.zeros(1, MAX_FRAMES, NUM_LANDMARKS, NUM_COORDS, device=device)
        dummy_i = torch.full((1, MAX_FRAMES), -1.0, device=device)
        dummy_i[0, 0] = 0.0
        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                model(dummy_f, dummy_i)
        print("CUDA warmup done")

    # Load MediaPipe
    print("Loading MediaPipe...")
    if not os.path.exists(args.mediapipe_dir):
        print(f"ERROR: MediaPipe models not found: {args.mediapipe_dir}")
        sys.exit(1)
    extractor = LandmarkExtractor(args.mediapipe_dir)

    # Open webcam
    print(f"Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        extractor.close()
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print()
    print("=" * 45)
    print("  ASL Sign Recognition - READY")
    print("  Just show your hands and perform a sign!")
    print("  Q=Quit  C=Clear  SPACE=Force predict")
    print("=" * 45)
    print()

    detector = SignDetector()
    smoother = PredictionSmoother(history_size=3)

    fps = 0.0
    fps_time = time.time()
    fps_count = 0
    frame_idx = 0
    manual_buffer = []
    manual_recording = False

    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        bgr = cv2.flip(bgr, 1)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # FPS
        fps_count += 1
        now = time.time()
        if now - fps_time >= 1.0:
            fps = fps_count / (now - fps_time)
            fps_count = 0
            fps_time = now

        # Extract landmarks
        # Full extraction every 3rd frame, hands-only otherwise (faster)
        if frame_idx % 3 == 0:
            landmarks = extractor.extract(rgb)
        else:
            landmarks = extractor.extract_hands_only(rgb)
        frame_idx += 1

        # Manual recording mode
        if manual_recording:
            if np.any(landmarks != 0):
                manual_buffer.append(landmarks.copy())
            draw_landmarks(bgr, landmarks)
            draw_ui(bgr, smoother, "signing", len(manual_buffer), fps, args.threshold)
            cv2.imshow("ASL Sign Recognition", bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                # Force predict
                manual_recording = False
                if len(manual_buffer) >= 5:
                    sign, conf, top5 = predict_sign(model, manual_buffer, device, class_names)
                    if sign:
                        smoother.update(sign, conf, top5)
                        print(f">> {sign.upper()} ({conf:.0%})")
                manual_buffer = []
            elif key == ord('q') or key == ord('Q'):
                break
            elif key == ord('c') or key == ord('C'):
                manual_recording = False
                manual_buffer = []
                smoother.clear()
            continue

        # Auto detection
        event, frames = detector.update(landmarks)

        if event == "predict" and len(frames) >= 5:
            sign, conf, top5 = predict_sign(model, frames, device, class_names)
            if sign:
                smoother.update(sign, conf, top5)
                print(f">> {sign.upper()} ({conf:.0%})")

        # Draw
        draw_landmarks(bgr, landmarks)
        draw_ui(bgr, smoother, detector.state,
                len(detector.frames), fps, args.threshold)

        cv2.imshow("ASL Sign Recognition", bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('c') or key == ord('C'):
            smoother.clear()
            detector = SignDetector()
        elif key == ord(' '):
            # Force predict on whatever we have
            if len(detector.frames) >= 5:
                sign, conf, top5 = predict_sign(
                    model, detector.frames, device, class_names)
                if sign:
                    smoother.update(sign, conf, top5)
                    print(f">> {sign.upper()} ({conf:.0%})")
                detector = SignDetector()
            else:
                # Start manual recording
                manual_recording = True
                manual_buffer = []
                print("Manual recording - press SPACE again to predict")

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()
    print("Done.")


if __name__ == "__main__":
    main()
