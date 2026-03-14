"""
Standalone webcam real-time ASL prediction using the 3D Landmark Transformer.

Usage:
  python predict_realtime.py
  python predict_realtime.py --camera 1

Controls: Q = Quit, ESC = Quit, C = Clear
"""

import argparse
import collections
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

import mediapipe as mp_lib
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker, FaceLandmarkerOptions,
    HandLandmarker, HandLandmarkerOptions,
    PoseLandmarker, PoseLandmarkerOptions,
    RunningMode,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pth")
CLASS_MAP_PATH = os.path.join(BASE_DIR, "models", "class_map.json")
MP_MODEL_DIR = os.path.join(BASE_DIR, "mediapipe_models")

MAX_FRAMES = 64
NUM_LANDMARKS = 92
NUM_COORDS = 3
PREDICT_INTERVAL = 4
SMOOTH_ALPHA = 0.6
MIN_BUFFER = 8
CONFIDENCE_THRESHOLD = 0.15

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


# ---------------------------------------------------------------
# Model architecture (must match training)
# ---------------------------------------------------------------
class LandmarkEmbedding(nn.Module):
    def __init__(self, in_features, units):
        super().__init__()
        self.empty_embedding = nn.Parameter(torch.zeros(units))
        self.proj = nn.Sequential(
            nn.Linear(in_features, units, bias=False), nn.GELU(),
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
            nn.GELU(), nn.Linear(units, units, bias=False),
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
        mu = max(lips_emb.shape[-1], lh_emb.shape[-1], rh_emb.shape[-1], pose_emb.shape[-1])
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
        self.embedding = LandmarkTransformerEmbedding(max_frames, units, lips_units, hands_units, pose_units)
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.blocks = nn.ModuleList([TransformerBlock(units, num_heads, mlp_ratio, dropout) for _ in range(num_blocks)])
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
# Main
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Real-time ASL Sign Recognition")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    num_classes = ckpt.get("num_classes", 256)

    with open(CLASS_MAP_PATH) as f:
        class_names = {int(k): v for k, v in json.load(f).items()}

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
    print(f"Loaded: {num_classes} classes")

    if device.type == "cuda":
        d_f = torch.zeros(1, MAX_FRAMES, NUM_LANDMARKS, NUM_COORDS, device=device)
        d_i = torch.full((1, MAX_FRAMES), -1.0, device=device)
        d_i[0, 0] = 0.0
        with torch.no_grad(), torch.amp.autocast("cuda"):
            model(d_f, d_i)
        del d_f, d_i
        print("GPU warm")

    print("Loading MediaPipe...")
    face_lm = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(MP_MODEL_DIR, "face_landmarker.task")),
        running_mode=RunningMode.IMAGE, num_faces=1,
        min_face_detection_confidence=0.1, min_face_presence_confidence=0.1,
    ))
    hand_lm = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(MP_MODEL_DIR, "hand_landmarker.task")),
        running_mode=RunningMode.IMAGE, num_hands=2,
        min_hand_detection_confidence=0.2, min_hand_presence_confidence=0.2,
    ))
    pose_lm = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(MP_MODEL_DIR, "pose_landmarker_heavy.task")),
        running_mode=RunningMode.IMAGE, num_poses=1,
        min_pose_detection_confidence=0.2, min_pose_presence_confidence=0.2,
    ))

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("ERROR: camera failed")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    for _ in range(10):
        cap.read()

    print("\n  READY - Show signs!\n")

    hand_buffer = collections.deque(maxlen=MAX_FRAMES)
    avg_probs = None
    cur_sign = None
    cur_conf = 0.0
    frame_idx = 0
    fps = 0.0
    fps_t = time.time()
    fps_c = 0

    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        lm = np.zeros((NUM_LANDMARKS, NUM_COORDS), dtype=np.float32)

        # Face
        fr = face_lm.detect(mp_img)
        if fr.face_landmarks:
            face = fr.face_landmarks[0]
            for i, fi in enumerate(LIPS_FACE_IDXS):
                if fi < len(face):
                    lm[i] = [face[fi].x, face[fi].y, face[fi].z]

        # Hands
        hr = hand_lm.detect(mp_img)
        if hr.hand_landmarks:
            for hi, (hm, hn) in enumerate(zip(hr.hand_landmarks, hr.handedness)):
                if hi >= 2:
                    break
                label = hn[0].category_name.lower() if hn else "left"
                offset = 40 if label == "left" else 61
                for j in range(min(21, len(hm))):
                    lm[offset + j] = [hm[j].x, hm[j].y, hm[j].z]

        # Pose
        pr = pose_lm.detect(mp_img)
        if pr.pose_landmarks:
            pose = pr.pose_landmarks[0]
            for k, pidx in enumerate(POSE_UPPER_IDXS):
                if pidx < len(pose):
                    lm[82 + k] = [pose[pidx].x, pose[pidx].y, pose[pidx].z]

        hands_visible = np.any(lm[40:82] != 0)
        if hands_visible:
            hand_buffer.append(lm.copy())

        fps_c += 1
        now = time.time()
        if now - fps_t >= 1.0:
            fps = fps_c / (now - fps_t)
            fps_c = 0
            fps_t = now

        if frame_idx % PREDICT_INTERVAL == 0 and len(hand_buffer) >= MIN_BUFFER:
            n = len(hand_buffer)
            frames_list = list(hand_buffer)
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
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

            if avg_probs is None:
                avg_probs = probs
            else:
                avg_probs = SMOOTH_ALPHA * probs + (1 - SMOOTH_ALPHA) * avg_probs

            top_idx = int(np.argmax(avg_probs))
            cur_sign = class_names.get(top_idx, "?")
            cur_conf = float(avg_probs[top_idx])

            print(f"\r  >> {cur_sign.upper():15s} {cur_conf:5.0%}  buf={len(hand_buffer):2d}", end="", flush=True)

        # Display
        display = cv2.flip(bgr, 1)
        h, w = display.shape[:2]

        # Draw landmarks on flipped display
        for i in range(40):
            x, y = lm[i, 0], lm[i, 1]
            if x > 0 or y > 0:
                cv2.circle(display, (int((1.0 - x) * w), int(y * h)), 2, (0, 255, 0), -1)
        for offset, color in [(40, (255, 150, 0)), (61, (0, 130, 255))]:
            for i in range(21):
                x, y = lm[offset + i, 0], lm[offset + i, 1]
                if x > 0 or y > 0:
                    cv2.circle(display, (int((1.0 - x) * w), int(y * h)), 3, color, -1)
            for a, b in HAND_CONNS:
                x1, y1 = lm[offset + a, 0], lm[offset + a, 1]
                x2, y2 = lm[offset + b, 0], lm[offset + b, 1]
                if (x1 > 0 or y1 > 0) and (x2 > 0 or y2 > 0):
                    cv2.line(display, (int((1.0 - x1) * w), int(y1 * h)),
                             (int((1.0 - x2) * w), int(y2 * h)), color, 2)

        if cur_sign and cur_conf > 0.05:
            color = (0, 255, 100) if cur_conf > 0.4 else (0, 220, 255)
            cv2.putText(display, f"{cur_sign.upper()} {cur_conf:.0%}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        cv2.putText(display, f"FPS:{fps:.0f} Buf:{len(hand_buffer)}", (w - 180, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(display, "Q:Quit  C:Clear", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        cv2.imshow("SignFlow - ASL Recognition", display)
        frame_idx += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:
            break
        elif key == ord('c') or key == ord('C'):
            hand_buffer.clear()
            avg_probs = None
            cur_sign = None
            cur_conf = 0.0

    print()
    cap.release()
    cv2.destroyAllWindows()
    face_lm.close()
    hand_lm.close()
    pose_lm.close()


if __name__ == "__main__":
    main()
