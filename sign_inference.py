"""
Real-time ASL Sign Language Recognition from Webcam.

Uses MediaPipe Tasks API for landmark extraction and a trained
3D Landmark Transformer for sign classification (256 classes).

Controls:
  - Press 'r' to start/stop recording a sign
  - Press 'q' to quit
  - Press 'c' to clear the current prediction

Usage:
  python sign_inference.py
  python sign_inference.py --model path/to/best_model.pth
  python sign_inference.py --camera 1          # use a different camera
  python sign_inference.py --threshold 0.3     # lower confidence threshold
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

# ═══════════════════════════════════════════════════════════
# Constants (must match training)
# ═══════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════
# Model (copied from train_landmark_transformer.py)
# ═══════════════════════════════════════════════════════════
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
            max_frames, units, lips_units, hands_units, pose_units
        )
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(units, num_heads, mlp_ratio, dropout)
            for _ in range(num_blocks)
        ])
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


# ═══════════════════════════════════════════════════════════
# Landmark Extraction (real-time)
# ═══════════════════════════════════════════════════════════
class LandmarkExtractor:
    """Extracts 92 landmarks (lips+hands+pose) from a single frame using MediaPipe Tasks API."""

    def __init__(self, model_dir):
        self.face_lm = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=os.path.join(model_dir, "face_landmarker.task")),
            running_mode=RunningMode.IMAGE, num_faces=1,
            min_face_detection_confidence=0.3, min_face_presence_confidence=0.3,
        ))
        self.hand_lm = HandLandmarker.create_from_options(HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=os.path.join(model_dir, "hand_landmarker.task")),
            running_mode=RunningMode.IMAGE, num_hands=2,
            min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3,
        ))
        self.pose_lm = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=os.path.join(model_dir, "pose_landmarker_heavy.task")),
            running_mode=RunningMode.IMAGE, num_poses=1,
            min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3,
        ))

    def extract(self, rgb_frame):
        """Extract 92 landmarks from an RGB frame. Returns [92, 3] numpy array."""
        mp_img = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=rgb_frame)
        lm = np.zeros((NUM_LANDMARKS, NUM_COORDS), dtype=np.float32)

        # Face -> lips
        fr = self.face_lm.detect(mp_img)
        if fr.face_landmarks and len(fr.face_landmarks) > 0:
            face = fr.face_landmarks[0]
            for i, fi in enumerate(LIPS_FACE_IDXS):
                if fi < len(face):
                    lm[i] = [face[fi].x, face[fi].y, face[fi].z]

        # Hands
        hr = self.hand_lm.detect(mp_img)
        if hr.hand_landmarks:
            for hi, (hm, hn) in enumerate(zip(hr.hand_landmarks, hr.handedness)):
                if hi >= 2:
                    break
                label = hn[0].category_name.lower() if hn else "left"
                offset = 40 if label == "left" else 61
                for j in range(min(21, len(hm))):
                    lm[offset + j] = [hm[j].x, hm[j].y, hm[j].z]

        # Pose
        pr = self.pose_lm.detect(mp_img)
        if pr.pose_landmarks and len(pr.pose_landmarks) > 0:
            pose = pr.pose_landmarks[0]
            for k, pidx in enumerate(POSE_UPPER_IDXS):
                if pidx < len(pose):
                    lm[82 + k] = [pose[pidx].x, pose[pidx].y, pose[pidx].z]

        return lm

    def has_hands(self, rgb_frame):
        """Quick check if hands are visible in frame."""
        mp_img = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=rgb_frame)
        hr = self.hand_lm.detect(mp_img)
        return bool(hr.hand_landmarks and len(hr.hand_landmarks) > 0)

    def close(self):
        self.face_lm.close()
        self.hand_lm.close()
        self.pose_lm.close()


# ═══════════════════════════════════════════════════════════
# Drawing Utilities
# ═══════════════════════════════════════════════════════════
def draw_landmarks_on_frame(frame, landmarks):
    """Draw detected landmarks on the frame for visual feedback."""
    h, w = frame.shape[:2]

    # Draw lips (green)
    for i in range(40):
        x, y = landmarks[i, 0], landmarks[i, 1]
        if x > 0 or y > 0:
            px, py = int(x * w), int(y * h)
            cv2.circle(frame, (px, py), 1, (0, 255, 0), -1)

    # Draw left hand (blue)
    for i in range(40, 61):
        x, y = landmarks[i, 0], landmarks[i, 1]
        if x > 0 or y > 0:
            px, py = int(x * w), int(y * h)
            cv2.circle(frame, (px, py), 3, (255, 100, 0), -1)

    # Draw hand connections (left hand)
    hand_connections = [
        (0,1),(1,2),(2,3),(3,4),  # thumb
        (0,5),(5,6),(6,7),(7,8),  # index
        (0,9),(9,10),(10,11),(11,12),  # middle
        (0,13),(13,14),(14,15),(15,16),  # ring
        (0,17),(17,18),(18,19),(19,20),  # pinky
        (5,9),(9,13),(13,17),  # palm
    ]
    for a, b in hand_connections:
        x1, y1 = landmarks[40 + a, 0], landmarks[40 + a, 1]
        x2, y2 = landmarks[40 + b, 0], landmarks[40 + b, 1]
        if (x1 > 0 or y1 > 0) and (x2 > 0 or y2 > 0):
            cv2.line(frame, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), (255, 100, 0), 1)

    # Draw right hand (red)
    for i in range(61, 82):
        x, y = landmarks[i, 0], landmarks[i, 1]
        if x > 0 or y > 0:
            px, py = int(x * w), int(y * h)
            cv2.circle(frame, (px, py), 3, (0, 100, 255), -1)

    # Right hand connections
    for a, b in hand_connections:
        x1, y1 = landmarks[61 + a, 0], landmarks[61 + a, 1]
        x2, y2 = landmarks[61 + b, 0], landmarks[61 + b, 1]
        if (x1 > 0 or y1 > 0) and (x2 > 0 or y2 > 0):
            cv2.line(frame, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), (0, 100, 255), 1)

    # Draw pose (yellow)
    for i in range(82, 92):
        x, y = landmarks[i, 0], landmarks[i, 1]
        if x > 0 or y > 0:
            px, py = int(x * w), int(y * h)
            cv2.circle(frame, (px, py), 4, (0, 255, 255), -1)

    return frame


def draw_ui(frame, prediction, confidence, top5, recording, frame_count, fps):
    """Draw the UI overlay on the frame."""
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Title
    cv2.putText(frame, "ASL Sign Recognition", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 120, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Recording status
    if recording:
        # Blinking red dot
        if int(time.time() * 3) % 2 == 0:
            cv2.circle(frame, (w - 30, 55), 8, (0, 0, 255), -1)
        cv2.putText(frame, f"Recording: {frame_count}/{MAX_FRAMES}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        # Progress bar
        progress = min(frame_count / MAX_FRAMES, 1.0)
        bar_w = w - 20
        cv2.rectangle(frame, (10, 70), (10 + bar_w, 85), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, 70), (10 + int(bar_w * progress), 85), (0, 200, 255), -1)
    else:
        cv2.putText(frame, "Press 'R' to record a sign", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

    # Prediction result
    if prediction:
        # Bottom bar
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, h - 120), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)

        # Main prediction
        color = (0, 255, 0) if confidence > 0.5 else (0, 200, 255) if confidence > 0.3 else (0, 100, 255)
        cv2.putText(frame, f"Prediction: {prediction}", (10, h - 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, f"Confidence: {confidence:.1%}", (10, h - 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        # Top-5 predictions
        if top5:
            for i, (sign, conf) in enumerate(top5):
                y_pos = h - 45 + i * 18
                bar_len = int(conf * 150)
                cv2.rectangle(frame, (10, y_pos - 10), (10 + bar_len, y_pos + 4), (80, 80, 80), -1)
                cv2.putText(frame, f"{sign}: {conf:.1%}", (15, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # Controls hint
    cv2.putText(frame, "R: Record | C: Clear | Q: Quit", (10, h - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

    return frame


# ═══════════════════════════════════════════════════════════
# Inference
# ═══════════════════════════════════════════════════════════
def predict_sign(model, frame_buffer, device, class_names):
    """Run model inference on a buffered sequence of landmarks."""
    num_frames = len(frame_buffer)
    if num_frames < 3:
        return None, 0.0, []

    # Stack frames into [T, 92, 3]
    landmarks = np.stack(list(frame_buffer), axis=0)

    # Pad to MAX_FRAMES
    if num_frames < MAX_FRAMES:
        pad = np.zeros((MAX_FRAMES - num_frames, NUM_LANDMARKS, NUM_COORDS), dtype=np.float32)
        landmarks = np.concatenate([landmarks, pad], axis=0)
    else:
        landmarks = landmarks[:MAX_FRAMES]

    # Compute non-empty frame indices
    non_empty = np.any(landmarks != 0, axis=(1, 2))
    non_empty_idxs = np.where(non_empty, np.arange(MAX_FRAMES, dtype=np.float32), -1.0)

    # To tensor [1, T, 92, 3]
    frames_t = torch.from_numpy(landmarks).unsqueeze(0).to(device)
    idxs_t = torch.from_numpy(non_empty_idxs).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        logits = model(frames_t, idxs_t)
        probs = torch.softmax(logits, dim=1)[0]

    # Top-5
    top5_probs, top5_indices = probs.topk(5)
    top5 = [(class_names[idx.item()], prob.item()) for idx, prob in zip(top5_indices, top5_probs)]

    best_sign = top5[0][0]
    best_conf = top5[0][1]

    return best_sign, best_conf, top5


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Real-time ASL Sign Recognition")
    parser.add_argument("--model", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "outputs", "landmark_transformer_3d", "best_model.pth"),
                        help="Path to trained model checkpoint")
    parser.add_argument("--mediapipe-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "mediapipe_models"),
                        help="Path to MediaPipe .task model files")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--threshold", type=float, default=0.15,
                        help="Minimum confidence to show prediction")
    parser.add_argument("--auto-record", action="store_true",
                        help="Automatically record when hands are detected")
    args = parser.parse_args()

    # ---- Load model ----
    print("Loading model...")
    if not os.path.exists(args.model):
        print(f"ERROR: Model not found at {args.model}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint = torch.load(args.model, map_location=device, weights_only=False)

    # Get config from checkpoint
    config = checkpoint.get("config", {})
    num_classes = checkpoint.get("num_classes", 256)
    class_names_list = checkpoint.get("class_names", [])

    # Also load class_map.json for idx->name mapping
    class_map_path = os.path.join(os.path.dirname(args.model), "class_map.json")
    if os.path.exists(class_map_path):
        with open(class_map_path) as f:
            class_map = json.load(f)
        class_names = {int(k): v for k, v in class_map.items()}
    elif class_names_list:
        class_names = {i: name for i, name in enumerate(class_names_list)}
    else:
        class_names = {i: f"class_{i}" for i in range(num_classes)}

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

    best_acc = checkpoint.get("best_val_acc", 0)
    print(f"Model loaded: {num_classes} classes, val accuracy: {best_acc*100:.2f}%")

    # ---- Load MediaPipe ----
    print("Loading MediaPipe landmarks...")
    if not os.path.exists(args.mediapipe_dir):
        print(f"ERROR: MediaPipe models not found at {args.mediapipe_dir}")
        print("Download from: https://developers.google.com/mediapipe/solutions/vision/")
        sys.exit(1)

    extractor = LandmarkExtractor(args.mediapipe_dir)
    print("MediaPipe loaded.")

    # ---- Open webcam ----
    print(f"Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        extractor.close()
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n" + "=" * 50)
    print("  ASL Sign Language Recognition - Ready!")
    print("  Press 'R' to start/stop recording a sign")
    print("  Press 'C' to clear prediction")
    print("  Press 'Q' to quit")
    if args.auto_record:
        print("  Auto-record mode: ON (records when hands visible)")
    print("=" * 50 + "\n")

    # State
    recording = False
    frame_buffer = collections.deque(maxlen=MAX_FRAMES)
    prediction = None
    confidence = 0.0
    top5 = []
    fps = 0.0
    fps_timer = time.time()
    fps_counter = 0
    auto_record_cooldown = 0

    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        # Mirror for natural interaction
        bgr = cv2.flip(bgr, 1)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # FPS calculation
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            fps = fps_counter / (time.time() - fps_timer)
            fps_counter = 0
            fps_timer = time.time()

        # Extract landmarks
        landmarks = extractor.extract(rgb)
        has_any_landmark = np.any(landmarks != 0)

        # Auto-record: start when hands appear, stop when they disappear
        if args.auto_record:
            hands_visible = np.any(landmarks[40:82] != 0)
            if hands_visible and not recording and auto_record_cooldown <= 0:
                recording = True
                frame_buffer.clear()
                prediction = None
            elif not hands_visible and recording and len(frame_buffer) >= 8:
                # Hands disappeared - predict
                recording = False
                prediction, confidence, top5 = predict_sign(
                    model, frame_buffer, device, class_names
                )
                if confidence < args.threshold:
                    prediction = None
                auto_record_cooldown = 15  # skip frames before next auto-record
            elif not hands_visible:
                auto_record_cooldown = max(0, auto_record_cooldown - 1)

        # Buffer frames during recording
        if recording and has_any_landmark:
            frame_buffer.append(landmarks)

            # Auto-predict when buffer is full
            if len(frame_buffer) >= MAX_FRAMES and not args.auto_record:
                recording = False
                prediction, confidence, top5 = predict_sign(
                    model, frame_buffer, device, class_names
                )
                if confidence < args.threshold:
                    prediction = None
                print(f"Prediction: {prediction} ({confidence:.1%})")

        # Draw
        if has_any_landmark:
            bgr = draw_landmarks_on_frame(bgr, landmarks)
        bgr = draw_ui(bgr, prediction, confidence, top5, recording,
                       len(frame_buffer), fps)

        cv2.imshow("ASL Sign Recognition", bgr)

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('r') or key == ord('R'):
            if recording:
                # Stop recording and predict
                recording = False
                if len(frame_buffer) >= 3:
                    prediction, confidence, top5 = predict_sign(
                        model, frame_buffer, device, class_names
                    )
                    if confidence < args.threshold:
                        prediction = f"{prediction} (low confidence)"
                    print(f"Prediction: {prediction} ({confidence:.1%})")
                    if top5:
                        for sign, conf in top5:
                            print(f"  {sign}: {conf:.1%}")
                else:
                    print("Not enough frames recorded (need at least 3)")
            else:
                # Start recording
                recording = True
                frame_buffer.clear()
                prediction = None
                confidence = 0.0
                top5 = []
                print("Recording started... perform the sign")
        elif key == ord('c') or key == ord('C'):
            prediction = None
            confidence = 0.0
            top5 = []
            frame_buffer.clear()

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()
    print("Done.")


if __name__ == "__main__":
    main()
