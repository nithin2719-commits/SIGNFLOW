# Extract MediaPipe 3D landmarks from ALL available ASL datasets:
#   1. MS-ASL      - MS-ASL/dataset/{train|val}/{word}/*.mp4
#   2. WLASL       - New folder/videos/{video_id}.mp4
#   3. ASL Citizen - Downloads/ASL_Citizen/ASL_Citizen/videos/*.mp4
#
# Output: landmark_data_combined/{train|val}/{word}/*.npy  (shape 64 x 92 x 3)
# Run: python extract_msasl_video_landmarks.py

import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

# -- landmark indices (must match train script) --
LIPS_FACE_IDXS = np.array([
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
], dtype=np.int32)
POSE_UPPER_IDXS = np.array([0, 11, 12, 13, 14, 15, 16, 23, 24, 25], dtype=np.int32)
MAX_FRAMES    = 64
NUM_LANDMARKS = 92   # 40 lips + 21 LH + 21 RH + 10 pose

# -- dataset paths --
MSASL_DIR      = Path(r"c:\Users\Asus\project\New Msasl\MS-ASL\dataset")
WLASL_VIDEOS   = Path(r"c:\Users\Asus\project\New folder\videos")
WLASL_JSON     = Path(r"c:\Users\Asus\project\New folder\WLASL_v0.3.json")
EXISTING_LM    = Path(r"c:\Users\Asus\project\New Msasl\landmark_data")
OUT_DIR        = Path(r"c:\Users\Asus\project\New Msasl\landmark_data_combined")
ASL_CITIZEN_DIR= Path(r"c:\Users\Asus\Downloads\ASL_Citizen\ASL_Citizen")
VAL_SPLIT      = 0.15

# -- mediapipe Tasks API (0.10.30+) --
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

MODELS_DIR = Path(r"c:\Users\Asus\project\New Msasl\mediapipe_models")
HAND_MODEL = str(MODELS_DIR / "hand_landmarker.task")
FACE_MODEL = str(MODELS_DIR / "face_landmarker.task")
POSE_MODEL = str(MODELS_DIR / "pose_landmarker_heavy.task")
print(f"[INFO] mediapipe {mp.__version__} loaded OK (Tasks API)")


def _create_detectors():
    hand_det = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=HAND_MODEL),
            num_hands=2,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3))
    face_det = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=FACE_MODEL),
            num_faces=1,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3))
    pose_det = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=POSE_MODEL),
            num_poses=1,
            min_pose_detection_confidence=0.3,
            min_pose_presence_confidence=0.3))
    return hand_det, face_det, pose_det


# -- core extraction --
def extract_landmarks(video_path, frame_start=0, frame_end=-1):
    """Read video, extract (MAX_FRAMES, 92, 3) MediaPipe landmarks."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_end < 0 or frame_end >= total:
        frame_end = total - 1
    if frame_start > frame_end:
        frame_start = 0

    n_avail = frame_end - frame_start + 1
    if n_avail < 3:
        cap.release()
        return None

    read_idxs = np.linspace(frame_start, frame_end,
                             min(MAX_FRAMES, n_avail), dtype=int)

    frames_rgb = []
    for idx in read_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if len(frames_rgb) < 3:
        return None

    result = np.zeros((MAX_FRAMES, NUM_LANDMARKS, 3), dtype=np.float32)

    hand_det, face_det, pose_det = _create_detectors()

    for fi, rgb in enumerate(frames_rgb):
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # face -> lips
        fr = face_det.detect(mp_img)
        if fr.face_landmarks:
            lms = fr.face_landmarks[0]
            for slot, orig in enumerate(LIPS_FACE_IDXS):
                if orig < len(lms):
                    lm = lms[orig]
                    result[fi, slot] = [lm.x, lm.y, lm.z]
        # hands
        hr = hand_det.detect(mp_img)
        if hr.hand_landmarks and hr.handedness:
            for hinfo, hlms in zip(hr.handedness, hr.hand_landmarks):
                label = hinfo[0].category_name
                off = 40 if label == "Left" else 61
                for li, lm in enumerate(hlms):
                    result[fi, off + li] = [lm.x, lm.y, lm.z]
        # pose
        pr = pose_det.detect(mp_img)
        if pr.pose_landmarks:
            plms = pr.pose_landmarks[0]
            for slot, orig in enumerate(POSE_UPPER_IDXS):
                if orig < len(plms):
                    lm = plms[orig]
                    result[fi, 82 + slot] = [lm.x, lm.y, lm.z]

    hand_det.close()
    face_det.close()
    pose_det.close()

    # Reject if no hand detected at all
    if not np.any(result[:, 40:82] != 0):
        return None
    return result


def save(arr, out_dir, stem):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(out_dir / (stem + ".npy")), arr)


# -- Step 1: Copy existing landmark_data as base --
def copy_existing():
    print("[1/4] Copying existing landmark_data (already extracted)...")
    total = 0
    for split in ["train", "val"]:
        src = EXISTING_LM / split
        if not src.exists():
            continue
        for cls_dir in src.iterdir():
            if not cls_dir.is_dir():
                continue
            dst = OUT_DIR / split / cls_dir.name
            dst.mkdir(parents=True, exist_ok=True)
            for f in cls_dir.glob("*.npy"):
                dst_f = dst / f.name
                if not dst_f.exists():
                    import shutil
                    shutil.copy2(f, dst_f)
                    total += 1
    print(f"    Copied {total} existing .npy files\n")


# -- Step 2: Extract from MS-ASL videos --
def extract_msasl():
    print("[2/4] Extracting from MS-ASL videos...")
    all_words = set()
    for split in ["train", "val"]:
        d = MSASL_DIR / split
        if d.exists():
            for fd in d.iterdir():
                if fd.is_dir() and any(fd.glob("*.mp4")):
                    all_words.add(fd.name)

    all_words = sorted(all_words)
    print(f"    Found {len(all_words)} words with MS-ASL clips")
    done = nt = nv = 0
    for i, word in enumerate(all_words):
        clips = []
        for split in ["train", "val"]:
            d = MSASL_DIR / split / word
            if d.exists():
                clips.extend(d.glob("*.mp4"))
        if not clips:
            continue

        random.shuffle(clips)
        n_val = max(1, int(len(clips) * VAL_SPLIT))
        split_map = [("val", clips[:n_val]), ("train", clips[n_val:])]
        added = 0
        for out_split, clip_list in split_map:
            out_dir = OUT_DIR / out_split / word
            for clip in clip_list:
                out_f = out_dir / ("msasl_" + clip.stem + ".npy")
                if out_f.exists():
                    continue
                arr = extract_landmarks(clip)
                if arr is not None:
                    save(arr, out_dir, "msasl_" + clip.stem)
                    added += 1
                    if out_split == "train":
                        nt += 1
                    else:
                        nv += 1
        done += 1
        if (i + 1) % 20 == 0:
            print(f"    MS-ASL progress: {i+1}/{len(all_words)} words, "
                  f"{nt} train / {nv} val samples so far")
    print(f"    MS-ASL done: {done} words, {nt} train + {nv} val samples\n")


# -- Step 3: Extract from WLASL videos --
def extract_wlasl():
    print("[3/4] Extracting from WLASL videos...")
    if not WLASL_JSON.exists():
        print(f"    [SKIP] {WLASL_JSON} not found")
        return
    if not WLASL_VIDEOS.exists():
        print(f"    [SKIP] {WLASL_VIDEOS} not found")
        return

    with open(WLASL_JSON, "r") as f:
        data = json.load(f)

    nt = nv = skipped = 0
    for entry in data:
        word = entry["gloss"].lower().strip()
        instances = entry.get("instances", [])
        random.shuffle(instances)

        for inst in instances:
            vid_id    = str(inst.get("video_id", ""))
            split     = inst.get("split", "train")
            if split == "test":
                split = "val"
            f_start   = max(0, inst.get("frame_start", 0) - 1)
            f_end     = inst.get("frame_end", -1)

            vid_path = WLASL_VIDEOS / f"{vid_id}.mp4"
            if not vid_path.exists():
                skipped += 1
                continue

            out_dir = OUT_DIR / split / word
            out_f   = out_dir / f"wlasl_{vid_id}.npy"
            if out_f.exists():
                continue

            arr = extract_landmarks(vid_path, f_start, f_end)
            if arr is not None:
                save(arr, out_dir, f"wlasl_{vid_id}")
                if split == "train":
                    nt += 1
                else:
                    nv += 1

    print(f"    WLASL done: {nt} train + {nv} val samples "
          f"({skipped} videos not found)\n")


# -- Step 4: Extract from ASL Citizen videos --
def extract_asl_citizen():
    import csv
    print("[4/4] Extracting from ASL Citizen videos...")
    videos_dir = ASL_CITIZEN_DIR / "videos"
    splits_dir = ASL_CITIZEN_DIR / "splits"
    if not videos_dir.exists():
        print(f"    [SKIP] {videos_dir} not found")
        return
    if not splits_dir.exists():
        print(f"    [SKIP] {splits_dir} not found")
        return

    split_remap = {"train": "train", "val": "val", "test": "val"}

    nt = nv = skipped = 0
    for csv_name in ["train.csv", "val.csv", "test.csv"]:
        csv_path = splits_dir / csv_name
        if not csv_path.exists():
            continue
        out_split = split_remap[csv_name.replace(".csv", "")]
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            for row in reader:
                if len(row) < 3:
                    continue
                video_file = row[1].strip()
                gloss      = row[2].strip().lower()
                if not gloss or not video_file:
                    continue

                vid_path = videos_dir / video_file
                if not vid_path.exists():
                    skipped += 1
                    continue

                out_dir = OUT_DIR / out_split / gloss
                stem    = "aslcitizen_" + Path(video_file).stem
                out_f   = out_dir / (stem + ".npy")
                if out_f.exists():
                    continue

                arr = extract_landmarks(vid_path)
                if arr is not None:
                    save(arr, out_dir, stem)
                    if out_split == "train":
                        nt += 1
                    else:
                        nv += 1

    print(f"    ASL Citizen done: {nt} train + {nv} val samples "
          f"({skipped} videos not found)\n")


# -- Main --
def main():
    random.seed(42)
    print("=" * 60)
    print("ASL Landmark Extraction - MS-ASL + WLASL + ASL Citizen + Existing")
    print("=" * 60 + "\n")

    copy_existing()
    extract_msasl()
    # Skipping WLASL + ASL Citizen for faster extraction
    # extract_wlasl()
    # extract_asl_citizen()

    # Final stats
    all_classes = sorted([d.name for d in (OUT_DIR / "train").iterdir()
                          if d.is_dir()]) if (OUT_DIR / "train").exists() else []
    total_train = sum(len(list((OUT_DIR / "train" / c).glob("*.npy")))
                      for c in all_classes)
    total_val   = sum(len(list((OUT_DIR / "val"   / c).glob("*.npy")))
                      for c in all_classes if (OUT_DIR / "val" / c).exists())

    print("=" * 60)
    print(f"EXTRACTION COMPLETE")
    print(f"  Total classes : {len(all_classes)}")
    print(f"  Train samples : {total_train}")
    print(f"  Val   samples : {total_val}")
    print(f"  Output dir    : {OUT_DIR}")
    print("=" * 60)

    # Auto-start training
    print("\n[AUTO] Starting training automatically...\n")
    import subprocess
    result = subprocess.run(
        [sys.executable, str(Path(r"c:\Users\Asus\project\New Msasl\train_common_words.py"))],
        cwd=str(Path(r"c:\Users\Asus\project\New Msasl")))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()