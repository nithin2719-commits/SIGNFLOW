"""
FAST Kaggle parquet -> npy converter using vectorized numpy.
Then starts MS-ASL + WLASL video landmark extraction.
"""
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

LIPS_FACE_IDXS = np.array([
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
], dtype=np.int32)
POSE_UPPER_IDXS = np.array([0, 11, 12, 13, 14, 15, 16, 23, 24, 25], dtype=np.int32)

NUM_LANDMARKS = 92
NUM_COORDS = 3
MAX_FRAMES = 64

# Pre-build lip index lookup for fast vectorized access
LIPS_LOOKUP = {int(v): i for i, v in enumerate(LIPS_FACE_IDXS)}
POSE_LOOKUP = {int(v): i for i, v in enumerate(POSE_UPPER_IDXS)}


def fast_parquet_to_npy(parquet_path):
    """Vectorized parquet to [MAX_FRAMES, 92, 3] conversion."""
    df = pq.read_table(parquet_path).to_pandas()

    frames = df["frame"].unique()
    frames.sort()
    num_frames = len(frames)
    if num_frames < 3:
        return None

    # Sample frames if too many
    if num_frames > MAX_FRAMES:
        idx = np.linspace(0, num_frames - 1, MAX_FRAMES, dtype=int)
        frames = frames[idx]
        num_frames = MAX_FRAMES

    # Build frame->index map
    frame_map = {f: i for i, f in enumerate(frames)}

    result = np.zeros((MAX_FRAMES, NUM_LANDMARKS, NUM_COORDS), dtype=np.float32)

    # Fill NaN with 0
    for col in ["x", "y", "z"]:
        df[col] = df[col].fillna(0.0)

    # Filter to only selected frames
    df = df[df["frame"].isin(set(frames))]

    # Process each type in bulk
    # FACE -> lips
    face_df = df[df["type"] == "face"]
    if len(face_df) > 0:
        for lip_orig_idx, lip_slot in LIPS_LOOKUP.items():
            rows = face_df[face_df["landmark_index"] == lip_orig_idx]
            if len(rows) > 0:
                for _, r in rows.iterrows():
                    fi = frame_map.get(r["frame"])
                    if fi is not None:
                        result[fi, lip_slot, 0] = r["x"]
                        result[fi, lip_slot, 1] = r["y"]
                        result[fi, lip_slot, 2] = r["z"]

    # LEFT HAND
    lh_df = df[df["type"] == "left_hand"]
    if len(lh_df) > 0:
        for _, r in lh_df.iterrows():
            fi = frame_map.get(r["frame"])
            li = int(r["landmark_index"])
            if fi is not None and 0 <= li < 21:
                result[fi, 40 + li, 0] = r["x"]
                result[fi, 40 + li, 1] = r["y"]
                result[fi, 40 + li, 2] = r["z"]

    # RIGHT HAND
    rh_df = df[df["type"] == "right_hand"]
    if len(rh_df) > 0:
        for _, r in rh_df.iterrows():
            fi = frame_map.get(r["frame"])
            li = int(r["landmark_index"])
            if fi is not None and 0 <= li < 21:
                result[fi, 61 + li, 0] = r["x"]
                result[fi, 61 + li, 1] = r["y"]
                result[fi, 61 + li, 2] = r["z"]

    # POSE
    pose_df = df[df["type"] == "pose"]
    if len(pose_df) > 0:
        for pose_orig_idx, pose_slot in POSE_LOOKUP.items():
            rows = pose_df[pose_df["landmark_index"] == pose_orig_idx]
            if len(rows) > 0:
                for _, r in rows.iterrows():
                    fi = frame_map.get(r["frame"])
                    if fi is not None:
                        result[fi, 82 + pose_slot, 0] = r["x"]
                        result[fi, 82 + pose_slot, 1] = r["y"]
                        result[fi, 82 + pose_slot, 2] = r["z"]

    return result


def worker_convert(args):
    parquet_path, out_path = args
    try:
        result = fast_parquet_to_npy(parquet_path)
        if result is not None:
            np.save(out_path, result)
            return True
    except Exception:
        pass
    return False


def convert_kaggle(kaggle_dir, output_dir):
    csv_path = os.path.join(kaggle_dir, "train.csv")
    df = pd.read_csv(csv_path)

    participants = df["participant_id"].unique()
    np.random.seed(42)
    np.random.shuffle(participants)
    val_participants = set(participants[:max(1, len(participants) // 5)])

    tasks = []
    skipped = 0
    for _, row in df.iterrows():
        sign = row["sign"]
        parquet_path = os.path.join(kaggle_dir, row["path"])
        split = "val" if row["participant_id"] in val_participants else "train"
        out_cls_dir = os.path.join(output_dir, split, sign)
        os.makedirs(out_cls_dir, exist_ok=True)
        out_path = os.path.join(out_cls_dir, f"kaggle_{row['sequence_id']}.npy")
        if os.path.exists(out_path):
            skipped += 1
            continue
        tasks.append((parquet_path, out_path))

    print(f"  {len(df)} total, {skipped} already done, {len(tasks)} to convert")
    if not tasks:
        return

    ok = 0
    fail = 0
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(worker_convert, t): t for t in tasks}
        for i, future in enumerate(as_completed(futures)):
            if future.result():
                ok += 1
            else:
                fail += 1
            if (i + 1) % 5000 == 0 or i == len(tasks) - 1:
                print(f"  [{i+1}/{len(tasks)}] OK={ok} FAIL={fail}", flush=True)

    print(f"  Done: {ok} converted, {fail} failed")


def convert_msasl(msasl_dir, output_dir):
    """Convert MS-ASL videos - uses MediaPipe Tasks API."""
    import mediapipe as mp_lib
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        FaceLandmarker, FaceLandmarkerOptions,
        HandLandmarker, HandLandmarkerOptions,
        PoseLandmarker, PoseLandmarkerOptions,
        RunningMode,
    )

    MODEL_DIR = "c:/Users/Asus/project/New Msasl/mediapipe_models"
    face_lm = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(MODEL_DIR, "face_landmarker.task")),
        running_mode=RunningMode.IMAGE, num_faces=1,
        min_face_detection_confidence=0.3, min_face_presence_confidence=0.3,
    ))
    hand_lm = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(MODEL_DIR, "hand_landmarker.task")),
        running_mode=RunningMode.IMAGE, num_hands=2,
        min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3,
    ))
    pose_lm = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(MODEL_DIR, "pose_landmarker_heavy.task")),
        running_mode=RunningMode.IMAGE, num_poses=1,
        min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3,
    ))

    total_ok = 0
    total_fail = 0
    global_count = 0

    for split in ["train", "val"]:
        split_dir = os.path.join(msasl_dir, "dataset", split)
        if not os.path.exists(split_dir):
            continue
        classes = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
        for ci, cls_name in enumerate(classes):
            cls_dir = os.path.join(split_dir, cls_name)
            out_cls_dir = os.path.join(output_dir, split, cls_name)
            os.makedirs(out_cls_dir, exist_ok=True)
            videos = [f for f in os.listdir(cls_dir) if f.endswith((".mp4", ".avi", ".mov"))]
            for vf in videos:
                out_path = os.path.join(out_cls_dir, vf.rsplit(".", 1)[0] + ".npy")
                if os.path.exists(out_path):
                    total_ok += 1
                    continue
                try:
                    vid_path = os.path.join(cls_dir, vf)
                    cap = cv2.VideoCapture(vid_path)
                    if not cap.isOpened():
                        total_fail += 1
                        continue
                    total_frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if total_frames_count <= 0:
                        total_frames_count = 300
                    if total_frames_count > MAX_FRAMES:
                        sample_set = set(np.linspace(0, total_frames_count - 1, MAX_FRAMES, dtype=int).tolist())
                    else:
                        sample_set = set(range(total_frames_count))

                    frames_data = []
                    fidx = 0
                    while True:
                        ret, bgr = cap.read()
                        if not ret:
                            break
                        if fidx in sample_set:
                            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                            mp_img = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=rgb)
                            lm = np.zeros((NUM_LANDMARKS, NUM_COORDS), dtype=np.float32)

                            fr = face_lm.detect(mp_img)
                            if fr.face_landmarks and len(fr.face_landmarks) > 0:
                                face = fr.face_landmarks[0]
                                for i, fi in enumerate(LIPS_FACE_IDXS):
                                    if fi < len(face):
                                        lm[i] = [face[fi].x, face[fi].y, face[fi].z]

                            hr = hand_lm.detect(mp_img)
                            if hr.hand_landmarks:
                                for hi, (hm, hn) in enumerate(zip(hr.hand_landmarks, hr.handedness)):
                                    if hi >= 2:
                                        break
                                    label = hn[0].category_name.lower() if hn else "left"
                                    offset = 40 if label == "left" else 61
                                    for j in range(min(21, len(hm))):
                                        lm[offset + j] = [hm[j].x, hm[j].y, hm[j].z]

                            pr = pose_lm.detect(mp_img)
                            if pr.pose_landmarks and len(pr.pose_landmarks) > 0:
                                pose = pr.pose_landmarks[0]
                                for k, pidx in enumerate(POSE_UPPER_IDXS):
                                    if pidx < len(pose):
                                        lm[82 + k] = [pose[pidx].x, pose[pidx].y, pose[pidx].z]

                            frames_data.append(lm)
                        fidx += 1
                        if len(frames_data) >= MAX_FRAMES:
                            break
                    cap.release()

                    if len(frames_data) >= 3:
                        arr = np.stack(frames_data, axis=0).astype(np.float32)
                        if arr.shape[0] < MAX_FRAMES:
                            pad = np.zeros((MAX_FRAMES - arr.shape[0], NUM_LANDMARKS, NUM_COORDS), dtype=np.float32)
                            arr = np.concatenate([arr, pad], axis=0)
                        np.save(out_path, arr[:MAX_FRAMES])
                        total_ok += 1
                    else:
                        total_fail += 1
                except Exception:
                    total_fail += 1

                global_count += 1
                if global_count % 100 == 0:
                    print(f"  [MS-ASL {split}] {global_count} videos | OK={total_ok} FAIL={total_fail}", flush=True)

    face_lm.close()
    hand_lm.close()
    pose_lm.close()
    print(f"  MS-ASL done: {total_ok} OK, {total_fail} failed")


def convert_wlasl(wlasl_dir, output_dir):
    """Convert WLASL videos."""
    import mediapipe as mp_lib
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        FaceLandmarker, FaceLandmarkerOptions,
        HandLandmarker, HandLandmarkerOptions,
        PoseLandmarker, PoseLandmarkerOptions,
        RunningMode,
    )

    nslt_json = os.path.join(wlasl_dir, "nslt_1000.json")
    class_file = os.path.join(wlasl_dir, "wlasl_class_list.txt")

    with open(nslt_json) as f:
        nslt = json.load(f)
    with open(class_file) as f:
        lines = [l.strip() for l in f if l.strip()]
    idx_to_label = {}
    for line in lines:
        parts = line.split("\t", 1)
        if len(parts) == 2:
            idx_to_label[int(parts[0])] = parts[1]

    video_dir = os.path.join(wlasl_dir, "videos")

    MODEL_DIR = "c:/Users/Asus/project/New Msasl/mediapipe_models"
    face_lm = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(MODEL_DIR, "face_landmarker.task")),
        running_mode=RunningMode.IMAGE, num_faces=1,
        min_face_detection_confidence=0.3, min_face_presence_confidence=0.3,
    ))
    hand_lm = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(MODEL_DIR, "hand_landmarker.task")),
        running_mode=RunningMode.IMAGE, num_hands=2,
        min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3,
    ))
    pose_lm = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(MODEL_DIR, "pose_landmarker_heavy.task")),
        running_mode=RunningMode.IMAGE, num_poses=1,
        min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3,
    ))

    total_ok = 0
    total_fail = 0
    count = 0

    for vid_id, info in nslt.items():
        label_idx = info["action"][0]
        if label_idx >= 1000:
            continue
        label_name = idx_to_label.get(label_idx, f"class_{label_idx}")
        video_path = os.path.join(video_dir, f"{vid_id}.mp4")
        if not os.path.exists(video_path):
            continue
        out_split = "train" if info["subset"] == "train" else "val"
        out_cls_dir = os.path.join(output_dir, out_split, label_name)
        os.makedirs(out_cls_dir, exist_ok=True)
        out_path = os.path.join(out_cls_dir, f"wlasl_{vid_id}.npy")
        if os.path.exists(out_path):
            total_ok += 1
            continue

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                total_fail += 1
                continue
            tfc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if tfc <= 0:
                tfc = 300
            if tfc > MAX_FRAMES:
                sample_set = set(np.linspace(0, tfc - 1, MAX_FRAMES, dtype=int).tolist())
            else:
                sample_set = set(range(tfc))

            frames_data = []
            fidx = 0
            while True:
                ret, bgr = cap.read()
                if not ret:
                    break
                if fidx in sample_set:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    mp_img = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=rgb)
                    lm = np.zeros((NUM_LANDMARKS, NUM_COORDS), dtype=np.float32)

                    fr = face_lm.detect(mp_img)
                    if fr.face_landmarks and len(fr.face_landmarks) > 0:
                        face = fr.face_landmarks[0]
                        for i, fi2 in enumerate(LIPS_FACE_IDXS):
                            if fi2 < len(face):
                                lm[i] = [face[fi2].x, face[fi2].y, face[fi2].z]

                    hr = hand_lm.detect(mp_img)
                    if hr.hand_landmarks:
                        for hi, (hm, hn) in enumerate(zip(hr.hand_landmarks, hr.handedness)):
                            if hi >= 2:
                                break
                            label = hn[0].category_name.lower() if hn else "left"
                            offset = 40 if label == "left" else 61
                            for j in range(min(21, len(hm))):
                                lm[offset + j] = [hm[j].x, hm[j].y, hm[j].z]

                    pr = pose_lm.detect(mp_img)
                    if pr.pose_landmarks and len(pr.pose_landmarks) > 0:
                        pose = pr.pose_landmarks[0]
                        for k, pidx in enumerate(POSE_UPPER_IDXS):
                            if pidx < len(pose):
                                lm[82 + k] = [pose[pidx].x, pose[pidx].y, pose[pidx].z]

                    frames_data.append(lm)
                fidx += 1
                if len(frames_data) >= MAX_FRAMES:
                    break
            cap.release()

            if len(frames_data) >= 3:
                arr = np.stack(frames_data, axis=0).astype(np.float32)
                if arr.shape[0] < MAX_FRAMES:
                    pad = np.zeros((MAX_FRAMES - arr.shape[0], NUM_LANDMARKS, NUM_COORDS), dtype=np.float32)
                    arr = np.concatenate([arr, pad], axis=0)
                np.save(out_path, arr[:MAX_FRAMES])
                total_ok += 1
            else:
                total_fail += 1
        except Exception:
            total_fail += 1

        count += 1
        if count % 100 == 0:
            print(f"  [WLASL] {count} videos | OK={total_ok} FAIL={total_fail}", flush=True)

    face_lm.close()
    hand_lm.close()
    pose_lm.close()
    print(f"  WLASL done: {total_ok} OK, {total_fail} failed")


def main():
    output_dir = "c:/Users/Asus/project/New Msasl/landmark_data"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60, flush=True)
    print("FAST LANDMARK CONVERSION PIPELINE", flush=True)
    print("=" * 60, flush=True)

    # Step 1: Kaggle (fast parquet conversion)
    kaggle_dir = "c:/Users/Asus/project/new asl dataset from kaggle"
    print("\n[1/3] Converting Kaggle parquets...", flush=True)
    convert_kaggle(kaggle_dir, output_dir)

    # Step 2: MS-ASL videos
    msasl_dir = "c:/Users/Asus/project/New Msasl/MS-ASL"
    if os.path.exists(os.path.join(msasl_dir, "dataset")):
        print("\n[2/3] Extracting MS-ASL video landmarks...", flush=True)
        convert_msasl(msasl_dir, output_dir)

    # Step 3: WLASL videos
    wlasl_dir = "c:/Users/Asus/project/New folder"
    if os.path.exists(os.path.join(wlasl_dir, "nslt_1000.json")):
        print("\n[3/3] Extracting WLASL video landmarks...", flush=True)
        convert_wlasl(wlasl_dir, output_dir)

    # Summary
    print("\n" + "=" * 60, flush=True)
    print("ALL DONE", flush=True)
    for split in ["train", "val"]:
        sp = os.path.join(output_dir, split)
        if not os.path.exists(sp):
            continue
        classes = [d for d in os.listdir(sp) if os.path.isdir(os.path.join(sp, d))]
        total = sum(len([f for f in os.listdir(os.path.join(sp, c)) if f.endswith(".npy")]) for c in classes)
        print(f"  {split}: {total} samples across {len(classes)} classes", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
