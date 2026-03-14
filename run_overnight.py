"""
OVERNIGHT AUTOMATION SCRIPT
Runs everything unattended:
1. Finish Kaggle parquet -> npy conversion
2. Extract MS-ASL video landmarks
3. Extract WLASL video landmarks
4. Train Landmark Transformer to >75% val accuracy
"""
import subprocess
import sys
import os
import time

PYTHON = r"c:\Users\Asus\project\New Msasl\.venv\Scripts\python.exe"
PROJECT = r"c:\Users\Asus\project\New Msasl"
LOG_FILE = os.path.join(PROJECT, "overnight_log.txt")


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_step(description, script_path, args=None):
    log(f"=== STARTING: {description} ===")
    cmd = [PYTHON, "-u", script_path]
    if args:
        cmd.extend(args)
    result = subprocess.run(cmd, capture_output=False, text=True, cwd=PROJECT)
    if result.returncode != 0:
        log(f"WARNING: {description} exited with code {result.returncode}")
    else:
        log(f"=== FINISHED: {description} ===")
    return result.returncode


def count_data():
    data_dir = os.path.join(PROJECT, "landmark_data")
    stats = {}
    for split in ["train", "val"]:
        sp = os.path.join(data_dir, split)
        if not os.path.exists(sp):
            stats[split] = (0, 0)
            continue
        classes = [d for d in os.listdir(sp) if os.path.isdir(os.path.join(sp, d))]
        total = sum(
            len([f for f in os.listdir(os.path.join(sp, c)) if f.endswith(".npy")])
            for c in classes
        )
        stats[split] = (total, len(classes))
    return stats


def main():
    log("=" * 60)
    log("OVERNIGHT AUTOMATION STARTED")
    log("=" * 60)

    # Step 1: Data extraction
    log("\n[STEP 1] Landmark extraction...")
    extract_script = os.path.join(PROJECT, "extract_landmarks.py")
    run_step("Landmark Extraction", extract_script)

    stats = count_data()
    for split, (total, classes) in stats.items():
        log(f"  {split}: {total} samples, {classes} classes")

    # Step 2: Train
    log("\n[STEP 2] Training Landmark Transformer...")
    train_script = os.path.join(PROJECT, "train_landmark_transformer.py")
    train_args = [
        "--epochs", "150",
        "--batch-size", "64",
        "--lr", "3e-4",
        "--num-workers", "4",
        "--patience", "30",
        "--warmup-epochs", "5",
        "--dropout", "0.2",
        "--label-smoothing", "0.1",
        "--mixup-alpha", "0.3",
        "--output-dir", os.path.join(PROJECT, "outputs", "landmark_transformer_1000"),
    ]
    run_step("Training", train_script, train_args)

    log("\n" + "=" * 60)
    log("OVERNIGHT AUTOMATION COMPLETE")
    log("=" * 60)

    # Print final results
    output_dir = os.path.join(PROJECT, "outputs", "landmark_transformer_1000")
    log_csv = os.path.join(output_dir, "training_log.csv")
    if os.path.exists(log_csv):
        import csv
        with open(log_csv, "r") as f:
            reader = csv.DictReader(f)
            best_acc = 0
            best_epoch = 0
            for row in reader:
                acc = float(row["val_acc"])
                if acc > best_acc:
                    best_acc = acc
                    best_epoch = int(row["epoch"])
            log(f"BEST VALIDATION ACCURACY: {best_acc:.2f}% at epoch {best_epoch}")


if __name__ == "__main__":
    main()
