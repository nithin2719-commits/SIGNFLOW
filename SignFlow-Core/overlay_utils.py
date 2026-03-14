import os
import random
import sys

from PyQt5.QtGui import QImage

from overlay_constants import EXCLUDE_OVERLAY_FROM_CAPTURE

FRAME_DISPATCHER = None


def set_frame_dispatcher(callback):
    global FRAME_DISPATCHER
    FRAME_DISPATCHER = callback


def process_frame(frame):
    if callable(FRAME_DISPATCHER):
        FRAME_DISPATCHER(frame)


def stop_capture():
    pass


def restart_current_process():
    if getattr(sys, "frozen", False):
        os.execv(sys.executable, [sys.executable] + sys.argv[1:])
    else:
        script_path = os.path.abspath(sys.argv[0])
        os.execv(sys.executable, [sys.executable, script_path] + sys.argv[1:])


def generate_fake_status(system_state: str):
    hands = random.randint(0, 2)
    left_conf = random.random() if hands >= 1 else 0.0
    right_conf = random.random() if hands == 2 else 0.0
    fps = random.randint(20, 30)
    model_state = random.choice(["Idle", "Detecting Hands", "Processing Frame", "Waiting for Input"])
    capture_state = "Active" if system_state == "Running" else ("Paused" if system_state == "Paused" else "Idle")
    return {
        "System": system_state,
        "Capture Region": capture_state,
        "Hands Detected": hands,
        "Left Hand Confidence": left_conf,
        "Right Hand Confidence": right_conf,
        "Processing FPS": fps,
        "Model State": model_state,
    }


def _frame_to_qimage(frame):
    if not isinstance(frame, dict):
        return None
    rgb = frame.get("rgb")
    width = int(frame.get("width", 0) or 0)
    height = int(frame.get("height", 0) or 0)
    if rgb is None or width <= 0 or height <= 0:
        return None
    bytes_per_line = width * 3
    image = QImage(rgb, width, height, bytes_per_line, QImage.Format_RGB888)
    return image.copy()


def _set_window_excluded_from_capture(widget):
    if sys.platform != "win32":
        return
    try:
        import ctypes

        hwnd = int(widget.winId())
        affinity = 0x11 if EXCLUDE_OVERLAY_FROM_CAPTURE else 0x00
        ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, affinity)
    except Exception:
        pass
