import time

import mss
from PyQt5.QtCore import QThread, pyqtSignal

from overlay_constants import CAPTURE_FPS


class ScreenCaptureThread(QThread):
    frame_captured = pyqtSignal(object)

    def __init__(self, region: dict, parent=None):
        super().__init__(parent)
        self._region = dict(region) if region else None
        self._running = True

    def run(self):
        if not self._region:
            return
        monitor = {
            "left": int(self._region["x"]),
            "top": int(self._region["y"]),
            "width": int(self._region["width"]),
            "height": int(self._region["height"]),
        }
        frame_interval = 1.0 / float(CAPTURE_FPS)
        with mss.mss() as sct:
            next_frame_time = time.perf_counter()
            while self._running:
                screenshot = sct.grab(monitor)
                self.frame_captured.emit(
                    {
                        "rgb": screenshot.rgb,
                        "width": screenshot.width,
                        "height": screenshot.height,
                    }
                )
                next_frame_time += frame_interval
                sleep_for = next_frame_time - time.perf_counter()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    next_frame_time = time.perf_counter()

    def stop(self):
        self._running = False
        self.wait(500)
