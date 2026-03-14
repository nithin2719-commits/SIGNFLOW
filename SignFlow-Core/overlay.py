import sys

# IMPORTANT: torch must be imported BEFORE PyQt5 on Windows to avoid DLL conflicts
try:
    import torch  # noqa: F401
except Exception:
    pass

from PyQt5.QtWidgets import QApplication

from overlay_preferences import ensure_preferences_files
from overlay_window import OverlayWindow


def main():
    defaults, preferences = ensure_preferences_files()

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)

    overlay = OverlayWindow(defaults=defaults, preferences=preferences)
    overlay.show()
    overlay.raise_()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
