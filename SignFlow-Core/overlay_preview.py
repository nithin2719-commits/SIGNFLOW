from PyQt5.QtCore import QPoint, Qt, QTimer
from PyQt5.QtGui import QGuiApplication, QImage, QPixmap
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from overlay_constants import (
    FONT_FAMILY,
    PREVIEW_HEIGHT,
    PREVIEW_HINT_TEXT,
    PREVIEW_MARGIN,
    PREVIEW_REGION_BG,
    PREVIEW_REGION_TEXT,
    PREVIEW_TITLE_BG,
    PREVIEW_TITLE_HEIGHT,
    PREVIEW_TITLE_SUBTEXT,
    PREVIEW_TITLE_TEXT,
    PREVIEW_WIDTH,
    STATUS_PANEL_BG,
    STATUS_PANEL_BORDER,
    STATUS_PANEL_FONT_SIZE,
    STATUS_PANEL_PADDING,
    STATUS_PANEL_TEXT,
    STATUS_PANEL_TITLE,
    STATUS_PANEL_TITLE_SIZE,
)
from overlay_utils import _set_window_excluded_from_capture


class PreviewWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, False)

        self._status_visible = False
        self._capture_state = "IDLE"
        self._region = None
        self._show_first_hint = True
        self._dragging = False
        self._drag_offset = QPoint()

        QTimer.singleShot(0, lambda: _set_window_excluded_from_capture(self))

        self.title_bar = QWidget()
        self.title_bar.setObjectName("previewTitleBar")
        self.title_bar.setFixedHeight(PREVIEW_TITLE_HEIGHT)

        self.title_label = QLabel("SignFlow Capture")
        self.title_label.setObjectName("previewTitle")

        self.state_label = QLabel()
        self.state_label.setObjectName("previewState")
        self._apply_capture_state()

        title_layout = QHBoxLayout(self.title_bar)
        title_layout.setContentsMargins(10, 4, 10, 4)
        title_layout.setSpacing(8)
        title_layout.addWidget(self.title_label, 1)
        title_layout.addWidget(self.state_label, 0, Qt.AlignRight)

        self.preview_container = QFrame()
        self.preview_container.setObjectName("previewContainer")
        self.preview_container.setFixedSize(PREVIEW_WIDTH, PREVIEW_HEIGHT)

        self.label = QLabel(self.preview_container)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("background-color: rgba(0, 0, 0, 210); border: 1px solid rgba(255, 255, 255, 40);")

        self.region_label = QLabel(self.preview_container)
        self.region_label.setObjectName("previewRegion")
        self.region_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.region_label.setVisible(False)

        self.empty_label = QLabel("No capture region selected\nClick the capture button to begin", self.preview_container)
        self.empty_label.setObjectName("previewEmpty")
        self.empty_label.setAlignment(Qt.AlignCenter)

        self.hint_label = QLabel("Click the capture button to select a signing video.", self.preview_container)
        self.hint_label.setObjectName("previewHint")
        self.hint_label.setAlignment(Qt.AlignCenter)

        self.status_panel = QWidget()
        self.status_panel.setVisible(False)
        self.status_panel.setObjectName("statusPanel")

        self.status_title = QLabel("Current Status")
        self.status_title.setObjectName("statusTitle")

        self.status_body = QLabel("")
        self.status_body.setObjectName("statusBody")
        self.status_body.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.status_body.setWordWrap(True)

        status_layout = QVBoxLayout(self.status_panel)
        status_layout.setContentsMargins(STATUS_PANEL_PADDING, STATUS_PANEL_PADDING, STATUS_PANEL_PADDING, STATUS_PANEL_PADDING)
        status_layout.setSpacing(6)
        status_layout.addWidget(self.status_title)
        status_layout.addWidget(self.status_body)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.title_bar)
        layout.addWidget(self.preview_container)
        layout.addWidget(self.status_panel)

        self.setStyleSheet(
            f"""
            QWidget#previewTitleBar {{
                background-color: {PREVIEW_TITLE_BG};
                border: 1px solid rgba(255, 255, 255, 24);
                border-bottom: none;
            }}
            QLabel#previewTitle {{
                color: {PREVIEW_TITLE_TEXT};
                font: 600 12px '{FONT_FAMILY}';
            }}
            QLabel#previewState {{
                color: {PREVIEW_TITLE_SUBTEXT};
                font: 600 11px '{FONT_FAMILY}';
            }}
            QFrame#previewContainer {{
                background-color: rgba(0, 0, 0, 210);
                border-left: 1px solid rgba(255, 255, 255, 40);
                border-right: 1px solid rgba(255, 255, 255, 40);
                border-bottom: 1px solid rgba(255, 255, 255, 40);
            }}
            QLabel#previewRegion {{
                color: {PREVIEW_REGION_TEXT};
                background-color: {PREVIEW_REGION_BG};
                border-radius: 6px;
                padding: 2px 8px;
                font: 600 11px '{FONT_FAMILY}';
            }}
            QLabel#previewEmpty {{
                color: {PREVIEW_TITLE_TEXT};
                font: 600 12px '{FONT_FAMILY}';
            }}
            QLabel#previewHint {{
                color: {PREVIEW_HINT_TEXT};
                font: 500 11px '{FONT_FAMILY}';
            }}
            QWidget#statusPanel {{
                background-color: {STATUS_PANEL_BG};
                border: 1px solid {STATUS_PANEL_BORDER};
                border-top: none;
            }}
            QLabel#statusTitle {{
                color: {STATUS_PANEL_TITLE};
                font: 600 {STATUS_PANEL_TITLE_SIZE}px '{FONT_FAMILY}';
            }}
            QLabel#statusBody {{
                color: {STATUS_PANEL_TEXT};
                font: 500 {STATUS_PANEL_FONT_SIZE}px '{FONT_FAMILY}';
            }}
            """
        )

        self._update_empty_state()
        self._layout_preview_overlays()
        self._update_window_size()
        self._position_near_corner()

    def _apply_capture_state(self):
        state = self._capture_state
        if state == "LIVE":
            color = "rgb(80, 200, 120)"
            text = "LIVE"
        elif state == "PAUSED":
            color = "rgb(240, 200, 80)"
            text = "PAUSED"
        else:
            color = "rgb(145, 145, 145)"
            text = "IDLE"
        self.state_label.setText(f"<span style=\"color:{color};\">●</span> {text}")

    def set_capture_state(self, state: str):
        normalized = (state or "IDLE").upper()
        if normalized == "RUNNING":
            normalized = "LIVE"
        elif normalized == "PAUSED":
            normalized = "PAUSED"
        else:
            normalized = "IDLE"
        if normalized != self._capture_state:
            self._capture_state = normalized
            self._apply_capture_state()

    def set_region_info(self, region: dict | None, show_hint: bool):
        self._region = region
        self._show_first_hint = bool(show_hint)
        if region:
            width = int(region.get("width", 0))
            height = int(region.get("height", 0))
            if width > 0 and height > 0:
                self.region_label.setText(f"Region: {width} × {height}")
                self.region_label.setVisible(True)
            else:
                self.region_label.setVisible(False)
        else:
            self.region_label.setVisible(False)
        self._update_empty_state()
        self._layout_preview_overlays()

    def _update_empty_state(self):
        has_region = bool(self._region)
        self.empty_label.setVisible(not has_region)
        self.hint_label.setVisible(not has_region and self._show_first_hint)

    def _layout_preview_overlays(self):
        self.label.setGeometry(0, 0, PREVIEW_WIDTH, PREVIEW_HEIGHT)
        margin = 10
        if self.region_label.isVisible():
            self.region_label.adjustSize()
            w = self.region_label.width()
            h = self.region_label.height()
            self.region_label.move(PREVIEW_WIDTH - w - margin, margin)
        if self.empty_label.isVisible():
            self.empty_label.adjustSize()
            hint_space = 0
            if self.hint_label.isVisible():
                self.hint_label.adjustSize()
                hint_space = self.hint_label.height() + 6
            total_height = self.empty_label.height() + hint_space
            y = int((PREVIEW_HEIGHT - total_height) / 2)
            self.empty_label.move(int((PREVIEW_WIDTH - self.empty_label.width()) / 2), y)
            if self.hint_label.isVisible():
                self.hint_label.move(int((PREVIEW_WIDTH - self.hint_label.width()) / 2), y + self.empty_label.height() + 6)

    def _update_window_size(self):
        height = PREVIEW_TITLE_HEIGHT + PREVIEW_HEIGHT
        if self._status_visible:
            self.status_panel.adjustSize()
            status_height = self.status_panel.sizeHint().height()
            height += status_height
        self.setFixedSize(PREVIEW_WIDTH, height)

    def set_status_visible(self, visible: bool):
        self._status_visible = bool(visible)
        self.status_panel.setVisible(self._status_visible)
        self._update_window_size()

    def set_status_text(self, text: str):
        self.status_body.setText(text or "")
        if self._status_visible:
            self._update_window_size()

    def _position_near_corner(self):
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        x = geo.x() + geo.width() - self.width() - PREVIEW_MARGIN
        y = geo.y() + PREVIEW_MARGIN
        self.move(x, y)

    def showEvent(self, event):
        super().showEvent(event)
        self._position_near_corner()
        _set_window_excluded_from_capture(self)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._layout_preview_overlays()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._drag_offset = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging and event.buttons() & Qt.LeftButton:
            self.move(event.globalPos() - self._drag_offset)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = False
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def update_frame(self, image: QImage):
        if image is None:
            return
        pixmap = QPixmap.fromImage(image)
        scaled = pixmap.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label.setPixmap(scaled)
