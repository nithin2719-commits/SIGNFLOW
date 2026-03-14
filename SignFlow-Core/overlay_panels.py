from PyQt5.QtCore import QRectF, QSize, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QFontMetrics, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from overlay_constants import (
    BUTTON_BG,
    BUTTON_COLUMN_SPACING,
    BUTTON_HEIGHT,
    BUTTON_HOVER_BG,
    BUTTON_WIDTH,
    CAPTION_HORIZONTAL_PADDING,
    CAPTION_VERTICAL_PADDING,
    CORNER_OPTIONS,
    DEFAULT_FONT_SIZE,
    DEFAULT_PRIMARY_BOX_SIZE,
    FONT_FAMILY,
    LABEL_DEFAULT_TEXT,
    MAX_OPACITY_PERCENT,
    MIN_OPACITY_PERCENT,
    MODEL_OPTIONS,
    OUTER_PADDING,
    OVERLAY_WIDTH,
    PRIMARY_BG,
    PRIMARY_BOX_SIZE_MAX,
    PRIMARY_BOX_SIZE_MIN,
    PRIMARY_INNER_SPACING,
    RADIUS,
    SECONDARY_ACTION_BUTTON_SIZE,
    SECONDARY_ACTION_ICON_SIZE,
    SECONDARY_ACTION_INDICATOR_ACTIVE,
    SECONDARY_ACTION_ROW_SPACING,
    SECONDARY_BG,
    SECONDARY_CHECKBOX_MIN_HEIGHT,
    SECONDARY_COLUMN_SPACING,
    SECONDARY_CONTROL_FONT_SIZE,
    SECONDARY_CONTROL_MIN_HEIGHT,
    SECONDARY_DROPDOWN_WIDTH,
    SECONDARY_INNER_SPACING,
    SECONDARY_LABEL_FONT_SIZE,
    SECONDARY_SLIDER_GROOVE_HEIGHT,
    SECONDARY_SLIDER_HANDLE_SIZE,
    TEXT_COLOR,
    BORDER_COLOR,
)


class PrimaryPanel(QFrame):
    toggle_requested = pyqtSignal()
    quit_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.user_box_size = DEFAULT_PRIMARY_BOX_SIZE
        self.setObjectName("primaryPanel")
        self.setFixedWidth(OVERLAY_WIDTH)

        root = QHBoxLayout(self)
        root.setContentsMargins(OUTER_PADDING, OUTER_PADDING, OUTER_PADDING, OUTER_PADDING)
        root.setSpacing(PRIMARY_INNER_SPACING)

        self.caption_label = QLabel(LABEL_DEFAULT_TEXT)
        self.caption_label.setWordWrap(True)
        self.caption_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.toggle_button = QPushButton("▾")
        self.toggle_button.setFixedSize(BUTTON_WIDTH, BUTTON_HEIGHT)
        self.toggle_button.clicked.connect(self.toggle_requested)

        self.quit_button = QPushButton("×")
        self.quit_button.setFixedSize(BUTTON_WIDTH, BUTTON_HEIGHT)
        self.quit_button.clicked.connect(self.quit_requested)

        right_buttons = QVBoxLayout()
        right_buttons.setSpacing(BUTTON_COLUMN_SPACING)
        right_buttons.addWidget(self.toggle_button)
        right_buttons.addWidget(self.quit_button)
        right_buttons.addStretch(1)

        root.addWidget(self.caption_label, 1)
        root.addLayout(right_buttons)

        self.setStyleSheet(
            f"""
            QFrame#primaryPanel {{
                background-color: {PRIMARY_BG};
                border: 1px solid {BORDER_COLOR};
                border-radius: {RADIUS}px;
            }}
            QLabel {{
                color: {TEXT_COLOR};
            }}
            QPushButton {{
                background-color: {BUTTON_BG};
                border: 1px solid {BORDER_COLOR};
                border-radius: 8px;
                color: {TEXT_COLOR};
                font: 600 13px '{FONT_FAMILY}';
            }}
            QPushButton:hover {{
                background-color: {BUTTON_HOVER_BG};
            }}
            """
        )

    def set_caption_text(self, text: str):
        self.caption_label.setText(text or LABEL_DEFAULT_TEXT)
        self._recompute_height()

    def set_caption_font_size(self, size: int):
        self.caption_label.setFont(QFont(FONT_FAMILY, DEFAULT_FONT_SIZE))
        self._recompute_height()

    def set_caption_box_size(self, size: int):
        self.user_box_size = max(PRIMARY_BOX_SIZE_MIN, min(PRIMARY_BOX_SIZE_MAX, int(size)))
        self._recompute_height()

    def set_expanded_icon(self, expanded: bool):
        self.toggle_button.setText("▴" if expanded else "▾")

    def _recompute_height(self):
        width = self.caption_label.width()
        if width < 120:
            fallback = OVERLAY_WIDTH - (OUTER_PADDING * 2) - BUTTON_WIDTH - PRIMARY_INNER_SPACING - (CAPTION_HORIZONTAL_PADDING * 2)
            width = max(120, fallback)

        metrics = QFontMetrics(self.caption_label.font())
        text_rect = metrics.boundingRect(0, 0, width, 10000, Qt.TextWordWrap, self.caption_label.text())
        caption_height = max(text_rect.height() + CAPTION_VERTICAL_PADDING, metrics.height() + CAPTION_VERTICAL_PADDING)

        self.caption_label.setMinimumHeight(caption_height)
        self.caption_label.setMaximumHeight(caption_height)
        self.caption_label.setContentsMargins(
            CAPTION_HORIZONTAL_PADDING,
            CAPTION_VERTICAL_PADDING // 2,
            CAPTION_HORIZONTAL_PADDING,
            CAPTION_VERTICAL_PADDING // 2,
        )

        controls_height = (BUTTON_HEIGHT * 2) + BUTTON_COLUMN_SPACING
        content_height = max(caption_height, controls_height)
        auto_height = (OUTER_PADDING * 2) + content_height
        panel_height = max(auto_height, self.user_box_size)
        self.setFixedHeight(panel_height)


class ThemedCheckBox(QCheckBox):
    def __init__(self, text: str):
        super().__init__(text)
        self._indicator_size = 16
        self._indicator_spacing = 10

    def sizeHint(self):
        base = super().sizeHint()
        width = self._indicator_size + self._indicator_spacing + base.width()
        height = max(base.height(), self._indicator_size)
        return base.expandedTo(base.__class__(width, height))

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        indicator_x = 0
        indicator_y = (self.height() - self._indicator_size) // 2
        indicator_rect = self.rect().adjusted(
            indicator_x,
            indicator_y,
            -(self.width() - self._indicator_size),
            -(self.height() - indicator_y - self._indicator_size),
        )

        border_color = QColor(255, 255, 255, 110 if self.isChecked() else 85)
        fill_color = QColor(255, 255, 255, 48 if self.isChecked() else 18)
        painter.setPen(QPen(border_color, 1))
        painter.setBrush(fill_color)
        painter.drawRoundedRect(indicator_rect, 3, 3)

        if self.isChecked():
            check_pen = QPen(QColor(245, 245, 245, 240), 2)
            painter.setPen(check_pen)
            x = indicator_rect.x()
            y = indicator_rect.y()
            w = indicator_rect.width()
            h = indicator_rect.height()
            painter.drawLine(x + int(w * 0.20), y + int(h * 0.55), x + int(w * 0.42), y + int(h * 0.78))
            painter.drawLine(x + int(w * 0.42), y + int(h * 0.78), x + int(w * 0.80), y + int(h * 0.28))

        text_x = self._indicator_size + self._indicator_spacing
        text_rect = self.rect().adjusted(text_x, 0, 0, 0)
        painter.setPen(self.palette().color(self.foregroundRole()))
        painter.setFont(self.font())
        painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, self.text())


class ThemedComboBox(QComboBox):
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(245, 245, 245, 230))

        cx = self.width() - (SECONDARY_DROPDOWN_WIDTH // 2)
        cy = (self.height() // 2) + 1
        half_w = 4
        half_h = 3

        path = QPainterPath()
        path.moveTo(cx - half_w, cy - half_h)
        path.lineTo(cx + half_w, cy - half_h)
        path.lineTo(cx, cy + half_h)
        path.closeSubpath()
        painter.drawPath(path)
        painter.end()


class SecondaryPanel(QFrame):
    crop_clicked = pyqtSignal()
    play_pause_toggled = pyqtSignal(bool)
    clear_clicked = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setObjectName("secondaryPanel")
        self.setFixedWidth(OVERLAY_WIDTH)
        self.setFixedHeight(0)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._is_playing = False

        root = QVBoxLayout(self)
        root.setContentsMargins(OUTER_PADDING, OUTER_PADDING, OUTER_PADDING, OUTER_PADDING)
        root.setSpacing(SECONDARY_COLUMN_SPACING)

        action_row = QHBoxLayout()
        action_row.setSpacing(SECONDARY_ACTION_ROW_SPACING)
        action_row.setContentsMargins(0, 0, 0, 0)
        self.crop_button = QPushButton("")
        self.crop_button.setObjectName("actionButton")
        self.crop_button.setMinimumHeight(SECONDARY_ACTION_BUTTON_SIZE)
        self.crop_button.setMaximumHeight(SECONDARY_ACTION_BUTTON_SIZE)
        self.crop_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.crop_button.setFocusPolicy(Qt.NoFocus)
        self.crop_button.clicked.connect(self.crop_clicked.emit)


        self.play_pause_button = QPushButton("")
        self.play_pause_button.setObjectName("actionToggleButton")
        self.play_pause_button.setMinimumHeight(SECONDARY_ACTION_BUTTON_SIZE)
        self.play_pause_button.setMaximumHeight(SECONDARY_ACTION_BUTTON_SIZE)
        self.play_pause_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.play_pause_button.setFocusPolicy(Qt.NoFocus)
        self.play_pause_button.clicked.connect(self._toggle_play_pause)

        self.clear_button = QPushButton("")
        self.clear_button.setObjectName("actionButton")
        self.clear_button.setMinimumHeight(SECONDARY_ACTION_BUTTON_SIZE)
        self.clear_button.setMaximumHeight(SECONDARY_ACTION_BUTTON_SIZE)
        self.clear_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.clear_button.setFocusPolicy(Qt.NoFocus)
        self.clear_button.clicked.connect(self.clear_clicked.emit)

        self._set_action_icon_sizes()
        self.crop_button.setIcon(self._build_crop_icon(SECONDARY_ACTION_ICON_SIZE))
        self.clear_button.setIcon(self._build_region_icon(SECONDARY_ACTION_ICON_SIZE))
        self._apply_play_pause_icon()

        self._status_active = False
        self.status_indicator = QLabel()
        self._apply_status_indicator()

        self.status_indicator.setObjectName("actionStatus")
        self.status_indicator.setAlignment(Qt.AlignCenter)
        self.status_indicator.setTextFormat(Qt.RichText)
        self.status_indicator.setMinimumHeight(SECONDARY_ACTION_BUTTON_SIZE)
        self.status_indicator.setMaximumHeight(SECONDARY_ACTION_BUTTON_SIZE)
        self.status_indicator.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        action_row.addWidget(self.crop_button, 1)
        action_row.addWidget(self.play_pause_button, 1)
        action_row.addWidget(self.clear_button, 1)
        action_row.addWidget(self.status_indicator, 3)

        action_divider = QFrame()
        action_divider.setObjectName("secondaryDivider")
        action_divider.setFrameShape(QFrame.HLine)
        action_divider.setFrameShadow(QFrame.Plain)

        columns_row = QHBoxLayout()
        columns_row.setSpacing(SECONDARY_INNER_SPACING)

        left_col = QVBoxLayout()
        left_col.setSpacing(SECONDARY_COLUMN_SPACING)

        right_col = QVBoxLayout()
        right_col.setSpacing(SECONDARY_COLUMN_SPACING)

        self.caption_box_size_slider = QSlider(Qt.Horizontal)
        self.caption_box_size_slider.setRange(PRIMARY_BOX_SIZE_MIN, PRIMARY_BOX_SIZE_MAX)

        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(MIN_OPACITY_PERCENT, MAX_OPACITY_PERCENT)

        self.show_raw_tokens_checkbox = ThemedCheckBox("Show raw tokens")
        self.freeze_on_loss_checkbox = ThemedCheckBox("Show model status")

        self.restart_button = QPushButton("Restart")
        self.restart_button.setObjectName("restartButton")
        self.restart_button.setMinimumHeight(SECONDARY_CONTROL_MIN_HEIGHT)

        self.enable_llm_checkbox = ThemedCheckBox("Enable LLM smoothing")

        self.model_combo = ThemedComboBox()
        self.model_combo.addItems(MODEL_OPTIONS)

        self.show_latency_checkbox = ThemedCheckBox("Show latency")

        self.corner_combo = ThemedComboBox()
        self.corner_combo.addItems(CORNER_OPTIONS)

        self.reset_preferences_button = QPushButton("Reset Preferences To Default")
        self.reset_preferences_button.setObjectName("restartButton")
        self.reset_preferences_button.setMinimumHeight(SECONDARY_CONTROL_MIN_HEIGHT)

        left_col.addLayout(self._labeled_row("Caption box size", self.caption_box_size_slider))
        left_col.addLayout(self._labeled_row("Overlay opacity", self.opacity_slider))
        left_col.addWidget(self.show_raw_tokens_checkbox)
        left_col.addWidget(self.freeze_on_loss_checkbox)
        left_col.addWidget(self.restart_button)
        left_col.addStretch(1)

        right_col.addWidget(self.enable_llm_checkbox)
        right_col.addLayout(self._labeled_row("Model selection", self.model_combo))
        right_col.addWidget(self.show_latency_checkbox)
        right_col.addLayout(self._labeled_row("Overlay corner", self.corner_combo))
        right_col.addStretch(1)

        columns_row.addLayout(left_col, 1)
        columns_row.addLayout(right_col, 1)

        root.addLayout(action_row)
        root.addWidget(action_divider)
        root.addLayout(columns_row)
        root.addWidget(self.reset_preferences_button)

        self.setStyleSheet(
            f"""
            QFrame#secondaryPanel {{
                background-color: {SECONDARY_BG};
                border: 1px solid {BORDER_COLOR};
                border-radius: {RADIUS}px;
            }}
            QFrame#secondaryDivider {{
                border: none;
                min-height: 1px;
                max-height: 1px;
                background-color: rgba(255, 255, 255, 30);
            }}
            QLabel, QCheckBox {{
                color: {TEXT_COLOR};
                font: 500 {SECONDARY_LABEL_FONT_SIZE}px '{FONT_FAMILY}';
            }}
            QCheckBox {{
                min-height: {SECONDARY_CHECKBOX_MIN_HEIGHT}px;
                spacing: 10px;
            }}
            QComboBox {{
                background-color: rgba(255, 255, 255, 24);
                color: {TEXT_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: 6px;
                padding: 6px 10px;
                padding-right: 28px;
                min-height: {SECONDARY_CONTROL_MIN_HEIGHT}px;
                font: 500 {SECONDARY_CONTROL_FONT_SIZE}px '{FONT_FAMILY}';
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: {SECONDARY_DROPDOWN_WIDTH}px;
                border: none;
                border-left: 1px solid {BORDER_COLOR};
                background: transparent;
                border-top-right-radius: 6px;
                border-bottom-right-radius: 6px;
            }}
            QComboBox::down-arrow {{
                image: none;
                width: 0px;
                height: 0px;
                border: none;
                margin: 0px;
            }}
            QComboBox QAbstractItemView {{
                background-color: rgba(40, 40, 43, 240);
                color: {TEXT_COLOR};
                border: 1px solid {BORDER_COLOR};
                selection-background-color: rgba(255, 255, 255, 46);
                selection-color: {TEXT_COLOR};
                outline: 0;
                padding: 6px;
                font: 500 {SECONDARY_CONTROL_FONT_SIZE}px '{FONT_FAMILY}';
            }}
            QSlider::groove:horizontal {{
                border: none;
                height: {SECONDARY_SLIDER_GROOVE_HEIGHT}px;
                background: rgba(255, 255, 255, 36);
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: rgba(255, 255, 255, 180);
                border: none;
                width: {SECONDARY_SLIDER_HANDLE_SIZE}px;
                margin: -5px 0;
                border-radius: 8px;
            }}
            QPushButton#restartButton {{
                background-color: {BUTTON_BG};
                border: 1px solid {BORDER_COLOR};
                border-radius: 8px;
                color: {TEXT_COLOR};
                font: 600 {SECONDARY_CONTROL_FONT_SIZE}px '{FONT_FAMILY}';
                padding: 4px 10px;
            }}
            QPushButton#restartButton:hover {{
                background-color: {BUTTON_HOVER_BG};
            }}
            QPushButton#actionButton {{
                background-color: {BUTTON_BG};
                border: 1px solid {BORDER_COLOR};
                border-radius: 7px;
                color: {TEXT_COLOR};
                font: 600 18px '{FONT_FAMILY}';
            }}
            QPushButton#actionButton:hover {{
                background-color: {BUTTON_HOVER_BG};
            }}
            QPushButton#actionButton:focus {{
                outline: none;
                border: 1px solid {BORDER_COLOR};
            }}
            QPushButton#actionToggleButton {{
                background-color: {BUTTON_BG};
                border: 1px solid {BORDER_COLOR};
                border-radius: 7px;
                color: {TEXT_COLOR};
                font: 600 18px '{FONT_FAMILY}';
            }}
            QPushButton#actionToggleButton:hover {{
                background-color: {BUTTON_HOVER_BG};
            }}
            QPushButton#actionToggleButton:pressed {{
                background-color: rgba(255, 255, 255, 38);
                border: 1px solid {BORDER_COLOR};
            }}
            QPushButton#actionToggleButton:focus {{
                outline: none;
                border: 1px solid {BORDER_COLOR};
            }}
            QLabel#actionStatus {{
                color: {TEXT_COLOR};
                background: rgba(255, 255, 255, 10);
                border: 1px solid {BORDER_COLOR};
                border-radius: 7px;
                padding: 0 10px;
                font: 600 13px '{FONT_FAMILY}';
            }}
            """
        )

    @staticmethod
    def _labeled_row(title: str, widget: QWidget):
        layout = QVBoxLayout()
        layout.setSpacing(14 if isinstance(widget, QSlider) else 10)
        label = QLabel(title)
        label.setMinimumHeight(int(SECONDARY_LABEL_FONT_SIZE * 1.4))
        layout.addWidget(label)
        layout.addWidget(widget)
        return layout

    def _set_action_icon_sizes(self):
        icon_size = QSize(SECONDARY_ACTION_ICON_SIZE, SECONDARY_ACTION_ICON_SIZE)
        self.crop_button.setIconSize(icon_size)
        self.play_pause_button.setIconSize(icon_size)
        self.clear_button.setIconSize(icon_size)

    def _new_icon_canvas(self, size: int):
        pix = QPixmap(size, size)
        pix.fill(Qt.transparent)
        return pix

    def _build_crop_icon(self, size: int):
        pix = self._new_icon_canvas(size)
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.Antialiasing, True)
        pen = QPen(QColor(245, 245, 245, 235), 1.8)
        painter.setPen(pen)
        m = 3
        painter.drawRect(m, m, size - (m * 2), size - (m * 2))
        painter.drawLine(m, size // 3, m, m)
        painter.drawLine(size // 3, m, m, m)
        painter.end()
        return QIcon(pix)

    def _build_region_icon(self, size: int):
        pix = self._new_icon_canvas(size)
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.Antialiasing, True)
        pen = QPen(QColor(245, 245, 245, 235), 1.6)
        pen.setCapStyle(Qt.SquareCap)
        painter.setPen(pen)

        s = float(size)
        pad = int(s * 0.18)
        x1 = int(s * 0.40)
        x2 = int(s * 0.60)
        y1 = int(s * 0.40)
        y2 = int(s * 0.60)
        left = pad
        right = int(s - pad)
        top = pad
        bottom = int(s - pad)

        # 3x3 tic-tac-toe grid
        painter.drawLine(x1, top, x1, bottom)
        painter.drawLine(x2, top, x2, bottom)
        painter.drawLine(left, y1, right, y1)
        painter.drawLine(left, y2, right, y2)

        painter.end()
        return QIcon(pix)

    def _build_play_icon(self, size: int):
        pix = self._new_icon_canvas(size)
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(245, 245, 245, 235))
        path = QPainterPath()
        path.moveTo(size * 0.30, size * 0.20)
        path.lineTo(size * 0.30, size * 0.80)
        path.lineTo(size * 0.78, size * 0.50)
        path.closeSubpath()
        painter.drawPath(path)
        painter.end()
        return QIcon(pix)

    def _build_pause_icon(self, size: int):
        pix = self._new_icon_canvas(size)
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(245, 245, 245, 235))
        w = size * 0.20
        gap = size * 0.14
        left = (size - (w * 2 + gap)) / 2.0
        r1 = QRectF(left, size * 0.20, w, size * 0.60)
        r2 = QRectF(left + w + gap, size * 0.20, w, size * 0.60)
        painter.drawRoundedRect(r1, 1.5, 1.5)
        painter.drawRoundedRect(r2, 1.5, 1.5)
        painter.end()
        return QIcon(pix)

    def _build_clear_icon(self, size: int):
        pix = self._new_icon_canvas(size)
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.Antialiasing, True)
        pen = QPen(QColor(245, 245, 245, 235), 2.0)
        painter.setPen(pen)
        m = int(size * 0.24)
        painter.drawLine(m, m, int(size - m), int(size - m))
        painter.drawLine(m, int(size - m), int(size - m), m)
        painter.end()
        return QIcon(pix)

    def set_playing(self, is_playing: bool):
        self._is_playing = bool(is_playing)
        self._apply_play_pause_icon()

    def _apply_status_indicator(self):
        indicator_symbol = "●" if self._status_active else "○"
        indicator_state = "Active" if self._status_active else "Inactive"
        indicator_symbol_color = "rgb(80, 200, 120)" if self._status_active else "rgb(145, 145, 145)"
        self.status_indicator.setText(
            f"Status: {indicator_state} "
            f"<span style=\"color:{indicator_symbol_color}; font-size:16px;\">{indicator_symbol}</span>"
        )

    def set_status_active(self, active: bool):
        self._status_active = bool(active)
        self._apply_status_indicator()

    def _apply_play_pause_icon(self):
        if self._is_playing:
            self.play_pause_button.setIcon(self._build_pause_icon(SECONDARY_ACTION_ICON_SIZE))
        else:
            self.play_pause_button.setIcon(self._build_play_icon(SECONDARY_ACTION_ICON_SIZE))

    def _toggle_play_pause(self):
        self._is_playing = not self._is_playing
        self._apply_play_pause_icon()
        self.play_pause_toggled.emit(self._is_playing)
