import time

from PyQt5.QtCore import (
    QAbstractAnimation,
    QEasingCurve,
    QRect,
    Qt,
    QTimer,
    QVariantAnimation,
)
from PyQt5.QtGui import QGuiApplication, QRegion
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget

from overlay_capture import ScreenCaptureThread
from overlay_constants import (
    ANIMATION_DURATION_MS,
    CAPTURE_FPS,
    CAPTURE_FLIP_HORIZONTAL,
    CORNER_BOTTOM_LEFT,
    CORNER_BOTTOM_RIGHT,
    CORNER_TOP_LEFT,
    CORNER_TOP_RIGHT,
    DEFAULT_FONT_SIZE,
    DEFAULT_SETTINGS,
    DEFAULT_SETTINGS_PATH,
    ENABLE_COLLAPSE_ANIMATION,
    FONT_FAMILY,
    HIGHLIGHT_DURATION_MS,
    LABEL_DEFAULT_TEXT,
    MAX_OPACITY_PERCENT,
    MIN_OPACITY_PERCENT,
    OUTER_PADDING,
    OVERLAY_MARGIN,
    OVERLAY_WIDTH,
    PANEL_SPACING,
    PRIMARY_BOX_SIZE_MAX,
    PRIMARY_BOX_SIZE_MIN,
    SECONDARY_EXPANDED_HEIGHT,
    STATUS_UPDATE_INTERVAL_MS,
)
from overlay_hand_tracking import HandTrackingWorker
from overlay_panels import PrimaryPanel, SecondaryPanel
from overlay_preferences import _read_json, _sanitize_settings, save_user_preferences
from overlay_preview import PreviewWindow
from overlay_selection import HighlightOverlay, RegionSelectionOverlay
from overlay_utils import (
    _frame_to_qimage,
    _set_window_excluded_from_capture,
    process_frame,
    restart_current_process,
    set_frame_dispatcher,
    stop_capture,
)


class OverlayWindow(QWidget):
    def __init__(self, defaults, preferences):
        super().__init__()

        self.defaults = defaults
        self.preferences = preferences
        self.caption_text = LABEL_DEFAULT_TEXT
        self.caption_font_size = DEFAULT_FONT_SIZE
        self.applied_caption_box_size = self.preferences["caption_box_size"]
        self.pending_caption_box_size = self.preferences["caption_box_size"]
        self.overlay_opacity = self.preferences["opacity_percent"] / 100.0
        self.show_raw_tokens = self.preferences["show_raw_tokens"]
        self.freeze_on_detection_loss = self.preferences["freeze_on_detection_loss"]
        self.enable_llm_smoothing = self.preferences["enable_llm_smoothing"]
        self.model_selection = self.preferences["model_selection"]
        self.show_latency = self.preferences["show_latency"]
        self.corner = self.preferences["corner"]
        self.secondary_expanded = False
        self.secondary_current_height = 0

        self.capture_state = {"region": None, "paused": False}
        self.capture_thread = None
        self.preview_window = None
        self.first_launch_hint = True
        self.selection_overlay = None
        self.highlight_overlay = None
        self.status_timer = QTimer(self)
        self.status_timer.setInterval(STATUS_UPDATE_INTERVAL_MS)
        self.status_timer.timeout.connect(self._update_status_panel)
        self.hand_worker = None
        self.last_detection = {"hands_detected": 0, "left_conf": 0.0, "right_conf": 0.0}
        self._processing_fps = 0.0
        self._capture_fps = 0.0
        self._latest_frame = None
        self._latest_processed_frame = None
        self._latest_frame_time = None
        self._latest_processed_time = None
        self._capture_frame_time = None
        self._last_prediction = None
        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(max(1, int(1000 / max(1, CAPTURE_FPS))))
        self._preview_timer.timeout.connect(self._update_preview_frame)

        set_frame_dispatcher(self._handle_frame)

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        QTimer.singleShot(0, lambda: _set_window_excluded_from_capture(self))

        self.root_layout = QVBoxLayout(self)
        self.root_layout.setContentsMargins(OUTER_PADDING, OUTER_PADDING, OUTER_PADDING, OUTER_PADDING)
        self.root_layout.setSpacing(0)

        self.primary_panel = PrimaryPanel()
        self.secondary_panel = SecondaryPanel()

        self.inter_panel_spacer = QWidget()
        self.inter_panel_spacer.setFixedHeight(0)
        self.inter_panel_spacer.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        self.secondary_animation = QVariantAnimation(self)
        self.secondary_animation.setDuration(ANIMATION_DURATION_MS)
        self.secondary_animation.setEasingCurve(QEasingCurve.InOutCubic)
        self.secondary_animation.valueChanged.connect(self.on_secondary_animation_value)
        self.secondary_animation.finished.connect(self.on_secondary_animation_finished)

        self._rebuild_stack()
        self._connect_signals()
        self.primary_panel.set_expanded_icon(self.secondary_expanded)
        self.apply_state_to_ui()

        app = QApplication.instance()
        if app is not None:
            app.screenAdded.connect(lambda _screen: self._position_window())
            app.screenRemoved.connect(lambda _screen: self._position_window())

        primary_screen = QGuiApplication.primaryScreen()
        if primary_screen is not None:
            primary_screen.geometryChanged.connect(lambda _rect: self._position_window())

    def showEvent(self, event):
        super().showEvent(event)
        _set_window_excluded_from_capture(self)

    def _write_preferences(self):
        self.preferences["caption_box_size"] = self.pending_caption_box_size
        self.preferences["opacity_percent"] = int(round(self.overlay_opacity * 100))
        self.preferences["show_raw_tokens"] = self.show_raw_tokens
        self.preferences["freeze_on_detection_loss"] = self.freeze_on_detection_loss
        self.preferences["enable_llm_smoothing"] = self.enable_llm_smoothing
        self.preferences["model_selection"] = self.model_selection
        self.preferences["show_latency"] = self.show_latency
        self.preferences["corner"] = self.corner
        save_user_preferences(self.preferences)

    def _connect_signals(self):
        self.primary_panel.toggle_requested.connect(self.toggle_secondary_panel)
        self.primary_panel.quit_requested.connect(QApplication.instance().quit)

        self.secondary_panel.caption_box_size_slider.valueChanged.connect(self.on_caption_box_size_changed)
        self.secondary_panel.opacity_slider.valueChanged.connect(self.on_opacity_changed)
        self.secondary_panel.show_raw_tokens_checkbox.toggled.connect(self.on_show_raw_tokens_toggled)
        self.secondary_panel.freeze_on_loss_checkbox.toggled.connect(self.on_freeze_on_loss_toggled)
        self.secondary_panel.enable_llm_checkbox.toggled.connect(self.on_enable_llm_toggled)
        self.secondary_panel.model_combo.currentTextChanged.connect(self.on_model_changed)
        self.secondary_panel.show_latency_checkbox.toggled.connect(self.on_show_latency_toggled)
        self.secondary_panel.corner_combo.currentTextChanged.connect(self.on_corner_changed)
        self.secondary_panel.freeze_on_loss_checkbox.toggled.connect(self.on_show_model_status_toggled)
        self.secondary_panel.restart_button.clicked.connect(self.on_restart_requested)
        self.secondary_panel.reset_preferences_button.clicked.connect(self.on_reset_preferences_requested)
        self.secondary_panel.crop_clicked.connect(self.on_fullscreen_capture)
        self.secondary_panel.play_pause_toggled.connect(self.on_play_pause_toggled)
        self.secondary_panel.clear_clicked.connect(self.on_crop_clicked)

    def _rebuild_stack(self):
        while self.root_layout.count():
            item = self.root_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

        if self.corner in (CORNER_BOTTOM_LEFT, CORNER_BOTTOM_RIGHT):
            self.root_layout.addStretch(1)
            self.root_layout.addWidget(self.secondary_panel)
            self.root_layout.addWidget(self.inter_panel_spacer)
            self.root_layout.addWidget(self.primary_panel)
        else:
            self.root_layout.addWidget(self.primary_panel)
            self.root_layout.addWidget(self.inter_panel_spacer)
            self.root_layout.addWidget(self.secondary_panel)
            self.root_layout.addStretch(1)

    def _screen_geometry(self):
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return None
        return screen.availableGeometry()

    def _primary_height(self):
        h = self.primary_panel.height()
        if h > 0:
            return h
        return self.primary_panel.sizeHint().height()

    def _full_window_height(self):
        return (OUTER_PADDING * 2) + self._primary_height() + PANEL_SPACING + SECONDARY_EXPANDED_HEIGHT

    def _visible_stack_height(self):
        extra = PANEL_SPACING + self.secondary_current_height if self.secondary_current_height > 0 else 0
        return (OUTER_PADDING * 2) + self._primary_height() + extra

    def _set_secondary_height(self, height: int, force_hide: bool = False):
        clamped = max(0, min(SECONDARY_EXPANDED_HEIGHT, int(height)))
        self.secondary_current_height = clamped
        self.inter_panel_spacer.setFixedHeight(PANEL_SPACING if clamped > 0 else 0)
        self.secondary_panel.setFixedHeight(clamped)

        if force_hide or clamped == 0:
            self.secondary_panel.hide()
        else:
            self.secondary_panel.show()

    def _update_mask(self):
        visible_height = max(1, min(self._visible_stack_height(), self.height()))
        y_offset = 0
        if self.corner in (CORNER_BOTTOM_LEFT, CORNER_BOTTOM_RIGHT):
            y_offset = self.height() - visible_height
        self.setMask(QRegion(0, y_offset, self.width(), visible_height))

    def _position_window(self):
        geo = self._screen_geometry()
        if geo is None:
            return

        if self.corner in (CORNER_TOP_LEFT, CORNER_BOTTOM_LEFT):
            x = geo.x() + OVERLAY_MARGIN
        else:
            x = geo.x() + geo.width() - self.width() - OVERLAY_MARGIN

        if self.corner in (CORNER_TOP_LEFT, CORNER_TOP_RIGHT):
            y = geo.y() + OVERLAY_MARGIN
        else:
            y = geo.y() + geo.height() - self.height() - OVERLAY_MARGIN

        self.move(x, y)

    def _refresh_window_geometry(self, reposition: bool):
        self.setFixedSize(OVERLAY_WIDTH + (OUTER_PADDING * 2), self._full_window_height())
        self._update_mask()
        if reposition:
            self._position_window()

    def apply_state_to_ui(self):
        self.primary_panel.set_caption_text(self.caption_text)
        self.primary_panel.set_caption_font_size(self.caption_font_size)
        self.primary_panel.set_caption_box_size(self.applied_caption_box_size)
        self.setWindowOpacity(self.overlay_opacity)

        self.secondary_panel.caption_box_size_slider.setValue(self.pending_caption_box_size)
        self.secondary_panel.opacity_slider.setValue(int(round(self.overlay_opacity * 100)))
        self.secondary_panel.show_raw_tokens_checkbox.setChecked(self.show_raw_tokens)
        self.secondary_panel.freeze_on_loss_checkbox.setChecked(self.freeze_on_detection_loss)
        self.secondary_panel.enable_llm_checkbox.setChecked(self.enable_llm_smoothing)
        self.secondary_panel.model_combo.setCurrentText(self.model_selection)
        self.secondary_panel.show_latency_checkbox.setChecked(self.show_latency)
        self.secondary_panel.corner_combo.setCurrentText(self.corner)
        self.secondary_panel.set_status_active(False)

        self._set_secondary_height(0, force_hide=True)
        self._refresh_window_geometry(reposition=True)

    def on_secondary_animation_value(self, value):
        self._set_secondary_height(int(value), force_hide=False)
        self._update_mask()

    def on_secondary_animation_finished(self):
        if not self.secondary_expanded:
            self._set_secondary_height(0, force_hide=True)
            self._update_mask()

    def on_caption_box_size_changed(self, value: int):
        self.pending_caption_box_size = max(PRIMARY_BOX_SIZE_MIN, min(PRIMARY_BOX_SIZE_MAX, int(value)))
        self._write_preferences()

    def on_opacity_changed(self, value: int):
        clamped = max(MIN_OPACITY_PERCENT, min(MAX_OPACITY_PERCENT, int(value)))
        self.overlay_opacity = clamped / 100.0
        self.setWindowOpacity(self.overlay_opacity)
        self._write_preferences()

    def on_show_raw_tokens_toggled(self, checked: bool):
        self.show_raw_tokens = checked
        self._write_preferences()

    def on_freeze_on_loss_toggled(self, checked: bool):
        self.freeze_on_detection_loss = checked
        self._write_preferences()

    def on_enable_llm_toggled(self, checked: bool):
        self.enable_llm_smoothing = checked
        self._write_preferences()

    def on_model_changed(self, text: str):
        self.model_selection = text
        self._write_preferences()

    def on_show_latency_toggled(self, checked: bool):
        self.show_latency = checked
        self._write_preferences()

    def on_corner_changed(self, text: str):
        self.corner = text
        self._rebuild_stack()
        self._refresh_window_geometry(reposition=True)
        self._write_preferences()

    def on_show_model_status_toggled(self, checked: bool):
        if self.preview_window is None:
            return
        self.preview_window.set_status_visible(checked)
        if checked:
            self._update_status_panel()
            self.status_timer.start()
        else:
            self.status_timer.stop()

    def _current_system_state(self):
        if not self.capture_state or not self.capture_state.get("region"):
            return "Idle"
        if self.capture_state.get("paused"):
            return "Paused"
        return "Running"

    def _update_status_panel(self):
        if self.preview_window is None or not self.preview_window._status_visible:
            return
        state = self._current_system_state()
        capture_line = "ACTIVE" if state == "Running" else ("PAUSED" if state == "Paused" else "IDLE")
        hands = int(self.last_detection.get("hands_detected", 0) or 0)
        left_conf = float(self.last_detection.get("left_conf", 0.0) or 0.0)
        right_conf = float(self.last_detection.get("right_conf", 0.0) or 0.0)
        prediction = self.last_detection.get("prediction", "")
        prediction_conf = float(self.last_detection.get("prediction_conf", 0.0) or 0.0)
        input_w = int(self.last_detection.get("input_w", 0) or 0)
        input_h = int(self.last_detection.get("input_h", 0) or 0)
        det_w = int(self.last_detection.get("det_w", 0) or 0)
        det_h = int(self.last_detection.get("det_h", 0) or 0)
        det_scale = float(self.last_detection.get("det_scale", 1.0) or 1.0)
        pad_x = int(self.last_detection.get("pad_x", 0) or 0)
        pad_y = int(self.last_detection.get("pad_y", 0) or 0)
        flip_on = bool(self.last_detection.get("flip", False))
        model_loaded = bool(self.last_detection.get("model_loaded", False))
        hand_label = self.last_detection.get("hand_label", "Unknown")
        processing_ms = float(self.last_detection.get("processing_ms", 0.0) or 0.0)
        hand_state = "Detected" if hands > 0 else "No Hands"
        fps_value = self._processing_fps
        lines = [
            f"System: {state}",
            f"Capture: {capture_line}",
            f"Hand Detection: {hand_state}",
            f"Hands Detected: {hands}",
            f"Left Hand Confidence: {left_conf:.2f}",
            f"Right Hand Confidence: {right_conf:.2f}",
            f"Prediction: {prediction}",
            f"Prediction Confidence: {prediction_conf:.2f}",
            f"Processing FPS: {fps_value:.1f}",
            f"Capture FPS: {self._capture_fps:.1f}",
            f"Input Size: {input_w}x{input_h}",
            f"Detect Size: {det_w}x{det_h}",
            f"Scale: {det_scale:.3f}",
            f"Pad: {pad_x},{pad_y}",
            f"Flip: {'On' if flip_on else 'Off'}",
            f"Model: {'Loaded' if model_loaded else 'Missing'}",
            f"Handedness: {hand_label}",
            f"Process Time: {processing_ms:.1f} ms",
        ]
        self.preview_window.set_status_text("\n".join(lines))

    def on_restart_requested(self):
        self._write_preferences()
        restart_current_process()

    def _start_region_selection(self):
        if self.selection_overlay is not None:
            self.selection_overlay.close()
            self.selection_overlay = None
        self.selection_overlay = RegionSelectionOverlay()
        self.selection_overlay.selection_confirmed.connect(self._on_region_selected)
        self.selection_overlay.selection_cancelled.connect(self._on_region_selection_cancelled)
        self.selection_overlay.show()
        self.selection_overlay.raise_()
        self.selection_overlay.activateWindow()

    def _on_region_selected(self, rect: QRect):
        if self.selection_overlay is not None:
            offset = self.selection_overlay.geometry().topLeft()
            rect = rect.translated(offset)
            self.selection_overlay.close()
            self.selection_overlay = None
        normalized = rect.normalized()
        if normalized.width() <= 0 or normalized.height() <= 0:
            self._on_region_selection_cancelled()
            return
        self._set_capture_state_from_rect(normalized)
        self._show_highlight(normalized)

    def _on_region_selection_cancelled(self):
        if self.selection_overlay is not None:
            self.selection_overlay.close()
            self.selection_overlay = None
        self.show()
        self.raise_()

    def _show_highlight(self, rect: QRect):
        if self.highlight_overlay is not None:
            self.highlight_overlay.close()
        self.highlight_overlay = HighlightOverlay(rect)
        self.highlight_overlay.show()
        self.highlight_overlay.raise_()
        QTimer.singleShot(HIGHLIGHT_DURATION_MS, self._finish_capture_start)

    def _finish_capture_start(self):
        if self.highlight_overlay is not None:
            self.highlight_overlay.close()
            self.highlight_overlay = None
        self.show()
        self.raise_()
        self._start_capture()

    def _set_capture_state_from_rect(self, rect: QRect):
        rect = self._rect_to_physical(rect)
        self.capture_state = {
            "region": {
                "x": int(rect.x()),
                "y": int(rect.y()),
                "width": int(rect.width()),
                "height": int(rect.height()),
            },
            "paused": False,
        }
        self.first_launch_hint = False
        if self.preview_window is not None:
            self.preview_window.set_region_info(self.capture_state.get("region"), self.first_launch_hint)

    def _rect_to_physical(self, rect: QRect):
        screen = QGuiApplication.screenAt(rect.center())
        if screen is None:
            screen = QGuiApplication.primaryScreen()
        if screen is None:
            return rect
        scale = screen.devicePixelRatio()
        if scale <= 0:
            return rect
        return QRect(
            int(rect.x() * scale),
            int(rect.y() * scale),
            max(1, int(rect.width() * scale)),
            max(1, int(rect.height() * scale)),
        )

    def _start_capture(self):
        if not self.capture_state or not self.capture_state.get("region"):
            return
        self._stop_capture_thread()
        self.capture_state["paused"] = False
        self.secondary_panel.set_playing(True)
        self.secondary_panel.set_status_active(True)
        self._ensure_preview_window()
        if self.preview_window is not None:
            self.preview_window.set_capture_state("LIVE")
            self.preview_window.set_region_info(self.capture_state.get("region"), self.first_launch_hint)
        self.capture_thread = ScreenCaptureThread(self.capture_state["region"])
        self.capture_thread.frame_captured.connect(self._on_frame_captured)
        self.capture_thread.start()
        if not self._preview_timer.isActive():
            self._preview_timer.start()

        if self.hand_worker is None:
            self.hand_worker = HandTrackingWorker(flip_horizontal=CAPTURE_FLIP_HORIZONTAL)
            self.hand_worker.status_updated.connect(self._on_detection_status)
            self.hand_worker.frame_processed.connect(self._on_processed_frame)
            self.hand_worker.fps_updated.connect(self._on_processing_fps)
            self.hand_worker.prediction_updated.connect(self._on_prediction_text)
            self.hand_worker.start()

    def _stop_capture_thread(self):
        if self.capture_thread is None:
            return
        self.capture_thread.stop()
        self.capture_thread = None
        if self._preview_timer.isActive():
            self._preview_timer.stop()

    def _ensure_preview_window(self):
        if self.preview_window is None:
            self.preview_window = PreviewWindow()
        self.preview_window.set_status_visible(self.secondary_panel.freeze_on_loss_checkbox.isChecked())
        self.preview_window.set_capture_state(self._current_system_state())
        self.preview_window.set_region_info(self.capture_state.get("region"), self.first_launch_hint)
        if self.preview_window._status_visible:
            self._update_status_panel()
            self.status_timer.start()
        self.preview_window.show()
        self.preview_window.raise_()

    def _on_frame_captured(self, frame):
        process_frame(frame)

    def _handle_frame(self, frame):
        if not self.capture_state:
            return
        now = time.perf_counter()
        if self._capture_frame_time is not None:
            delta = now - self._capture_frame_time
            if delta > 1e-6:
                instant = 1.0 / delta
                self._capture_fps = (self._capture_fps * 0.85) + (instant * 0.15)
        self._capture_frame_time = now
        self._latest_frame = frame
        self._latest_frame_time = now
        if self.hand_worker is not None:
            self.hand_worker.submit(frame)

    def _on_processed_frame(self, frame):
        self._latest_processed_frame = frame
        self._latest_processed_time = time.perf_counter()

    def _on_detection_status(self, status: dict):
        if status:
            self.last_detection = status

    def _on_processing_fps(self, fps: float):
        self._processing_fps = float(fps or 0.0)

    def _on_prediction_text(self, text: str):
        clean = (text or "").strip()
        if not clean:
            return
        if clean == self._last_prediction:
            return
        self._last_prediction = clean
        self.set_caption_text(clean)

    def _update_preview_frame(self):
        if self.preview_window is None:
            return
        if not self.capture_state or self.capture_state.get("paused"):
            return

        frame = None
        using_processed = False
        now = time.perf_counter()
        if self._latest_processed_frame is not None and self._latest_processed_time is not None:
            if now - self._latest_processed_time < 0.35:
                frame = self._latest_processed_frame
                using_processed = True

        if frame is None:
            frame = self._latest_frame

        image = _frame_to_qimage(frame)
        if image is None:
            return
        if CAPTURE_FLIP_HORIZONTAL and not using_processed:
            image = image.mirrored(True, False)
        self.preview_window.update_frame(image)

    def on_crop_clicked(self):
        self.hide()
        QTimer.singleShot(50, self._start_region_selection)

    def on_fullscreen_capture(self):
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        rect = screen.geometry()
        self._set_capture_state_from_rect(rect)
        self._start_capture()

    def on_play_pause_toggled(self, _is_playing: bool):
        if self.capture_state is None:
            self.capture_state = {"region": None, "paused": not _is_playing}
        else:
            self.capture_state["paused"] = not _is_playing
        self.secondary_panel.set_status_active(bool(_is_playing))
        if self.preview_window is not None:
            self.preview_window.set_capture_state("LIVE" if _is_playing else "PAUSED")

    def on_clear_clicked(self):
        self.secondary_panel.set_status_active(False)
        self.capture_state = {"region": None, "paused": False}
        if self.preview_window is not None:
            self.preview_window.set_capture_state("IDLE")
            self.preview_window.set_region_info(None, self.first_launch_hint)
        stop_capture()

    def on_reset_preferences_requested(self):
        defaults = _read_json(DEFAULT_SETTINGS_PATH)
        normalized_defaults = _sanitize_settings(defaults if defaults is not None else DEFAULT_SETTINGS)
        save_user_preferences(normalized_defaults)
        restart_current_process()

    def closeEvent(self, event):
        self._stop_capture_thread()
        if self.preview_window is not None:
            self.preview_window.close()
            self.preview_window = None
        if self.selection_overlay is not None:
            self.selection_overlay.close()
            self.selection_overlay = None
        if self.highlight_overlay is not None:
            self.highlight_overlay.close()
            self.highlight_overlay = None
        if self.hand_worker is not None:
            self.hand_worker.stop()
            self.hand_worker = None
        set_frame_dispatcher(None)
        super().closeEvent(event)

    def set_caption_text(self, text: str):
        self.caption_text = text or LABEL_DEFAULT_TEXT
        self.primary_panel.set_caption_text(self.caption_text)
        self._refresh_window_geometry(reposition=True)

    def toggle_secondary_panel(self):
        if ENABLE_COLLAPSE_ANIMATION and self.secondary_animation.state() == QAbstractAnimation.Running:
            return

        self.secondary_expanded = not self.secondary_expanded
        self.primary_panel.set_expanded_icon(self.secondary_expanded)

        target = SECONDARY_EXPANDED_HEIGHT if self.secondary_expanded else 0

        if not ENABLE_COLLAPSE_ANIMATION:
            self.secondary_animation.stop()
            self._set_secondary_height(target, force_hide=not self.secondary_expanded)
            self._update_mask()
            return

        self.secondary_animation.stop()
        self.secondary_animation.setStartValue(self.secondary_current_height)
        self.secondary_animation.setEndValue(target)
        self.secondary_animation.start()
