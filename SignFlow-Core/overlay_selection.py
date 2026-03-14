from PyQt5.QtCore import QPoint, QRect, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QFontMetrics, QGuiApplication, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import QWidget

from overlay_constants import (
    FONT_FAMILY,
    HIGHLIGHT_BORDER_WIDTH,
    SELECTION_BORDER_WIDTH,
    SELECTION_DIM_ALPHA,
    SELECTION_INSTRUCTION_TEXT,
    SELECTION_TEXT_FONT_SIZE,
)
from overlay_utils import _set_window_excluded_from_capture


class RegionSelectionOverlay(QWidget):
    selection_confirmed = pyqtSignal(QRect)
    selection_cancelled = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._dragging = False
        self._origin = QPoint()
        self._current = QPoint()
        self._selection_rect = None

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setCursor(Qt.CrossCursor)

        screen = QGuiApplication.primaryScreen()
        geometry = screen.virtualGeometry() if screen is not None else QRect(0, 0, 800, 600)
        self.setGeometry(geometry)

    def showEvent(self, event):
        super().showEvent(event)
        self.raise_()
        self.activateWindow()
        self.grabKeyboard()
        _set_window_excluded_from_capture(self)

    def closeEvent(self, event):
        self.releaseKeyboard()
        super().closeEvent(event)

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        self._dragging = True
        self._origin = event.pos()
        self._current = event.pos()
        self._selection_rect = QRect(self._origin, self._current).normalized()
        self.update()

    def mouseMoveEvent(self, event):
        if not self._dragging:
            return
        self._current = event.pos()
        self._selection_rect = QRect(self._origin, self._current).normalized()
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        self._dragging = False
        self._current = event.pos()
        self._selection_rect = QRect(self._origin, self._current).normalized()
        self.update()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            if self._has_valid_selection():
                self.selection_confirmed.emit(self._selection_rect)
            else:
                self.selection_cancelled.emit()
            self.close()
            return
        if event.key() == Qt.Key_Escape:
            self.selection_cancelled.emit()
            self.close()
            return
        super().keyPressEvent(event)

    def _has_valid_selection(self):
        if self._selection_rect is None:
            return False
        return self._selection_rect.width() > 0 and self._selection_rect.height() > 0

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        full_rect = self.rect()
        dim_color = QColor(0, 0, 0, SELECTION_DIM_ALPHA)

        if self._has_valid_selection():
            path = QPainterPath()
            path.addRect(QRectF(full_rect))
            path.addRect(QRectF(self._selection_rect))
            path.setFillRule(Qt.OddEvenFill)
            painter.fillPath(path, dim_color)
            pen = QPen(QColor(255, 255, 255, 230), SELECTION_BORDER_WIDTH)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(self._selection_rect)
        else:
            painter.fillRect(full_rect, dim_color)

        text = SELECTION_INSTRUCTION_TEXT
        suffix = " Press ENTER or SPACE to confirm."
        if text.endswith(suffix):
            base_text = text[: -len(suffix)]
            suffix_text = suffix
        else:
            base_text = text
            suffix_text = ""

        base_font = QFont(FONT_FAMILY, SELECTION_TEXT_FONT_SIZE)
        base_font.setWeight(QFont.Medium)
        suffix_font = QFont(FONT_FAMILY, SELECTION_TEXT_FONT_SIZE)
        suffix_font.setWeight(QFont.Medium)

        base_metrics = QFontMetrics(base_font)
        suffix_metrics = QFontMetrics(suffix_font)
        base_width = base_metrics.horizontalAdvance(base_text)
        suffix_width = suffix_metrics.horizontalAdvance(suffix_text)
        text_width = base_width + suffix_width
        text_height = max(base_metrics.height(), suffix_metrics.height())

        padding_h = 18
        padding_v = 8
        label_width = text_width + (padding_h * 2)
        label_height = text_height + (padding_v * 2)
        label_x = max(20, int((self.width() - label_width) / 2))
        label_rect = QRectF(label_x, 0, label_width, label_height)

        radius = 10
        label_path = QPainterPath()
        label_path.moveTo(label_rect.left(), label_rect.top())
        label_path.lineTo(label_rect.right(), label_rect.top())
        label_path.lineTo(label_rect.right(), label_rect.bottom() - radius)
        label_path.quadTo(label_rect.right(), label_rect.bottom(), label_rect.right() - radius, label_rect.bottom())
        label_path.lineTo(label_rect.left() + radius, label_rect.bottom())
        label_path.quadTo(label_rect.left(), label_rect.bottom(), label_rect.left(), label_rect.bottom() - radius)
        label_path.lineTo(label_rect.left(), label_rect.top())

        painter.setPen(QPen(QColor(255, 255, 255, 30), 1))
        painter.setBrush(QColor(28, 28, 30, 230))
        painter.drawPath(label_path)

        text_rect = QRectF(label_rect)
        painter.setPen(QColor(255, 255, 255, 235))
        baseline = label_rect.top() + (label_height + base_metrics.ascent() - base_metrics.descent()) / 2.0
        start_x = label_rect.left() + (label_width - text_width) / 2.0
        painter.setFont(base_font)
        painter.drawText(QPoint(int(start_x), int(baseline)), base_text)
        if suffix_text:
            painter.setFont(suffix_font)
            painter.drawText(QPoint(int(start_x + base_width), int(baseline)), suffix_text)


class HighlightOverlay(QWidget):
    def __init__(self, rect: QRect):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setGeometry(rect)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        pen = QPen(QColor(255, 255, 255, 235), HIGHLIGHT_BORDER_WIDTH)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        rect = self.rect().adjusted(0, 0, -HIGHLIGHT_BORDER_WIDTH, -HIGHLIGHT_BORDER_WIDTH)
        painter.drawRect(rect)
