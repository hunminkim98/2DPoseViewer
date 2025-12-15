"""
PoseCanvas - 2D Pose visualization canvas widget.
"""

import math
from typing import List, Optional, Dict, Tuple

from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QFontMetrics

from .models import Keypoint, Person, FrameData
from .skeletons import HALPE_26
from .constants import SKELETON_MODELS, BODY_PART_COLORS, get_skeleton_connections, get_keypoint_ids


class PoseCanvas(QWidget):
    """포즈 데이터를 시각화하는 캔버스 위젯"""
    
    # 시그널: 사람 클릭 시 발생
    person_clicked = Signal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame_data: Optional[FrameData] = None
        self.confidence_threshold: float = 0.3
        self.show_keypoints: bool = True
        self.show_skeleton: bool = True
        self.show_labels: bool = False
        self.show_bbox: bool = True
        self.selected_person: int = -1
        
        # 시각화 옵션
        self.keypoint_size: int = 6
        self.keypoint_opacity: int = 255
        self.skeleton_width: int = 3
        self.skeleton_opacity: int = 255
        self.label_font_size: int = 8
        self.label_opacity: int = 255
        self.bbox_width: int = 2
        self.bbox_opacity: int = 200
        
        # 마우스 인터랙션
        self.setMouseTracking(True)
        self.hovered_person: int = -1
        self.person_bboxes: List[tuple] = []
        
        # 스켈레톤 모델 설정
        self.current_model_name = "HALPE_26"
        self.skeleton_model = HALPE_26
        self.connections = get_skeleton_connections(self.skeleton_model, self.current_model_name)
        self.keypoint_names = get_keypoint_ids(self.skeleton_model)
        
        # 스타일 설정
        self.setMinimumSize(800, 600)
        self.setStyleSheet("background-color: #1a1a2e;")
    
    def _get_person_color(self, person_idx: int) -> QColor:
        """사람 인덱스에 따라 고유한 색상 생성 (황금비 사용)"""
        golden_ratio = 0.618033988749895
        hue = (person_idx * golden_ratio) % 1.0
        
        saturation = 0.7
        lightness = 0.6
        
        if saturation == 0:
            r = g = b = lightness
        else:
            def hue_to_rgb(p, q, t):
                if t < 0: t += 1
                if t > 1: t -= 1
                if t < 1/6: return p + (q - p) * 6 * t
                if t < 1/2: return q
                if t < 2/3: return p + (q - p) * (2/3 - t) * 6
                return p
            
            q = lightness * (1 + saturation) if lightness < 0.5 else lightness + saturation - lightness * saturation
            p = 2 * lightness - q
            r = hue_to_rgb(p, q, hue + 1/3)
            g = hue_to_rgb(p, q, hue)
            b = hue_to_rgb(p, q, hue - 1/3)
        
        return QColor(int(r * 255), int(g * 255), int(b * 255))
        
    def set_skeleton_model(self, model_name: str):
        """스켈레톤 모델 변경"""
        if model_name in SKELETON_MODELS:
            self.current_model_name = model_name
            self.skeleton_model = SKELETON_MODELS[model_name]
            self.connections = get_skeleton_connections(self.skeleton_model, model_name)
            self.keypoint_names = get_keypoint_ids(self.skeleton_model)
            self.update()
        
    def set_frame_data(self, data: FrameData):
        """프레임 데이터 설정"""
        self.frame_data = data
        self.update()
        
    def set_confidence_threshold(self, threshold: float):
        """confidence 임계값 설정"""
        self.confidence_threshold = threshold
        self.update()
        
    def set_show_keypoints(self, show: bool):
        self.show_keypoints = show
        self.update()
        
    def set_show_skeleton(self, show: bool):
        self.show_skeleton = show
        self.update()
        
    def set_show_labels(self, show: bool):
        self.show_labels = show
        self.update()
        
    def set_selected_person(self, person_idx: int):
        self.selected_person = person_idx
        self.update()
    
    def set_show_bbox(self, show: bool):
        self.show_bbox = show
        self.update()
    
    def set_keypoint_size(self, size: int):
        self.keypoint_size = size
        self.update()
    
    def set_keypoint_opacity(self, opacity: int):
        self.keypoint_opacity = opacity
        self.update()
    
    def set_skeleton_width(self, width: int):
        self.skeleton_width = width
        self.update()
    
    def set_skeleton_opacity(self, opacity: int):
        self.skeleton_opacity = opacity
        self.update()
    
    def set_label_font_size(self, size: int):
        self.label_font_size = size
        self.update()
    
    def set_label_opacity(self, opacity: int):
        self.label_opacity = opacity
        self.update()
    
    def set_bbox_width(self, width: int):
        self.bbox_width = width
        self.update()
    
    def set_bbox_opacity(self, opacity: int):
        self.bbox_opacity = opacity
        self.update()
    
    def _is_valid_confidence(self, conf: float) -> bool:
        """confidence 값이 유효한지 확인 (NaN, inf 체크)"""
        return not (math.isnan(conf) or math.isinf(conf))
    
    def _is_valid_coord(self, x: float, y: float) -> bool:
        """좌표 값이 유효한지 확인 (NaN, inf 체크)"""
        return not (math.isnan(x) or math.isinf(x) or math.isnan(y) or math.isinf(y))
    
    def _get_person_at(self, pos) -> int:
        """주어진 위치에 있는 사람의 인덱스 반환 (-1: 없음)"""
        x, y = pos.x(), pos.y()
        for person_idx, bx, by, bw, bh in self.person_bboxes:
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return person_idx
        return -1
    
    def mouseMoveEvent(self, event):
        """마우스 이동 시 호버 효과"""
        new_hovered = self._get_person_at(event.position().toPoint())
        if new_hovered != self.hovered_person:
            self.hovered_person = new_hovered
            if new_hovered >= 0:
                self.setCursor(Qt.CursorShape.PointingHandCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
            self.update()
    
    def mousePressEvent(self, event):
        """마우스 클릭 시 사람 선택"""
        if event.button() == Qt.MouseButton.LeftButton:
            clicked_person = self._get_person_at(event.position().toPoint())
            if clicked_person >= 0:
                self.person_clicked.emit(clicked_person)
        elif event.button() == Qt.MouseButton.RightButton:
            self.person_clicked.emit(-1)
    
    def leaveEvent(self, event):
        """마우스가 캔버스를 떠날 때"""
        if self.hovered_person >= 0:
            self.hovered_person = -1
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.update()
    
    def paintEvent(self, event):
        """캔버스 렌더링"""
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            painter.fillRect(self.rect(), QColor("#1a1a2e"))
            
            self.person_bboxes = []
            
            if not self.frame_data:
                painter.setPen(QColor("#ffffff"))
                painter.setFont(QFont("Segoe UI", 14))
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 
                               "JSON 폴더를 불러와주세요")
                return
            
            for idx, person in enumerate(self.frame_data.people):
                if self.selected_person >= 0 and idx != self.selected_person:
                    continue
                if self.show_bbox:
                    self._draw_bbox(painter, person, idx)
                self._draw_person(painter, person, idx)
        finally:
            painter.end()
            
    def _draw_person(self, painter: QPainter, person: Person, person_idx: int):
        """한 사람의 포즈 그리기"""
        keypoints = person.keypoints
        
        if self.show_skeleton:
            for parent_id, child_id, part in self.connections:
                if parent_id >= len(keypoints) or child_id >= len(keypoints):
                    continue
                    
                kp1 = keypoints[parent_id]
                kp2 = keypoints[child_id]
                
                if not self._is_valid_confidence(kp1.confidence) or \
                   not self._is_valid_confidence(kp2.confidence):
                    continue
                if kp1.confidence < self.confidence_threshold or \
                   kp2.confidence < self.confidence_threshold:
                    continue
                    
                color = QColor(BODY_PART_COLORS.get(part, BODY_PART_COLORS['center']))
                
                base_alpha = min(self.skeleton_opacity, int(min(kp1.confidence, kp2.confidence) * 255))
                alpha = max(0, min(255, base_alpha))
                color.setAlpha(alpha)
                
                if not self._is_valid_coord(kp1.x, kp1.y) or not self._is_valid_coord(kp2.x, kp2.y):
                    continue
                
                pen = QPen(color, self.skeleton_width)
                painter.setPen(pen)
                painter.drawLine(int(kp1.x), int(kp1.y), int(kp2.x), int(kp2.y))
        
        if self.show_keypoints:
            for kp_idx, kp in enumerate(keypoints):
                if not self._is_valid_confidence(kp.confidence):
                    continue
                if kp.confidence < self.confidence_threshold:
                    continue
                    
                radius = self.keypoint_size
                color = QColor("#ffffff")
                base_alpha = min(self.keypoint_opacity, int(kp.confidence * 255))
                alpha = max(0, min(255, base_alpha))
                color.setAlpha(alpha)
                
                if not self._is_valid_coord(kp.x, kp.y):
                    continue
                
                painter.setPen(QPen(color, 2))
                painter.setBrush(QBrush(color))
                painter.drawEllipse(int(kp.x) - radius, int(kp.y) - radius, 
                                   radius * 2, radius * 2)
                
                if self.show_labels and kp_idx in self.keypoint_names:
                    label_color = QColor("#ffffff")
                    label_color.setAlpha(self.label_opacity)
                    painter.setPen(label_color)
                    painter.setFont(QFont("Segoe UI", self.label_font_size))
                    painter.drawText(int(kp.x) + radius + 2, int(kp.y), 
                                   self.keypoint_names[kp_idx])
    
    def _draw_bbox(self, painter: QPainter, person: Person, person_idx: int):
        """사람 주변에 bounding box와 ID 그리기"""
        keypoints = person.keypoints
        
        valid_points = []
        for kp in keypoints:
            if self._is_valid_confidence(kp.confidence) and kp.confidence >= self.confidence_threshold:
                if kp.x > 0 and kp.y > 0:
                    valid_points.append((kp.x, kp.y))
        
        if len(valid_points) < 2:
            return
        
        xs = [p[0] for p in valid_points]
        ys = [p[1] for p in valid_points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        padding = 15
        x_min -= padding
        y_min -= padding
        x_max += padding
        y_max += padding
        
        width = x_max - x_min
        height = y_max - y_min
        
        self.person_bboxes.append((person_idx, int(x_min), int(y_min), int(width), int(height)))
        
        color = self._get_person_color(person_idx)
        color.setAlpha(self.bbox_opacity)
        
        if self.hovered_person == person_idx:
            hover_color = QColor(color)
            hover_color.setAlpha(40)
            painter.fillRect(int(x_min), int(y_min), int(width), int(height), hover_color)
        
        line_width = self.bbox_width + (1 if self.hovered_person == person_idx else 0)
        pen = QPen(color, line_width, Qt.PenStyle.DashLine if self.hovered_person != person_idx else Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(int(x_min), int(y_min), int(width), int(height))
        
        label_text = f"Person {person_idx}"
        font = QFont("Segoe UI", 10, QFont.Weight.Bold)
        painter.setFont(font)
        
        fm = QFontMetrics(font)
        text_width = fm.horizontalAdvance(label_text) + 10
        text_height = fm.height() + 4
        
        label_x = int(x_min)
        label_y = int(y_min) - text_height - 2
        if label_y < 0:
            label_y = int(y_min) + 2
        
        bg_color = QColor(color)
        bg_color.setAlpha(200)
        painter.fillRect(label_x, label_y, text_width, text_height, bg_color)
        
        painter.setPen(QColor("#1a1a2e"))
        painter.drawText(label_x + 5, label_y + text_height - 5, label_text)
