"""
Multi View Mode - 여러 카메라 뷰를 동시에 표시하는 기능
"""

import os
import json
import glob
from typing import List, Optional

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QFileDialog, QPushButton, QLabel, QSlider, QFrame, QScrollArea,
    QButtonGroup, QSizePolicy, QStatusBar
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont, QColor
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import math

from .models import Keypoint, Person, FrameData
from .canvas import PoseCanvas
from .constants import SKELETON_MODELS


class MiniPlaybackBar(QWidget):
    """미니 재생 컨트롤 바"""
    
    frame_changed = Signal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.total_frames = 0
        self.current_frame = 0
        self.is_playing = False
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)
        layout.setSpacing(5)
        
        # 재생/일시정지 버튼
        self.play_btn = QPushButton("▶")
        self.play_btn.setFixedSize(28, 24)
        self.play_btn.clicked.connect(self._toggle_playback)
        self.play_btn.setStyleSheet("""
            QPushButton {
                background-color: #4ECDC4;
                color: #1a1a2e;
                border: none;
                border-radius: 4px;
                font-size: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45b7aa;
            }
        """)
        layout.addWidget(self.play_btn)
        
        # 프레임 슬라이더
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #2d2d44;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #4ECDC4;
                width: 12px;
                margin: -3px 0;
                border-radius: 6px;
            }
            QSlider::sub-page:horizontal {
                background: #4ECDC4;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.slider, 1)
        
        # 프레임 레이블
        self.frame_label = QLabel("0/0")
        self.frame_label.setFixedWidth(60)
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_label.setStyleSheet("color: #e0e0e0; font-size: 10px;")
        layout.addWidget(self.frame_label)
        
    def _toggle_playback(self):
        self.is_playing = not self.is_playing
        self.play_btn.setText("⏸" if self.is_playing else "▶")
        
    def _on_slider_changed(self, value):
        if value != self.current_frame:
            self.current_frame = value
            self._update_label()
            self.frame_changed.emit(value)
            
    def set_total_frames(self, total: int):
        self.total_frames = total
        self.slider.setMaximum(max(0, total - 1))
        self._update_label()
        
    def set_current_frame(self, frame: int):
        self.current_frame = frame
        self.slider.blockSignals(True)
        self.slider.setValue(frame)
        self.slider.blockSignals(False)
        self._update_label()
        
    def _update_label(self):
        self.frame_label.setText(f"{self.current_frame + 1}/{self.total_frames}")
        
    def stop_playback(self):
        if self.is_playing:
            self.is_playing = False
            self.play_btn.setText("▶")


class ViewPanel(QFrame):
    """개별 뷰 패널 - 하나의 카메라/폴더용"""
    
    person_selected = Signal(int)  # 인물 선택 시그널 추가
    
    def __init__(self, view_id: int, parent=None):
        super().__init__(parent)
        self.view_id = view_id
        self.frame_files: List[str] = []
        self.all_frames_data: List[FrameData] = []
        self.current_frame: int = 0
        self.folder_path: Optional[str] = None
        
        # 재생 타이머
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._on_timer_tick)
        self.fps = 30
        
        # 필터링 관련 데이터
        self.filtered_frames_data: Optional[List[FrameData]] = None
        self.is_filtered: bool = False
        
        self._setup_ui()
        self._connect_signals()
        
    def _setup_ui(self):
        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        self.setStyleSheet("""
            ViewPanel {
                background-color: #1a1a2e;
                border: 1px solid #3d3d5c;
                border-radius: 8px;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(3)
        
        # 상단 헤더 (타이틀 + 폴더 버튼)
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(5, 2, 5, 2)
        header_layout.setSpacing(5)
        
        self.title_label = QLabel(f"📷 View {self.view_id + 1}")
        self.title_label.setStyleSheet("""
            color: #4ECDC4;
            font-size: 11px;
            font-weight: bold;
        """)
        header_layout.addWidget(self.title_label)
        
        self.person_label = QLabel("(All People)")
        self.person_label.setStyleSheet("color: #888; font-size: 10px;")
        header_layout.addWidget(self.person_label)
        
        header_layout.addStretch()
        
        self.folder_btn = QPushButton("📁")
        self.folder_btn.setFixedSize(28, 24)
        self.folder_btn.setToolTip("폴더 열기")
        self.folder_btn.clicked.connect(self._open_folder)
        self.folder_btn.setStyleSheet("""
            QPushButton {
                background-color: #2d2d44;
                color: #e0e0e0;
                border: 1px solid #3d3d5c;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #4ECDC4;
                color: #1a1a2e;
            }
        """)
        header_layout.addWidget(self.folder_btn)
        
        layout.addWidget(header)
        
        # 캔버스
        self.canvas = PoseCanvas()
        self.canvas.setMinimumSize(200, 150)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.canvas, 1)
        
        # 미니 플레이백 바
        self.playback_bar = MiniPlaybackBar()
        layout.addWidget(self.playback_bar)
        
    def _connect_signals(self):
        self.playback_bar.frame_changed.connect(self._go_to_frame)
        self.canvas.person_clicked.connect(self._on_canvas_person_clicked)
        
    def _on_canvas_person_clicked(self, person_idx: int):
        """이 패널의 인물 선택 처리 (독립적)"""
        self.canvas.set_selected_person(person_idx)
        if person_idx >= 0:
            self.person_label.setText(f"(Person {person_idx})")
            self.person_label.setStyleSheet("color: #FFD93D; font-size: 10px; font-weight: bold;")
        else:
            self.person_label.setText("(All People)")
            self.person_label.setStyleSheet("color: #888; font-size: 10px;")
        self.person_selected.emit(person_idx)
        
    def _open_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, f"View {self.view_id + 1} - JSON 폴더 선택", "",
            QFileDialog.Option.ShowDirsOnly
        )
        if folder:
            self._load_folder(folder)
            
    def _load_folder(self, folder: str):
        pattern = os.path.join(folder, "*.json")
        files = sorted(glob.glob(pattern))
        
        if not files:
            return
            
        self.folder_path = folder
        self.frame_files = files
        self.current_frame = 0
        self.all_frames_data = []
        
        # 전체 바운딩 박스 계산을 위한 변수
        all_x = []
        all_y = []
        
        for idx, file_path in enumerate(files):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                frame_data = self._parse_frame_data(data, idx)
                self.all_frames_data.append(frame_data)
                
                # 전체 바운딩 박스 계산을 위해 유효한 좌표 수집
                for person in frame_data.people:
                    for kp in person.keypoints:
                        if kp.confidence >= 0.3 and kp.x > 0 and kp.y > 0:
                            all_x.append(kp.x)
                            all_y.append(kp.y)
            except Exception:
                self.all_frames_data.append(FrameData(frame_number=idx, people=[]))
        
        # 전체 프레임 기반 고정 바운딩 박스 설정
        if len(all_x) >= 2 and len(all_y) >= 2:
            self.canvas.set_fixed_bounds(min(all_x), max(all_x), min(all_y), max(all_y))
        else:
            self.canvas.clear_fixed_bounds()
        
        # 폴더명을 타이틀에 표시
        folder_name = os.path.basename(folder)
        self.title_label.setText(f"📷 {folder_name}")
        self.title_label.setToolTip(folder)
        
        self.playback_bar.set_total_frames(len(files))
        self._load_current_frame()
        
    def _parse_frame_data(self, data: dict, frame_num: int) -> FrameData:
        people = []
        
        for person_data in data.get('people', []):
            keypoints_raw = person_data.get('pose_keypoints_2d', [])
            
            keypoints = []
            for i in range(0, len(keypoints_raw), 3):
                if i + 2 < len(keypoints_raw):
                    kp = Keypoint(
                        x=keypoints_raw[i],
                        y=keypoints_raw[i + 1],
                        confidence=keypoints_raw[i + 2]
                    )
                    keypoints.append(kp)
                    
            person = Person(
                person_id=person_data.get('person_id', [-1])[0] if person_data.get('person_id') else -1,
                keypoints=keypoints
            )
            people.append(person)
            
        return FrameData(frame_number=frame_num, people=people)
        
    def _load_current_frame(self):
        if not self.all_frames_data or self.current_frame >= len(self.all_frames_data):
            return
        
        # 필터링 여부에 따라 데이터 선택
        if self.is_filtered and self.filtered_frames_data:
            frame_data = self.filtered_frames_data[self.current_frame]
        else:
            frame_data = self.all_frames_data[self.current_frame]
            
        self.canvas.set_frame_data(frame_data)
        self.playback_bar.set_current_frame(self.current_frame)
        
    def apply_filter(self, filter_type: str, params: dict):
        """필터 적용 (app.py의 로직과 동일)"""
        if not self.all_frames_data:
            return
            
        try:
            max_people = max(len(fd.people) for fd in self.all_frames_data)
            max_keypoints = max(
                max((len(p.keypoints) for p in fd.people), default=0)
                for fd in self.all_frames_data
            )
            
            if max_people == 0 or max_keypoints == 0:
                return
            
            n_frames = len(self.all_frames_data)
            data_array = np.full((n_frames, max_people, max_keypoints, 3), np.nan)
            
            for f_idx, frame_data in enumerate(self.all_frames_data):
                for p_idx, person in enumerate(frame_data.people):
                    for k_idx, kp in enumerate(person.keypoints):
                        data_array[f_idx, p_idx, k_idx, 0] = kp.x
                        data_array[f_idx, p_idx, k_idx, 1] = kp.y
                        data_array[f_idx, p_idx, k_idx, 2] = kp.confidence
            
            filtered_array = data_array.copy()
            
            for p_idx in range(max_people):
                for k_idx in range(max_keypoints):
                    for coord_idx in range(2):
                        col = data_array[:, p_idx, k_idx, coord_idx].copy()
                        valid_mask = ~np.isnan(col) & (col > 0)
                        if np.sum(valid_mask) < 5:
                            continue
                        
                        if filter_type == "butterworth":
                            filtered_col = self._butterworth_filter(
                                col, valid_mask,
                                params.get('order', 4),
                                params.get('cutoff', 6),
                                frame_rate=30
                            )
                        elif filter_type == "gaussian":
                            filtered_col = self._gaussian_filter(
                                col, valid_mask,
                                params.get('sigma', 3)
                            )
                        elif filter_type == "median":
                            filtered_col = self._median_filter(
                                col, valid_mask,
                                params.get('kernel_size', 5)
                            )
                        else:
                            continue
                        
                        filtered_array[:, p_idx, k_idx, coord_idx] = filtered_col
            
            self.filtered_frames_data = []
            for f_idx in range(n_frames):
                people = []
                orig_frame = self.all_frames_data[f_idx]
                for p_idx, orig_person in enumerate(orig_frame.people):
                    keypoints = []
                    for k_idx, orig_kp in enumerate(orig_person.keypoints):
                        x_val = filtered_array[f_idx, p_idx, k_idx, 0]
                        y_val = filtered_array[f_idx, p_idx, k_idx, 1]
                        conf_val = filtered_array[f_idx, p_idx, k_idx, 2]
                        
                        kp = Keypoint(
                            x=float(x_val) if not np.isnan(x_val) else orig_kp.x,
                            y=float(y_val) if not np.isnan(y_val) else orig_kp.y,
                            confidence=float(conf_val) if not np.isnan(conf_val) else orig_kp.confidence
                        )
                        keypoints.append(kp)
                    people.append(Person(person_id=orig_person.person_id, keypoints=keypoints))
                self.filtered_frames_data.append(FrameData(frame_number=f_idx, people=people))
            
            self.is_filtered = True
            self._load_current_frame()
            
        except Exception as e:
            print(f"Filter error in ViewPanel {self.view_id}: {e}")

    def _butterworth_filter(self, col: np.ndarray, valid_mask: np.ndarray, 
                            order: int, cutoff: int, frame_rate: int) -> np.ndarray:
        result = col.copy()
        b, a = signal.butter(order // 2, cutoff / (frame_rate / 2), 'low', analog=False)
        padlen = 3 * max(len(a), len(b))
        
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            return result
        
        gaps = np.where(np.diff(valid_indices) > 1)[0] + 1
        sequences = np.split(valid_indices, gaps)
        
        for seq in sequences:
            if len(seq) > padlen:
                result[seq] = signal.filtfilt(b, a, col[seq])
        
        return result
    
    def _gaussian_filter(self, col: np.ndarray, valid_mask: np.ndarray, sigma: int) -> np.ndarray:
        result = col.copy()
        
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            return result
        
        gaps = np.where(np.diff(valid_indices) > 1)[0] + 1
        sequences = np.split(valid_indices, gaps)
        
        for seq in sequences:
            if len(seq) > sigma * 2:
                result[seq] = gaussian_filter1d(col[seq], sigma)
        
        return result
    
    def _median_filter(self, col: np.ndarray, valid_mask: np.ndarray, kernel_size: int) -> np.ndarray:
        result = col.copy()
        
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            return result
        
        gaps = np.where(np.diff(valid_indices) > 1)[0] + 1
        sequences = np.split(valid_indices, gaps)
        
        for seq in sequences:
            if len(seq) >= kernel_size:
                result[seq] = signal.medfilt(col[seq], kernel_size)
        
        return result

    def revert_filter(self):
        """필터 해제"""
        self.filtered_frames_data = None
        self.is_filtered = False
        self._load_current_frame()
        
    def _go_to_frame(self, frame: int):
        if not self.frame_files:
            return
        self.current_frame = max(0, min(frame, len(self.frame_files) - 1))
        self._load_current_frame()
        
    def _next_frame(self):
        if self.current_frame < len(self.frame_files) - 1:
            self._go_to_frame(self.current_frame + 1)
        else:
            self._go_to_frame(0)  # 루프
            
    def _on_timer_tick(self):
        if self.playback_bar.is_playing:
            self._next_frame()
        else:
            self.play_timer.stop()
            
    def update_playback(self):
        """재생 상태 업데이트 (외부에서 호출)"""
        if self.playback_bar.is_playing and not self.play_timer.isActive():
            self.play_timer.start(int(1000 / self.fps))
        elif not self.playback_bar.is_playing and self.play_timer.isActive():
            self.play_timer.stop()


class MultiViewContainer(QWidget):
    """여러 ViewPanel을 그리드로 배치하는 컨테이너"""
    
    person_selected = Signal(int)  # 상위(app.py)로 전달할 시그널
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.view_panels: List[ViewPanel] = []
        self.current_layout = (1, 1)  # (rows, cols)
        
        # 업데이트 타이머 (모든 뷰의 재생 상태 체크)
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self._update_all_views)
        self.update_timer.start(33)  # ~30fps
        
        self._setup_ui()
        
    def _setup_ui(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)
        
        # 레이아웃 선택 툴바
        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.setSpacing(10)
        
        toolbar_layout.addWidget(QLabel("레이아웃:"))
        
        self.layout_buttons = QButtonGroup(self)
        layouts = [
            ("1×1", 1, 1),
            ("1×2", 1, 2),
            ("2×2", 2, 2),
            ("2×3", 2, 3),
            ("3×3", 3, 3),
        ]
        
        for text, rows, cols in layouts:
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.setFixedSize(50, 30)
            btn.setProperty("rows", rows)
            btn.setProperty("cols", cols)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #2d2d44;
                    color: #e0e0e0;
                    border: 1px solid #3d3d5c;
                    border-radius: 4px;
                    font-size: 12px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #3d3d5c;
                }
                QPushButton:checked {
                    background-color: #4ECDC4;
                    color: #1a1a2e;
                    border: none;
                }
            """)
            self.layout_buttons.addButton(btn)
            toolbar_layout.addWidget(btn)
            
            if rows == 1 and cols == 1:
                btn.setChecked(True)
        
        toolbar_layout.addStretch()
        
        # 도움말 레이블
        help_label = QLabel("각 뷰의 📁 버튼으로 폴더를 로드하세요")
        help_label.setStyleSheet("color: #888; font-size: 11px;")
        toolbar_layout.addWidget(help_label)
        
        self.main_layout.addWidget(toolbar)
        
        # 버튼 그룹 시그널 연결
        self.layout_buttons.buttonClicked.connect(self._on_layout_button_clicked)
        
        # 그리드 컨테이너
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setSpacing(8)
        
        self.main_layout.addWidget(self.grid_widget, 1)
        
        # 초기 레이아웃 설정
        self.set_layout(1, 1)
    
    def _on_layout_button_clicked(self, button):
        """레이아웃 버튼 클릭 핸들러"""
        rows = button.property("rows")
        cols = button.property("cols")
        self.set_layout(rows, cols)
        
    def set_layout(self, rows: int, cols: int):
        """그리드 레이아웃 변경"""
        old_rows, old_cols = self.current_layout
        self.current_layout = (rows, cols)
        total_views = rows * cols
        
        # 기존 뷰 패널 제거
        for panel in self.view_panels:
            panel.play_timer.stop()
            self.grid_layout.removeWidget(panel)
            panel.deleteLater()
        self.view_panels.clear()
        
        # 기존 행/열의 stretch를 0으로 리셋 (이전 레이아웃 여백 제거)
        for i in range(max(old_rows, rows)):
            self.grid_layout.setRowStretch(i, 0)
        for i in range(max(old_cols, cols)):
            self.grid_layout.setColumnStretch(i, 0)
        
        # 새 뷰 패널 생성
        for i in range(total_views):
            row = i // cols
            col = i % cols
            panel = ViewPanel(view_id=i)
            panel.person_selected.connect(self.person_selected.emit)  # 패널 시그널 릴레이
            self.view_panels.append(panel)
            self.grid_layout.addWidget(panel, row, col)
            
        # 그리드 셀 균등 분배 (현재 레이아웃에만 적용)
        for i in range(rows):
            self.grid_layout.setRowStretch(i, 1)
        for i in range(cols):
            self.grid_layout.setColumnStretch(i, 1)

    # 전체 패널에 대한 제어 기능 위임 메서드들
    def set_skeleton_model(self, model_name: str):
        for panel in self.view_panels:
            panel.canvas.set_skeleton_model(model_name)

    def set_confidence_threshold(self, threshold: float):
        for panel in self.view_panels:
            panel.canvas.set_confidence_threshold(threshold)

    def set_show_keypoints(self, show: bool):
        for panel in self.view_panels:
            panel.canvas.set_show_keypoints(show)

    def set_show_skeleton(self, show: bool):
        for panel in self.view_panels:
            panel.canvas.set_show_skeleton(show)

    def set_show_labels(self, show: bool):
        for panel in self.view_panels:
            panel.canvas.set_show_labels(show)

    def set_show_bbox(self, show: bool):
        for panel in self.view_panels:
            panel.canvas.set_show_bbox(show)

    def set_keypoint_size(self, size: int):
        for panel in self.view_panels:
            panel.canvas.set_keypoint_size(size)

    def set_keypoint_opacity(self, opacity: int):
        for panel in self.view_panels:
            panel.canvas.set_keypoint_opacity(opacity)

    def set_skeleton_width(self, width: int):
        for panel in self.view_panels:
            panel.canvas.set_skeleton_width(width)

    def set_skeleton_opacity(self, opacity: int):
        for panel in self.view_panels:
            panel.canvas.set_skeleton_opacity(opacity)

    def set_label_font_size(self, size: int):
        for panel in self.view_panels:
            panel.canvas.set_label_font_size(size)

    def set_label_opacity(self, opacity: int):
        for panel in self.view_panels:
            panel.canvas.set_label_opacity(opacity)

    def set_bbox_width(self, width: int):
        for panel in self.view_panels:
            panel.canvas.set_bbox_width(width)

    def set_bbox_opacity(self, opacity: int):
        for panel in self.view_panels:
            panel.canvas.set_bbox_opacity(opacity)

    def apply_filter(self, filter_type: str, params: dict):
        for panel in self.view_panels:
            panel.apply_filter(filter_type, params)

    def revert_filter(self):
        for panel in self.view_panels:
            panel.revert_filter()
            
    def _update_all_views(self):
        """모든 뷰의 재생 상태 업데이트"""
        for panel in self.view_panels:
            panel.update_playback()


class MultiViewWindow(QMainWindow):
    """Multi View 모드 독립 윈도우"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        
    def _setup_ui(self):
        self.setWindowTitle("2D Pose Viewer - Multi View")
        self.setMinimumSize(1000, 700)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a2e;
            }
        """)
        
        central = QWidget()
        self.setCentralWidget(central)
        
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # 레이아웃 선택 버튼 바
        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.setSpacing(10)
        
        toolbar_layout.addWidget(QLabel("레이아웃:"))
        
        self.layout_buttons = QButtonGroup(self)
        layouts = [
            ("1×1", 1, 1),
            ("1×2", 1, 2),
            ("2×2", 2, 2),
            ("2×3", 2, 3),
            ("3×3", 3, 3),
        ]
        
        for text, rows, cols in layouts:
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.setFixedSize(50, 30)
            btn.setProperty("rows", rows)
            btn.setProperty("cols", cols)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #2d2d44;
                    color: #e0e0e0;
                    border: 1px solid #3d3d5c;
                    border-radius: 4px;
                    font-size: 12px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #3d3d5c;
                }
                QPushButton:checked {
                    background-color: #4ECDC4;
                    color: #1a1a2e;
                    border: none;
                }
            """)
            self.layout_buttons.addButton(btn)
            toolbar_layout.addWidget(btn)
            
            if rows == 1 and cols == 1:
                btn.setChecked(True)
        
        toolbar_layout.addStretch()
        
        # 도움말 레이블
        help_label = QLabel("각 뷰의 📁 버튼으로 폴더를 로드하세요")
        help_label.setStyleSheet("color: #888; font-size: 11px;")
        toolbar_layout.addWidget(help_label)
        
        layout.addWidget(toolbar)
        
        # Multi View 컨테이너
        self.container = MultiViewContainer()
        layout.addWidget(self.container, 1)
        
        # 버튼 그룹 시그널 연결
        self.layout_buttons.buttonClicked.connect(self._on_layout_changed)
        
        # 상태바
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("""
            QStatusBar {
                background-color: #16213e;
                color: #e0e0e0;
            }
        """)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("레이아웃을 선택하고 각 뷰에 폴더를 로드하세요")
        
    def _on_layout_changed(self, button):
        rows = button.property("rows")
        cols = button.property("cols")
        self.container.set_layout(rows, cols)
        self.status_bar.showMessage(f"레이아웃 변경: {rows}×{cols} ({rows * cols}개 뷰)")
        
    def closeEvent(self, event):
        # 모든 타이머 정리
        self.container.update_timer.stop()
        for panel in self.container.view_panels:
            panel.play_timer.stop()
        super().closeEvent(event)
