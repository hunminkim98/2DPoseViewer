"""
2D Pose Viewer - Main Application Window
"""

import sys
import json
import glob
import os
from typing import List, Optional

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFileDialog, QStatusBar, QSplitter
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeyEvent

from .models import Keypoint, Person, FrameData
from .canvas import PoseCanvas
from .controls import PlaybackBar, ControlPanel
from .constants import SKELETON_MODELS


class PoseViewerWindow(QMainWindow):
    """메인 윈도우"""
    
    def __init__(self):
        super().__init__()
        self.frame_files: List[str] = []
        self.current_frame: int = 0
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._on_timer_tick)
        
        # 필터링 관련 데이터
        self.all_frames_data: List[FrameData] = []
        self.filtered_frames_data: Optional[List[FrameData]] = None
        self.is_filtered: bool = False
        
        self._setup_ui()
        self._connect_signals()
        
    def _setup_ui(self):
        self.setWindowTitle("2D Pose Viewer")
        self.setMinimumSize(1200, 800)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a2e;
            }
            QStatusBar {
                background-color: #16213e;
                color: #e0e0e0;
            }
        """)
        
        # 중앙 위젯
        central = QWidget()
        self.setCentralWidget(central)
        
        # 메인 레이아웃
        outer_layout = QVBoxLayout(central)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)
        
        # 상단 컨텐츠 영역
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(10)
        
        # 캔버스
        self.canvas = PoseCanvas()
        
        # 컨트롤 패널
        self.control_panel = ControlPanel()
        self.control_panel.setFixedWidth(300)
        
        # 스플리터
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.canvas)
        splitter.addWidget(self.control_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        
        content_layout.addWidget(splitter)
        outer_layout.addWidget(content_widget, 1)
        
        # 재생 컨트롤 바
        self.playback_bar = PlaybackBar()
        outer_layout.addWidget(self.playback_bar)
        
        # 상태바
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("폴더를 불러오려면 Ctrl+O를 누르세요")
        
        # 메뉴바
        self._setup_menubar()
        
    def _setup_menubar(self):
        menubar = self.menuBar()
        menubar.setStyleSheet("""
            QMenuBar {
                background-color: #16213e;
                color: #e0e0e0;
                padding: 5px;
            }
            QMenuBar::item:selected {
                background-color: #4ECDC4;
                color: #1a1a2e;
            }
            QMenu {
                background-color: #2d2d44;
                color: #e0e0e0;
                border: 1px solid #3d3d5c;
            }
            QMenu::item:selected {
                background-color: #4ECDC4;
                color: #1a1a2e;
            }
        """)
        
        file_menu = menubar.addMenu("파일")
        
        open_action = file_menu.addAction("폴더 열기")
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_folder)
        
        file_menu.addSeparator()
        
        exit_action = file_menu.addAction("종료")
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        
    def _connect_signals(self):
        # 컨트롤 패널 시그널
        self.control_panel.confidence_changed.connect(self.canvas.set_confidence_threshold)
        self.control_panel.show_keypoints_changed.connect(self.canvas.set_show_keypoints)
        self.control_panel.show_skeleton_changed.connect(self.canvas.set_show_skeleton)
        self.control_panel.show_labels_changed.connect(self.canvas.set_show_labels)
        self.control_panel.show_bbox_changed.connect(self.canvas.set_show_bbox)
        self.control_panel.person_selected.connect(self.canvas.set_selected_person)
        self.control_panel.model_changed.connect(self.canvas.set_skeleton_model)
        self.control_panel.filter_apply_requested.connect(self._apply_filter)
        self.control_panel.filter_revert_requested.connect(self._revert_filter)
        
        # 재생 바 시그널
        self.playback_bar.frame_changed.connect(self._go_to_frame)
        self.playback_bar.playback_toggled.connect(self._toggle_playback)
        
        # 시각화 옵션 시그널
        self.control_panel.keypoint_size_changed.connect(self.canvas.set_keypoint_size)
        self.control_panel.keypoint_opacity_changed.connect(self.canvas.set_keypoint_opacity)
        self.control_panel.skeleton_width_changed.connect(self.canvas.set_skeleton_width)
        self.control_panel.skeleton_opacity_changed.connect(self.canvas.set_skeleton_opacity)
        self.control_panel.label_size_changed.connect(self.canvas.set_label_font_size)
        self.control_panel.label_opacity_changed.connect(self.canvas.set_label_opacity)
        self.control_panel.bbox_width_changed.connect(self.canvas.set_bbox_width)
        self.control_panel.bbox_opacity_changed.connect(self.canvas.set_bbox_opacity)
        
        # 캔버스 시그널
        self.canvas.person_clicked.connect(self._on_person_clicked)
        
    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Space:
            self.playback_bar._toggle_playback()
            self._toggle_playback(self.playback_bar.is_playing)
        elif event.key() == Qt.Key.Key_Left:
            self._prev_frame()
        elif event.key() == Qt.Key.Key_Right:
            self._next_frame()
        elif event.key() == Qt.Key.Key_Home:
            self._go_to_frame(0)
        elif event.key() == Qt.Key.Key_End:
            self._go_to_frame(len(self.frame_files) - 1)
        else:
            super().keyPressEvent(event)
    
    def _on_person_clicked(self, person_idx: int):
        self.control_panel.person_spin.setValue(person_idx)
        self.canvas.set_selected_person(person_idx)
            
    def _open_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "JSON 파일이 있는 폴더 선택", "",
            QFileDialog.Option.ShowDirsOnly
        )
        if folder:
            self._load_folder(folder)
            
    def _load_folder(self, folder: str):
        pattern = os.path.join(folder, "*.json")
        files = sorted(glob.glob(pattern))
        
        if not files:
            self.status_bar.showMessage(f"⚠ JSON 파일을 찾을 수 없습니다: {folder}")
            return
            
        self.frame_files = files
        self.current_frame = 0
        
        self.all_frames_data = []
        self.filtered_frames_data = None
        self.is_filtered = False
        self.control_panel.set_filter_applied(False)
        
        for idx, file_path in enumerate(files):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                frame_data = self._parse_frame_data(data, idx)
                self.all_frames_data.append(frame_data)
            except Exception:
                self.all_frames_data.append(FrameData(frame_number=idx, people=[]))
        
        self.playback_bar.set_total_frames(len(files))
        self._load_current_frame()
        self.status_bar.showMessage(f"✓ {len(files)}개 프레임 로드됨: {folder}")
        
    def _load_current_frame(self):
        if not self.all_frames_data or self.current_frame >= len(self.all_frames_data):
            return
        
        if self.is_filtered and self.filtered_frames_data:
            frame_data = self.filtered_frames_data[self.current_frame]
        else:
            frame_data = self.all_frames_data[self.current_frame]
        
        self.canvas.set_frame_data(frame_data)
        self.playback_bar.set_current_frame(self.current_frame)
            
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
        
    def _go_to_frame(self, frame: int):
        if not self.frame_files:
            return
        self.current_frame = max(0, min(frame, len(self.frame_files) - 1))
        self._load_current_frame()
        
    def _prev_frame(self):
        if self.current_frame > 0:
            self._go_to_frame(self.current_frame - 1)
            
    def _next_frame(self):
        if self.current_frame < len(self.frame_files) - 1:
            self._go_to_frame(self.current_frame + 1)
            
    def _toggle_playback(self, playing: bool):
        if playing:
            fps = self.playback_bar.get_fps()
            self.play_timer.start(int(1000 / fps))
        else:
            self.play_timer.stop()
            
    def _on_timer_tick(self):
        if self.current_frame >= len(self.all_frames_data) - 1:
            self._go_to_frame(0)
        else:
            self._next_frame()
    
    def _apply_filter(self, filter_type: str, params: dict):
        if not self.all_frames_data:
            return
        
        self.status_bar.showMessage("필터 적용 중...")
        QApplication.processEvents()
        
        try:
            max_people = max(len(fd.people) for fd in self.all_frames_data)
            max_keypoints = max(
                max((len(p.keypoints) for p in fd.people), default=0)
                for fd in self.all_frames_data
            )
            
            if max_people == 0 or max_keypoints == 0:
                self.status_bar.showMessage("⚠ 필터링할 데이터가 없습니다")
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
            filter_name = filter_type.capitalize()
            self.control_panel.set_filter_applied(True, filter_name)
            self._load_current_frame()
            self.status_bar.showMessage(f"✓ {filter_name} 필터 적용 완료")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status_bar.showMessage(f"⚠ 필터링 오류: {e}")
    
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
    
    def _revert_filter(self):
        self.filtered_frames_data = None
        self.is_filtered = False
        self.control_panel.set_filter_applied(False)
        self._load_current_frame()
        self.status_bar.showMessage("✓ 원본 데이터로 복원됨")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = PoseViewerWindow()
    window.show()
    sys.exit(app.exec())


def run_app():
    """Entry point for the application."""
    main()


if __name__ == "__main__":
    main()
