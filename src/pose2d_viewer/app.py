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
    QFileDialog, QStatusBar, QSplitter, QMessageBox, QStackedWidget
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeyEvent

from .models import Keypoint, Person, FrameData
from .canvas import PoseCanvas
from .controls import PlaybackBar, ControlPanel
from .constants import SKELETON_MODELS
from .stroke_detector import detect_rowing_strokes
from .stroke_dialog import StrokeAnalysisDialog
from .multi_view import MultiViewContainer


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
        
        # 분석 다이얼로그 참조 보관 (Modeless용)
        self.analysis_dialog = None
        
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
        
        # 캔버스 (싱글 뷰)
        self.canvas = PoseCanvas()
        
        # 멀티 뷰 컨테이너
        self.multi_view_container = MultiViewContainer()
        
        # 스택 위젯 (싱글뷰/멀티뷰 전환용)
        self.view_stack = QStackedWidget()
        self.view_stack.addWidget(self.canvas)  # index 0: 싱글뷰
        self.view_stack.addWidget(self.multi_view_container)  # index 1: 멀티뷰
        self.view_stack.setCurrentIndex(0)
        self.is_multi_view_mode = False
        
        # 컨트롤 패널
        self.control_panel = ControlPanel()
        self.control_panel.setFixedWidth(300)
        
        # 스플리터
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.view_stack)
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
        
        export_action = file_menu.addAction("선택된 Person 내보내기")
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self._export_person_json)
        
        file_menu.addSeparator()
        
        exit_action = file_menu.addAction("종료")
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        
        # 분석 메뉴
        analysis_menu = menubar.addMenu("분석")
        
        rowing_action = analysis_menu.addAction("조정")
        rowing_action.setShortcut("Ctrl+R")
        rowing_action.triggered.connect(self._analyze_rowing_stroke)
        
        # 보기 메뉴
        view_menu = menubar.addMenu("보기")
        
        self.multi_view_action = view_menu.addAction("Multi View 모드")
        self.multi_view_action.setShortcut("Ctrl+M")
        self.multi_view_action.setCheckable(True)
        self.multi_view_action.triggered.connect(self._toggle_multi_view)
        
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
        self.multi_view_container.person_selected.connect(self._on_person_clicked)
        
        # 인물 선택 변경 시 멀티뷰에도 적용 (이 부분은 싱글뷰 설정 시에만 의미 있도록 유지하거나 제거 가능)
        # 멀티뷰에서 전역 제어를 원할 수도 있으므로 일단 연결은 유지하되, 각 패널 클릭은 독립적임
        self.control_panel.person_selected.connect(self._on_global_person_change)
        
        # 멀티뷰 제어를 위해 컨트롤 패널 시그널을 멀티뷰 컨테이너에도 연결
        self.control_panel.confidence_changed.connect(self.multi_view_container.set_confidence_threshold)
        self.control_panel.show_keypoints_changed.connect(self.multi_view_container.set_show_keypoints)
        self.control_panel.show_skeleton_changed.connect(self.multi_view_container.set_show_skeleton)
        self.control_panel.show_labels_changed.connect(self.multi_view_container.set_show_labels)
        self.control_panel.show_bbox_changed.connect(self.multi_view_container.set_show_bbox)
        self.control_panel.model_changed.connect(self.multi_view_container.set_skeleton_model)
        
        # 시각화 옵션
        self.control_panel.keypoint_size_changed.connect(self.multi_view_container.set_keypoint_size)
        self.control_panel.keypoint_opacity_changed.connect(self.multi_view_container.set_keypoint_opacity)
        self.control_panel.skeleton_width_changed.connect(self.multi_view_container.set_skeleton_width)
        self.control_panel.skeleton_opacity_changed.connect(self.multi_view_container.set_skeleton_opacity)
        self.control_panel.label_size_changed.connect(self.multi_view_container.set_label_font_size)
        self.control_panel.label_opacity_changed.connect(self.multi_view_container.set_label_opacity)
        self.control_panel.bbox_width_changed.connect(self.multi_view_container.set_bbox_width)
        self.control_panel.bbox_opacity_changed.connect(self.multi_view_container.set_bbox_opacity)
        
        # 필터링
        self.control_panel.filter_apply_requested.connect(self.multi_view_container.apply_filter)
        self.control_panel.filter_revert_requested.connect(self.multi_view_container.revert_filter)
        
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
        """캔버스 클릭 시 호출 (싱글뷰/멀티뷰 공통)"""
        self.control_panel.person_spin.setValue(person_idx)
        self.canvas.set_selected_person(person_idx)
        # 멀티뷰 강제 동기화(set_selected_person)는 하지 않음으로써 독립 선택 보장
        if person_idx >= 0:
            self.status_bar.showMessage(f"✓ Person {person_idx} 선택됨")
        else:
            self.status_bar.showMessage("✓ 선택 해제")

    def _on_global_person_change(self, person_idx: int):
        """컨트롤 패널에서 인물 변경 시 호출 (필요 시 전체 적용용)"""
        self.canvas.set_selected_person(person_idx)
        # 사용자가 컨트롤 패널 스핀박스를 직접 조작하면 여전히 전체에 적용될 수 있으나 
        # 캔버스 직접 클릭은 독립적으로 유지됨.
        # 만약 컨트롤 패널 조작마저도 독립적이어야 한다면 이 로직을 비활성화.
        pass
            
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
        
        # 분석 다이얼로그가 열려있으면 현재 프레임 동기화
        if self.analysis_dialog is not None and self.analysis_dialog.isVisible():
            self.analysis_dialog.update_current_frame(self.current_frame)
            
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
    
    def _export_person_json(self):
        """선택된 person 데이터를 JSON으로 저장"""
        # 선택된 person 확인
        selected_person = self.canvas.selected_person
        if selected_person < 0:
            QMessageBox.warning(self, "경고", "먼저 Person을 선택해주세요.\n(bbox를 클릭하거나 컨트롤 패널에서 선택)")
            return
        
        # 저장 폴더 선택
        parent_folder = QFileDialog.getExistingDirectory(
            self, "내보낼 결과 폴더 선택", "",
            QFileDialog.Option.ShowDirsOnly
        )
        if not parent_folder:
            return

        # 멀티뷰 모드 분기
        if self.is_multi_view_mode:
            self._export_multi_view_person(parent_folder, selected_person)
        else:
            # 기존 싱글뷰 저장
            if not self.all_frames_data:
                QMessageBox.warning(self, "경고", "먼저 데이터를 로드해주세요.")
                return
            output_folder = os.path.join(parent_folder, f"Person_{selected_person}")
            os.makedirs(output_folder, exist_ok=True)
            self._perform_json_export(self.all_frames_data, self.filtered_frames_data, 
                                     self.frame_files, output_folder, selected_person,
                                     is_filtered=(self.is_filtered and self.filtered_frames_data))

    def _export_multi_view_person(self, parent_folder: str, global_selected_person: int):
        """멀티뷰의 모든 패널 데이터를 각각의 선택된 ID로 내보내기"""
        active_panels = [p for p in self.multi_view_container.view_panels if p.folder_path]
        if not active_panels:
            QMessageBox.warning(self, "경고", "멀티뷰에 로드된 폴더가 없습니다.")
            return

        total_saved = 0
        exported_info = []
        
        for panel in active_panels:
            # 각 패널의 캔버스에서 개별적으로 선택된 ID 가져오기
            panel_selected_person = panel.canvas.selected_person
            
            # 아무도 선택 안 된 패널은 건너뛰거나 전체 내보낼지 결정 (요구사항상 선택된 인물만)
            if panel_selected_person < 0:
                continue
                
            folder_name = os.path.basename(panel.folder_path)
            output_folder = os.path.join(parent_folder, f"Person_{panel_selected_person}_{folder_name}")
            os.makedirs(output_folder, exist_ok=True)
            
            count = self._perform_json_export(panel.all_frames_data, panel.filtered_frames_data, 
                                            panel.frame_files, output_folder, panel_selected_person, 
                                            is_filtered=panel.is_filtered, update_status=False)
            total_saved += count
            exported_info.append(f"- {folder_name}: Person {panel_selected_person}")
            
        if not exported_info:
            QMessageBox.warning(self, "경고", "선택된 인물이 있는 뷰가 없습니다.")
            return

        info_text = "\n".join(exported_info)
        QMessageBox.information(self, "저장 완료", 
                              f"각 뷰에서 선택된 인물 데이터를 내보냈습니다.\n\n"
                              f"{info_text}\n\n"
                              f"총 저장된 파일: {total_saved}개\n"
                              f"저장 위치: {parent_folder}")

    def _perform_json_export(self, all_data, filtered_data, files, output_folder, selected_person, 
                            is_filtered=False, update_status=True):
        """실제 데이터 저장 로직"""
        frames_data = filtered_data if (is_filtered and filtered_data) else all_data
        data_source = "필터링된 데이터" if is_filtered else "원본 데이터"
            
        if update_status:
            self.status_bar.showMessage(f"JSON 내보내기 중... (Person {selected_person}, {data_source})")
            QApplication.processEvents()
            
        saved_count = 0
        try:
            for frame_idx, frame_data in enumerate(frames_data):
                if selected_person >= len(frame_data.people):
                    continue
                
                person = frame_data.people[selected_person]
                pose_keypoints_2d = []
                for kp in person.keypoints:
                    pose_keypoints_2d.extend([kp.x, kp.y, kp.confidence])
                
                export_person_id = selected_person if person.person_id == -1 else person.person_id
                
                output_data = {
                    "people": [{
                        "person_id": [export_person_id],
                        "pose_keypoints_2d": pose_keypoints_2d
                    }]
                }
                
                original_filename = os.path.basename(files[frame_idx])
                output_path = os.path.join(output_folder, original_filename)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2)
                saved_count += 1
                
            if update_status:
                self.status_bar.showMessage(f"✓ {saved_count}개 파일 저장 완료 (Person {selected_person})")
        except Exception as e:
            if update_status:
                QMessageBox.critical(self, "오류", f"저장 중 오류 발생: {str(e)}")
            
        return saved_count
    
    def _analyze_rowing_stroke(self):
        """조정 스트로크 분석 실행"""
        if not self.all_frames_data:
            self.status_bar.showMessage("⚠ 먼저 데이터를 로드해주세요")
            return
        
        self.status_bar.showMessage("조정 스트로크 분석 중...")
        QApplication.processEvents()
        
        try:
            # 현재 표시중인 데이터 사용 (필터링 적용시 필터링된 데이터)
            if self.is_filtered and self.filtered_frames_data:
                frames_to_analyze = self.filtered_frames_data
                data_source = "필터링된 데이터"
            else:
                frames_to_analyze = self.all_frames_data
                data_source = "원본 데이터"
            
            # 최대 사람 수 확인
            max_people = 0
            for frame in frames_to_analyze:
                max_people = max(max_people, len(frame.people))
            
            if max_people == 0:
                self.status_bar.showMessage(f"⚠ 분석할 사람 데이터가 없습니다 ({data_source})")
                return
            
            # 현재 스켈레톤 모델
            model_name = self.control_panel.model_combo.currentText()
            
            # FPS 가져오기
            fps = self.playback_bar.get_fps()
            
            results = {}
            valid_count = 0
            
            # 모든 사람에 대해 분석 실행
            import numpy as np  # numpy 필요
            
            for person_idx in range(max_people):
                result = detect_rowing_strokes(
                    frames_data=frames_to_analyze,
                    person_idx=person_idx,
                    model_name=model_name,
                    fps=fps,
                    side="both"
                )
                
                # 유효한 데이터가 있고 스트로크가 1회 이상 감지된 경우만 포함
                if (result.normalized_wrist_data is not None and 
                    len(result.normalized_wrist_data) > 0 and 
                    not np.all(np.isnan(result.normalized_wrist_data)) and
                    result.stroke_count >= 1):  # 스트로크 1회 이상만
                    
                    results[person_idx] = result
                    valid_count += 1
            
            if valid_count == 0:
                self.status_bar.showMessage(f"⚠ 스트로크가 감지된 사람이 없습니다 ({data_source})")
                return
            
            # 기존 다이얼로그가 있으면 닫기
            if self.analysis_dialog is not None:
                self.analysis_dialog.close()
            
            # 다이얼로그 표시 (Modeless)
            self.analysis_dialog = StrokeAnalysisDialog(self)
            self.analysis_dialog.set_results(results)
            self.analysis_dialog.go_to_frame.connect(self._go_to_frame)
            
            self.status_bar.showMessage(
                f"✓ 분석 완료 ({valid_count}명): Person {', '.join(map(str, results.keys()))}"
            )
            
            self.analysis_dialog.show()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status_bar.showMessage(f"⚠ 분석 오류: {e}")
    
    def _toggle_multi_view(self, checked: bool):
        """Multi View 모드 전환"""
        if checked:
            self.view_stack.setCurrentIndex(1)
            self.is_multi_view_mode = True
            # 멀티뷰 모드 활성화 시 레이블 변경 (독립 선택임을 명시)
            self.control_panel.person_id_label.setText("Last Selected ID:")
            self.control_panel.person_id_label.setStyleSheet("color: #FFD93D; font-weight: bold;")
            
            # 멀티뷰 모드 활성화 시 현재 컨트롤 패널의 설정을 모든 뷰에 동기화
            self.multi_view_container.set_skeleton_model(self.control_panel.model_combo.currentText())
            self.status_bar.showMessage("✓ Multi View 모드 활성화")
        else:
            self.view_stack.setCurrentIndex(0)
            self.is_multi_view_mode = False
            # 원복
            self.control_panel.person_id_label.setText("Person ID:")
            self.control_panel.person_id_label.setStyleSheet("color: #e0e0e0; font-weight: normal;")
            self.status_bar.showMessage("✓ Single View 모드 활성화")


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
