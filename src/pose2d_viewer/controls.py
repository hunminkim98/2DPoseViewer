"""
Control widgets for pose viewer - PlaybackBar and ControlPanel.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QSlider, QLabel, QPushButton, QSpinBox,
    QGroupBox, QComboBox, QCheckBox, QScrollArea
)
from PySide6.QtCore import Qt, Signal

from .constants import SKELETON_MODELS


class PlaybackBar(QWidget):
    """하단 재생 컨트롤 바"""
    
    frame_changed = Signal(int)
    playback_toggled = Signal(bool)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.total_frames = 0
        self.is_playing = False
        self._setup_ui()
        
    def _setup_ui(self):
        self.setFixedHeight(60)
        self.setObjectName("playbackBar")
        self.setStyleSheet("""
            #playbackBar {
                background-color: #16213e;
                border-top: 1px solid #3d3d5c;
            }
            QLabel {
                background-color: transparent;
                border: none;
            }
        """)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 8, 15, 8)
        layout.setSpacing(15)
        
        # 재생/일시정지 버튼
        self.play_btn = QPushButton("▶")
        self.play_btn.setFixedSize(40, 40)
        self.play_btn.clicked.connect(self._toggle_playback)
        self.play_btn.setStyleSheet("""
            QPushButton {
                background-color: #4ECDC4;
                color: #1a1a2e;
                border: none;
                font-size: 16px;
                font-weight: bold;
                border-radius: 20px;
            }
            QPushButton:hover {
                background-color: #5FE6DD;
            }
            QPushButton:pressed {
                background-color: #3DBDB5;
            }
        """)
        layout.addWidget(self.play_btn)
        
        # 프레임 슬라이더
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.valueChanged.connect(self._on_slider_changed)
        self.frame_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: none;
                height: 6px;
                background: #2d2d44;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #4ECDC4;
                border: none;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #5FE6DD;
            }
            QSlider::sub-page:horizontal {
                background: #4ECDC4;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.frame_slider, 1)
        
        # 프레임 라벨
        self.frame_label = QLabel("0 / 0")
        self.frame_label.setStyleSheet("color: #e0e0e0; font-size: 13px; min-width: 100px; background: transparent;")
        layout.addWidget(self.frame_label)
        
        # FPS 설정
        fps_label = QLabel("FPS:")
        fps_label.setStyleSheet("color: #a0a0a0; background: transparent;")
        layout.addWidget(fps_label)
        
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 120)
        self.fps_spin.setValue(30)
        self.fps_spin.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.fps_spin.setStyleSheet("""
            QSpinBox {
                background-color: #2d2d44;
                color: #e0e0e0;
                border: 1px solid #3d3d5c;
                border-radius: 4px;
                padding: 4px 8px;
                min-width: 40px;
            }
        """)
        layout.addWidget(self.fps_spin)
    
    def set_total_frames(self, total: int):
        self.total_frames = total
        self.frame_slider.setMaximum(max(0, total - 1))
        self._update_frame_label()
    
    def set_current_frame(self, frame: int):
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(frame)
        self.frame_slider.blockSignals(False)
        self._update_frame_label()
    
    def _update_frame_label(self):
        current = self.frame_slider.value()
        self.frame_label.setText(f"{current} / {self.total_frames - 1}")
    
    def _on_slider_changed(self, value):
        self._update_frame_label()
        self.frame_changed.emit(value)
    
    def _toggle_playback(self):
        self.is_playing = not self.is_playing
        self.play_btn.setText("■" if self.is_playing else "▶")
        self.playback_toggled.emit(self.is_playing)
    
    def get_fps(self) -> int:
        return self.fps_spin.value()
    
    def stop_playback(self):
        self.is_playing = False
        self.play_btn.setText("▶")


class ControlPanel(QWidget):
    """컨트롤 패널 위젯"""
    
    # 시그널 정의
    confidence_changed = Signal(float)
    show_keypoints_changed = Signal(bool)
    show_skeleton_changed = Signal(bool)
    show_labels_changed = Signal(bool)
    show_bbox_changed = Signal(bool)
    person_selected = Signal(int)
    model_changed = Signal(str)
    filter_apply_requested = Signal(str, dict)
    filter_revert_requested = Signal()
    
    # 시각화 옵션 시그널
    keypoint_size_changed = Signal(int)
    keypoint_opacity_changed = Signal(int)
    skeleton_width_changed = Signal(int)
    skeleton_opacity_changed = Signal(int)
    label_size_changed = Signal(int)
    label_opacity_changed = Signal(int)
    bbox_width_changed = Signal(int)
    bbox_opacity_changed = Signal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        
    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # 스크롤 영역 생성
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                background-color: #2d2d44;
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background-color: #4ECDC4;
                border-radius: 5px;
                min-height: 30px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        # 스크롤 컨텐트 위젯
        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)
        layout.setSpacing(12)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # === 스켈레톤 모델 선택 ===
        model_group = QGroupBox("스켈레톤 모델")
        model_group.setStyleSheet(self._get_group_style())
        model_layout = QVBoxLayout(model_group)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(list(SKELETON_MODELS.keys()))
        self.model_combo.setCurrentText("HALPE_26")
        self.model_combo.currentTextChanged.connect(lambda t: self.model_changed.emit(t))
        self.model_combo.setStyleSheet(self._get_combo_style())
        
        model_layout.addWidget(self.model_combo)
        layout.addWidget(model_group)
        
        # === 필터링 ===
        filter_group = QGroupBox("필터링")
        filter_group.setStyleSheet(self._get_group_style())
        filter_layout = QVBoxLayout(filter_group)
        
        # Confidence 임계값
        conf_layout = QHBoxLayout()
        conf_label = QLabel("Confidence:")
        conf_label.setStyleSheet("color: #e0e0e0;")
        
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setMinimum(0)
        self.conf_slider.setMaximum(100)
        self.conf_slider.setValue(30)
        self.conf_slider.valueChanged.connect(self._on_confidence_changed)
        self.conf_slider.setStyleSheet(self._get_slider_style())
        
        self.conf_value_label = QLabel("0.30")
        self.conf_value_label.setStyleSheet("color: #4ECDC4; font-weight: bold; min-width: 40px;")
        
        conf_layout.addWidget(conf_label)
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_value_label)
        filter_layout.addLayout(conf_layout)
        
        # 사람 선택
        person_layout = QHBoxLayout()
        self.person_id_label = QLabel("Person ID:")
        self.person_id_label.setStyleSheet("color: #e0e0e0;")
        
        self.person_spin = QSpinBox()
        self.person_spin.setRange(-1, 10)
        self.person_spin.setValue(-1)
        self.person_spin.setSpecialValueText("전체")
        self.person_spin.valueChanged.connect(lambda v: self.person_selected.emit(v))
        self.person_spin.setStyleSheet(self._get_spinbox_style())
        
        person_layout.addWidget(self.person_id_label)
        person_layout.addWidget(self.person_spin)
        person_layout.addStretch()
        filter_layout.addLayout(person_layout)
        
        layout.addWidget(filter_group)
        
        # === 스무딩 필터 ===
        smooth_group = QGroupBox("스무딩 필터")
        smooth_group.setStyleSheet(self._get_group_style())
        smooth_layout = QVBoxLayout(smooth_group)
        
        # 필터 유형 선택
        filter_type_layout = QHBoxLayout()
        filter_type_label = QLabel("필터 유형:")
        filter_type_label.setStyleSheet("color: #e0e0e0;")
        
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["Butterworth", "Gaussian", "Median"])
        self.filter_combo.setStyleSheet(self._get_combo_style())
        self.filter_combo.currentTextChanged.connect(self._on_filter_type_changed)
        
        filter_type_layout.addWidget(filter_type_label)
        filter_type_layout.addWidget(self.filter_combo)
        smooth_layout.addLayout(filter_type_layout)
        
        # Butterworth 파라미터
        self.butter_params_widget = QWidget()
        butter_layout = QVBoxLayout(self.butter_params_widget)
        butter_layout.setContentsMargins(0, 5, 0, 0)
        
        order_layout = QHBoxLayout()
        order_label = QLabel("Order:")
        order_label.setStyleSheet("color: #a0a0a0;")
        self.butter_order_spin = QSpinBox()
        self.butter_order_spin.setRange(2, 8)
        self.butter_order_spin.setValue(4)
        self.butter_order_spin.setStyleSheet(self._get_spinbox_style())
        order_layout.addWidget(order_label)
        order_layout.addWidget(self.butter_order_spin)
        order_layout.addStretch()
        butter_layout.addLayout(order_layout)
        
        cutoff_layout = QHBoxLayout()
        cutoff_label = QLabel("Cutoff (Hz):")
        cutoff_label.setStyleSheet("color: #a0a0a0;")
        self.butter_cutoff_spin = QSpinBox()
        self.butter_cutoff_spin.setRange(1, 30)
        self.butter_cutoff_spin.setValue(6)
        self.butter_cutoff_spin.setStyleSheet(self._get_spinbox_style())
        cutoff_layout.addWidget(cutoff_label)
        cutoff_layout.addWidget(self.butter_cutoff_spin)
        cutoff_layout.addStretch()
        butter_layout.addLayout(cutoff_layout)
        smooth_layout.addWidget(self.butter_params_widget)
        
        # Gaussian 파라미터
        self.gauss_params_widget = QWidget()
        gauss_layout = QVBoxLayout(self.gauss_params_widget)
        gauss_layout.setContentsMargins(0, 5, 0, 0)
        
        sigma_layout = QHBoxLayout()
        sigma_label = QLabel("Sigma:")
        sigma_label.setStyleSheet("color: #a0a0a0;")
        self.gauss_sigma_spin = QSpinBox()
        self.gauss_sigma_spin.setRange(1, 20)
        self.gauss_sigma_spin.setValue(3)
        self.gauss_sigma_spin.setStyleSheet(self._get_spinbox_style())
        sigma_layout.addWidget(sigma_label)
        sigma_layout.addWidget(self.gauss_sigma_spin)
        sigma_layout.addStretch()
        gauss_layout.addLayout(sigma_layout)
        smooth_layout.addWidget(self.gauss_params_widget)
        self.gauss_params_widget.hide()
        
        # Median 파라미터
        self.median_params_widget = QWidget()
        median_layout = QVBoxLayout(self.median_params_widget)
        median_layout.setContentsMargins(0, 5, 0, 0)
        
        kernel_layout = QHBoxLayout()
        kernel_label = QLabel("Kernel Size:")
        kernel_label.setStyleSheet("color: #a0a0a0;")
        self.median_kernel_spin = QSpinBox()
        self.median_kernel_spin.setRange(3, 21)
        self.median_kernel_spin.setValue(5)
        self.median_kernel_spin.setSingleStep(2)
        self.median_kernel_spin.setStyleSheet(self._get_spinbox_style())
        kernel_layout.addWidget(kernel_label)
        kernel_layout.addWidget(self.median_kernel_spin)
        kernel_layout.addStretch()
        median_layout.addLayout(kernel_layout)
        smooth_layout.addWidget(self.median_params_widget)
        self.median_params_widget.hide()
        
        # 적용/되돌리기 버튼
        btn_layout = QHBoxLayout()
        self.apply_filter_btn = QPushButton("적용")
        self.apply_filter_btn.clicked.connect(self._on_apply_filter)
        self.apply_filter_btn.setStyleSheet(self._get_button_style())
        
        self.revert_filter_btn = QPushButton("되돌리기")
        self.revert_filter_btn.clicked.connect(lambda: self.filter_revert_requested.emit())
        self.revert_filter_btn.setStyleSheet(self._get_button_style().replace("#4ECDC4", "#FF6B6B").replace("#5FE6DD", "#FF8E8E").replace("#3DBDB5", "#E65555"))
        self.revert_filter_btn.setEnabled(False)
        
        btn_layout.addWidget(self.apply_filter_btn)
        btn_layout.addWidget(self.revert_filter_btn)
        smooth_layout.addLayout(btn_layout)
        
        # 필터 상태 라벨
        self.filter_status_label = QLabel("원본 데이터")
        self.filter_status_label.setStyleSheet("color: #95E1D3; font-size: 11px;")
        smooth_layout.addWidget(self.filter_status_label)
        
        layout.addWidget(smooth_group)
        
        # === 표시 옵션 ===
        display_group = QGroupBox("표시 옵션")
        display_group.setStyleSheet(self._get_group_style())
        display_layout = QVBoxLayout(display_group)
        
        # 키포인트 옵션
        self.show_keypoints_cb = QCheckBox("키포인트 표시")
        self.show_keypoints_cb.setChecked(True)
        self.show_keypoints_cb.stateChanged.connect(
            lambda s: self.show_keypoints_changed.emit(s == Qt.CheckState.Checked.value))
        self.show_keypoints_cb.setStyleSheet(self._get_checkbox_style())
        display_layout.addWidget(self.show_keypoints_cb)
        
        kp_options = QHBoxLayout()
        kp_size_label = QLabel("크기:")
        kp_size_label.setStyleSheet("color: #a0a0a0; font-size: 11px;")
        self.kp_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.kp_size_slider.setRange(2, 15)
        self.kp_size_slider.setValue(6)
        self.kp_size_slider.setFixedWidth(60)
        self.kp_size_slider.setStyleSheet(self._get_mini_slider_style())
        self.kp_size_slider.valueChanged.connect(lambda v: self.keypoint_size_changed.emit(v))
        kp_opacity_label = QLabel("투명도:")
        kp_opacity_label.setStyleSheet("color: #a0a0a0; font-size: 11px;")
        self.kp_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.kp_opacity_slider.setRange(50, 255)
        self.kp_opacity_slider.setValue(255)
        self.kp_opacity_slider.setFixedWidth(60)
        self.kp_opacity_slider.setStyleSheet(self._get_mini_slider_style())
        self.kp_opacity_slider.valueChanged.connect(lambda v: self.keypoint_opacity_changed.emit(v))
        kp_options.addWidget(kp_size_label)
        kp_options.addWidget(self.kp_size_slider)
        kp_options.addWidget(kp_opacity_label)
        kp_options.addWidget(self.kp_opacity_slider)
        kp_options.addStretch()
        display_layout.addLayout(kp_options)
        
        # 스켈레톤 옵션
        self.show_skeleton_cb = QCheckBox("스켈레톤 표시")
        self.show_skeleton_cb.setChecked(True)
        self.show_skeleton_cb.stateChanged.connect(
            lambda s: self.show_skeleton_changed.emit(s == Qt.CheckState.Checked.value))
        self.show_skeleton_cb.setStyleSheet(self._get_checkbox_style())
        display_layout.addWidget(self.show_skeleton_cb)
        
        sk_options = QHBoxLayout()
        sk_width_label = QLabel("굵기:")
        sk_width_label.setStyleSheet("color: #a0a0a0; font-size: 11px;")
        self.sk_width_slider = QSlider(Qt.Orientation.Horizontal)
        self.sk_width_slider.setRange(1, 10)
        self.sk_width_slider.setValue(3)
        self.sk_width_slider.setFixedWidth(60)
        self.sk_width_slider.setStyleSheet(self._get_mini_slider_style())
        self.sk_width_slider.valueChanged.connect(lambda v: self.skeleton_width_changed.emit(v))
        sk_opacity_label = QLabel("투명도:")
        sk_opacity_label.setStyleSheet("color: #a0a0a0; font-size: 11px;")
        self.sk_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.sk_opacity_slider.setRange(50, 255)
        self.sk_opacity_slider.setValue(255)
        self.sk_opacity_slider.setFixedWidth(60)
        self.sk_opacity_slider.setStyleSheet(self._get_mini_slider_style())
        self.sk_opacity_slider.valueChanged.connect(lambda v: self.skeleton_opacity_changed.emit(v))
        sk_options.addWidget(sk_width_label)
        sk_options.addWidget(self.sk_width_slider)
        sk_options.addWidget(sk_opacity_label)
        sk_options.addWidget(self.sk_opacity_slider)
        sk_options.addStretch()
        display_layout.addLayout(sk_options)
        
        # 라벨 옵션
        self.show_labels_cb = QCheckBox("라벨 표시")
        self.show_labels_cb.setChecked(False)
        self.show_labels_cb.stateChanged.connect(
            lambda s: self.show_labels_changed.emit(s == Qt.CheckState.Checked.value))
        self.show_labels_cb.setStyleSheet(self._get_checkbox_style())
        display_layout.addWidget(self.show_labels_cb)
        
        label_options = QHBoxLayout()
        label_size_label = QLabel("폰트:")
        label_size_label.setStyleSheet("color: #a0a0a0; font-size: 11px;")
        self.label_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.label_size_slider.setRange(6, 20)
        self.label_size_slider.setValue(8)
        self.label_size_slider.setFixedWidth(60)
        self.label_size_slider.setStyleSheet(self._get_mini_slider_style())
        self.label_size_slider.valueChanged.connect(lambda v: self.label_size_changed.emit(v))
        label_opacity_label = QLabel("투명도:")
        label_opacity_label.setStyleSheet("color: #a0a0a0; font-size: 11px;")
        self.label_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.label_opacity_slider.setRange(50, 255)
        self.label_opacity_slider.setValue(255)
        self.label_opacity_slider.setFixedWidth(60)
        self.label_opacity_slider.setStyleSheet(self._get_mini_slider_style())
        self.label_opacity_slider.valueChanged.connect(lambda v: self.label_opacity_changed.emit(v))
        label_options.addWidget(label_size_label)
        label_options.addWidget(self.label_size_slider)
        label_options.addWidget(label_opacity_label)
        label_options.addWidget(self.label_opacity_slider)
        label_options.addStretch()
        display_layout.addLayout(label_options)
        
        # BBox 옵션
        self.show_bbox_cb = QCheckBox("Bounding Box 표시")
        self.show_bbox_cb.setChecked(True)
        self.show_bbox_cb.stateChanged.connect(
            lambda s: self.show_bbox_changed.emit(s == Qt.CheckState.Checked.value))
        self.show_bbox_cb.setStyleSheet(self._get_checkbox_style())
        display_layout.addWidget(self.show_bbox_cb)
        
        bbox_options = QHBoxLayout()
        bbox_width_label = QLabel("굵기:")
        bbox_width_label.setStyleSheet("color: #a0a0a0; font-size: 11px;")
        self.bbox_width_slider = QSlider(Qt.Orientation.Horizontal)
        self.bbox_width_slider.setRange(1, 8)
        self.bbox_width_slider.setValue(2)
        self.bbox_width_slider.setFixedWidth(60)
        self.bbox_width_slider.setStyleSheet(self._get_mini_slider_style())
        self.bbox_width_slider.valueChanged.connect(lambda v: self.bbox_width_changed.emit(v))
        bbox_opacity_label = QLabel("투명도:")
        bbox_opacity_label.setStyleSheet("color: #a0a0a0; font-size: 11px;")
        self.bbox_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.bbox_opacity_slider.setRange(50, 255)
        self.bbox_opacity_slider.setValue(200)
        self.bbox_opacity_slider.setFixedWidth(60)
        self.bbox_opacity_slider.setStyleSheet(self._get_mini_slider_style())
        self.bbox_opacity_slider.valueChanged.connect(lambda v: self.bbox_opacity_changed.emit(v))
        bbox_options.addWidget(bbox_width_label)
        bbox_options.addWidget(self.bbox_width_slider)
        bbox_options.addWidget(bbox_opacity_label)
        bbox_options.addWidget(self.bbox_opacity_slider)
        bbox_options.addStretch()
        display_layout.addLayout(bbox_options)
        
        layout.addWidget(display_group)
        
        # === 단축키 안내 ===
        help_group = QGroupBox("단축키")
        help_group.setStyleSheet(self._get_group_style())
        help_layout = QVBoxLayout(help_group)
        
        shortcuts = [
            ("Space", "재생/일시정지"),
            ("← / →", "이전/다음 프레임"),
            ("Home / End", "처음/끝 프레임"),
        ]
        
        for key, desc in shortcuts:
            row = QHBoxLayout()
            key_label = QLabel(key)
            key_label.setStyleSheet("""
                color: #4ECDC4; 
                background-color: #2d2d44; 
                padding: 3px 8px; 
                border-radius: 4px;
                font-family: 'Consolas', monospace;
            """)
            desc_label = QLabel(desc)
            desc_label.setStyleSheet("color: #a0a0a0;")
            row.addWidget(key_label)
            row.addWidget(desc_label)
            row.addStretch()
            help_layout.addLayout(row)
            
        layout.addWidget(help_group)
        layout.addStretch()
        
        # 스크롤 영역에 컨텐트 설정
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)
        
    def _get_group_style(self):
        return """
            QGroupBox {
                color: #e0e0e0;
                font-size: 13px;
                font-weight: bold;
                border: 1px solid #3d3d5c;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """
        
    def _get_slider_style(self):
        return """
            QSlider::groove:horizontal {
                border: 1px solid #3d3d5c;
                height: 6px;
                background: #2d2d44;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #4ECDC4;
                border: none;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #5FE6DD;
            }
            QSlider::sub-page:horizontal {
                background: #4ECDC4;
                border-radius: 3px;
            }
        """
    
    def _get_mini_slider_style(self):
        return """
            QSlider::groove:horizontal {
                border: none;
                height: 4px;
                background: #2d2d44;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #4ECDC4;
                border: none;
                width: 10px;
                height: 10px;
                margin: -3px 0;
                border-radius: 5px;
            }
            QSlider::handle:horizontal:hover {
                background: #5FE6DD;
            }
            QSlider::sub-page:horizontal {
                background: #4ECDC4;
                border-radius: 2px;
            }
        """
        
    def _get_button_style(self):
        return """
            QPushButton {
                background-color: #4ECDC4;
                color: #1a1a2e;
                border: none;
                padding: 8px 20px;
                font-size: 12px;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #5FE6DD;
            }
            QPushButton:pressed {
                background-color: #3DBDB5;
            }
        """
        
    def _get_spinbox_style(self):
        return """
            QSpinBox, QDoubleSpinBox {
                background-color: #2d2d44;
                color: #e0e0e0;
                border: 1px solid #3d3d5c;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QSpinBox::up-button, QSpinBox::down-button,
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                background-color: #3d3d5c;
                border: none;
            }
        """
        
    def _get_checkbox_style(self):
        return """
            QCheckBox {
                color: #e0e0e0;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 2px solid #3d3d5c;
                background-color: #2d2d44;
            }
            QCheckBox::indicator:checked {
                background-color: #4ECDC4;
                border-color: #4ECDC4;
            }
        """
    
    def _get_combo_style(self):
        return """
            QComboBox {
                background-color: #2d2d44;
                color: #e0e0e0;
                border: 1px solid #3d3d5c;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 12px;
            }
            QComboBox:hover {
                border-color: #4ECDC4;
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #4ECDC4;
                margin-right: 10px;
            }
            QComboBox QAbstractItemView {
                background-color: #2d2d44;
                color: #e0e0e0;
                border: 1px solid #3d3d5c;
                selection-background-color: #4ECDC4;
                selection-color: #1a1a2e;
            }
        """
        
    def _on_confidence_changed(self, value):
        conf = value / 100.0
        self.conf_value_label.setText(f"{conf:.2f}")
        self.confidence_changed.emit(conf)
    
    def _on_filter_type_changed(self, filter_type: str):
        """필터 유형 변경 시 파라미터 위젯 표시/숨김"""
        self.butter_params_widget.setVisible(filter_type == "Butterworth")
        self.gauss_params_widget.setVisible(filter_type == "Gaussian")
        self.median_params_widget.setVisible(filter_type == "Median")
    
    def _on_apply_filter(self):
        """필터 적용 버튼 클릭"""
        filter_type = self.filter_combo.currentText().lower()
        
        if filter_type == "butterworth":
            params = {
                'order': self.butter_order_spin.value(),
                'cutoff': self.butter_cutoff_spin.value()
            }
        elif filter_type == "gaussian":
            params = {
                'sigma': self.gauss_sigma_spin.value()
            }
        elif filter_type == "median":
            params = {
                'kernel_size': self.median_kernel_spin.value()
            }
        else:
            params = {}
        
        self.filter_apply_requested.emit(filter_type, params)
    
    def set_filter_applied(self, applied: bool, filter_name: str = ""):
        """필터 적용 상태 업데이트"""
        self.revert_filter_btn.setEnabled(applied)
        if applied:
            self.filter_status_label.setText(f"✓ {filter_name} 필터 적용됨")
            self.filter_status_label.setStyleSheet("color: #4ECDC4; font-size: 11px; font-weight: bold;")
        else:
            self.filter_status_label.setText("원본 데이터")
            self.filter_status_label.setStyleSheet("color: #95E1D3; font-size: 11px;")
