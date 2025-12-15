"""
Stroke Analysis Dialog - 조정 스트로크 분석 결과 표시
"""

from typing import Dict, Optional, List
import csv

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget,
    QTableWidgetItem, QPushButton, QGroupBox, QSpinBox,
    QComboBox, QFormLayout, QSplitter, QWidget, QProgressBar,
    QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt, Signal, QRectF
from PySide6.QtGui import QPainter, QPen, QColor, QFontMetrics

from .stroke_detector import StrokeAnalysisResult, StrokeEvent


class SignalPlotWidget(QWidget):
    """정규화된 신호를 시각화하는 위젯"""
    
    frame_clicked = Signal(int)  # 그래프 클릭 시 해당 프레임 방출
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.wrist_data = None
        self.elbow_data = None
        self.shoulder_data = None
        self.strokes = []
        self.current_frame = 0  # 현재 프레임 커서용
        self.total_frames = 0
        self.setMinimumHeight(250)
        self.setMouseTracking(True)
        self.margin = 30  # 여백 증가 (축 표시용)
        
    def set_data(self, result: StrokeAnalysisResult):
        """분석 결과 데이터 설정"""
        # 필터링된 데이터가 있으면 우선 사용 (더 깨끗한 신호)
        if result.filtered_wrist_data is not None:
            self.wrist_data = result.filtered_wrist_data
            self.elbow_data = result.filtered_elbow_data
            self.shoulder_data = result.filtered_shoulder_data
            self.is_filtered = True
        else:
            self.wrist_data = result.normalized_wrist_data
            self.elbow_data = result.normalized_elbow_data
            self.shoulder_data = result.normalized_shoulder_data
            self.is_filtered = False
            
        self.strokes = result.strokes
        # 총 프레임 수 추정 (데이터 길이)
        if self.wrist_data is not None:
            self.total_frames = len(self.wrist_data)
        self.update()

    def set_current_frame(self, frame_idx: int):
        """현재 프레임 커서 위치 설정"""
        self.current_frame = frame_idx
        self.update()

    def mousePressEvent(self, event):
        """마우스 클릭 시 프레임 이동"""
        if self.total_frames == 0 or self.wrist_data is None:
            return
            
        x = event.position().x()
        w = self.width()
        margin = self.margin
        plot_w = w - 2 * margin
        
        # X좌표를 프레임으로 변환
        if margin <= x <= w - margin:
            ratio = (x - margin) / plot_w
            frame = int(ratio * self.total_frames)
            frame = max(0, min(frame, self.total_frames - 1))
            self.frame_clicked.emit(frame)
    
    def clear_data(self):
        """데이터 초기화"""
        self.wrist_data = None
        self.elbow_data = None
        self.shoulder_data = None
        self.strokes = []
        self.update()
    
    def paintEvent(self, event):
        """신호 그리기"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 배경
        painter.fillRect(self.rect(), QColor("#1a1a2e"))
        
        if self.wrist_data is None or len(self.wrist_data) == 0:
            painter.setPen(QColor("#888888"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "데이터 없음")
            return
        
        w = self.width()
        h = self.height()
        margin = self.margin
        plot_w = w - 2 * margin
        plot_h = h - 2 * margin
        
        # 1. 그리드 및 축 그리기
        self._draw_grid(painter, plot_w, plot_h, margin)
        
        # 2. 스트로크 영역 표시 (배경)
        for stroke in self.strokes:
            x1 = margin + int((stroke.frame_start / self.total_frames) * plot_w)
            x2 = margin + int((stroke.frame_end / self.total_frames) * plot_w)
            
            # 범위 체크
            x1 = max(margin, min(x1, w - margin))
            x2 = max(margin, min(x2, w - margin))
            
            if stroke.stroke_type == 'drive':
                color = QColor(78, 205, 196, 40)  # 청록색
            else:
                color = QColor(255, 107, 107, 30)  # 빨간색
            
            painter.fillRect(x1, margin, x2 - x1, plot_h, color)
        
        # 데이터 범위 계산
        all_data = []
        for data in [self.wrist_data, self.elbow_data, self.shoulder_data]:
            if data is not None:
                valid = data[~self._isnan(data)]
                all_data.extend(valid)
        
        if not all_data:
            return
            
        data_min = min(all_data)
        data_max = max(all_data)
        data_range = data_max - data_min if data_max != data_min else 1
        
        # 3. 신호 데이터 그리기
        # 손목 데이터 (청록색)
        if self.wrist_data is not None:
            self._draw_signal(painter, self.wrist_data, QColor("#4ECDC4"),
                            margin, plot_w, plot_h, data_min, data_range)
        
        # 팔꿈치 데이터 (노란색)
        if self.elbow_data is not None:
            self._draw_signal(painter, self.elbow_data, QColor("#FFE66D"),
                            margin, plot_w, plot_h, data_min, data_range)
        
        # 어깨 데이터 (분홍색)
        if self.shoulder_data is not None:
            self._draw_signal(painter, self.shoulder_data, QColor("#FF6B6B"),
                            margin, plot_w, plot_h, data_min, data_range)
        
        # 4. 현재 프레임 커서 그리기
        if 0 <= self.current_frame < self.total_frames:
            cx = margin + int((self.current_frame / self.total_frames) * plot_w)
            painter.setPen(QPen(QColor("#FFE66D"), 1, Qt.PenStyle.DashLine))  # 노란 점선
            painter.drawLine(cx, margin, cx, margin + plot_h)
            
            # 커서 상단 역삼각형 (심플하게)
            painter.setBrush(QColor("#FFE66D"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawConvexPolygon([
                QRectF(cx - 5, margin, 10, 0).topLeft(), 
                QRectF(cx + 5, margin, 0, 0).topLeft(), 
                QRectF(cx, margin + 6, 0, 0).topLeft()
            ])

        # 5. 범례
        self._draw_legend(painter)
    
    def _draw_grid(self, painter, plot_w, plot_h, margin):
        """그리드 및 축 그리기"""
        painter.setPen(QPen(QColor("#3d3d5c"), 1, Qt.PenStyle.DotLine))
        
        # 세로선 (프레임 단위) - 10등분
        frame_step = self.total_frames / 10 if self.total_frames > 0 else 10
        for i in range(11):
            x = margin + int((i / 10) * plot_w)
            painter.drawLine(x, margin, x, margin + plot_h)
            
            # X축 레이블 (프레임 번호)
            frame_num = int(i * frame_step)
            painter.setPen(QColor("#888888"))
            painter.drawText(x - 15, margin + plot_h + 15, 30, 20, 
                           Qt.AlignmentFlag.AlignCenter, str(frame_num))
            painter.setPen(QPen(QColor("#3d3d5c"), 1, Qt.PenStyle.DotLine))

        # 가로선 (중앙 0선 포함)
        mid_y = margin + plot_h // 2
        painter.setPen(QPen(QColor("#555555"), 1)) # 중앙선은 좀 더 진하게
        painter.drawLine(margin, mid_y, margin + plot_w, mid_y)
        
        # 테두리
        painter.setPen(QPen(QColor("#555555"), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(margin, margin, plot_w, plot_h)
    
    def _isnan(self, data):
        """NaN 체크"""
        import numpy as np
        return np.isnan(data)
    
    def _draw_signal(self, painter, data, color, margin, plot_w, plot_h, data_min, data_range):
        """신호 그리기"""
        import numpy as np
        
        pen = QPen(color)
        pen.setWidth(2)
        painter.setPen(pen)
        
        n = len(data)
        prev_x, prev_y = None, None
        
        for i, val in enumerate(data):
            if np.isnan(val):
                prev_x, prev_y = None, None
                continue
            
            x = margin + int((i / n) * plot_w)
            y = margin + plot_h - int(((val - data_min) / data_range) * plot_h)
            
            if prev_x is not None:
                painter.drawLine(prev_x, prev_y, x, y)
            
            prev_x, prev_y = x, y
    
    def _draw_legend(self, painter):
        """범례 그리기"""
        legend_x = 30
        legend_y = 30
        spacing = 60
        
        # 손목
        painter.setPen(QPen(QColor("#4ECDC4"), 3))
        painter.drawLine(legend_x, legend_y, legend_x + 20, legend_y)
        painter.setPen(QColor("#e0e0e0"))
        painter.drawText(legend_x + 25, legend_y + 5, "Wrist")
        
        # 팔꿈치
        painter.setPen(QPen(QColor("#FFE66D"), 3))
        painter.drawLine(legend_x + spacing, legend_y, legend_x + spacing + 20, legend_y)
        painter.setPen(QColor("#e0e0e0"))
        painter.drawText(legend_x + spacing + 25, legend_y + 5, "Elbow")
        
        # 어깨
        painter.setPen(QPen(QColor("#FF6B6B"), 3))
        painter.drawLine(legend_x + spacing * 2, legend_y, legend_x + spacing * 2 + 20, legend_y)
        painter.setPen(QColor("#e0e0e0"))
        painter.drawText(legend_x + spacing * 2 + 25, legend_y + 5, "Shoulder")


class StrokeAnalysisDialog(QDialog):
    """조정 스트로크 분석 결과 다이얼로그"""
    
    go_to_frame = Signal(int)  # 프레임 이동 시그널
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.results: Dict[int, StrokeAnalysisResult] = {}
        self.current_person_idx = -1
        self._setup_ui()
    
    def _setup_ui(self):
        self.setWindowTitle("조정 스트로크 분석")
        self.setMinimumSize(800, 600)
        self.setStyleSheet("""
            QDialog {
                background-color: #1a1a2e;
                color: #e0e0e0;
            }
            QGroupBox {
                border: 1px solid #3d3d5c;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                color: #e0e0e0;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QLabel {
                color: #e0e0e0;
            }
            QTableWidget {
                background-color: #2d2d44;
                color: #e0e0e0;
                border: 1px solid #3d3d5c;
                gridline-color: #3d3d5c;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QTableWidget::item:selected {
                background-color: #4ECDC4;
                color: #1a1a2e;
            }
            QHeaderView::section {
                background-color: #16213e;
                color: #e0e0e0;
                padding: 5px;
                border: 1px solid #3d3d5c;
            }
            QPushButton {
                background-color: #4ECDC4;
                color: #1a1a2e;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45B7AA;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #888888;
            }
        """)
        
        layout = QVBoxLayout(self)
        
        # 상단 컨트롤 (사람 선택)
        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("분석 대상:"))
        self.person_combo = QComboBox()
        self.person_combo.currentIndexChanged.connect(self._on_person_changed)
        top_layout.addWidget(self.person_combo)
        top_layout.addStretch()
        layout.addLayout(top_layout)
        
        # 요약 정보
        summary_group = QGroupBox("분석 요약")
        summary_layout = QHBoxLayout(summary_group)
        
        self.stroke_count_label = QLabel("스트로크 수: -")
        self.stroke_rate_label = QLabel("스트로크 레이트: - spm")
        self.frequency_label = QLabel("주기: - Hz")
        
        for label in [self.stroke_count_label, self.stroke_rate_label, self.frequency_label]:
            label.setStyleSheet("font-size: 14px; padding: 10px;")
            summary_layout.addWidget(label)
        
        layout.addWidget(summary_group)
        
        # 신호 플롯
        plot_group = QGroupBox("분석된 움직임")
        plot_layout = QVBoxLayout(plot_group)
        self.signal_plot = SignalPlotWidget()
        # 그래프 클릭 시 메인 뷰 이동
        self.signal_plot.frame_clicked.connect(self.go_to_frame.emit)
        plot_layout.addWidget(self.signal_plot)
        layout.addWidget(plot_group)
        
        # 스트로크 테이블
        table_group = QGroupBox("감지된 스트로크")
        table_layout = QVBoxLayout(table_group)
        
        self.stroke_table = QTableWidget()
        self.stroke_table.setColumnCount(6)  # 컬럼 6개로 증가
        self.stroke_table.setHorizontalHeaderLabels([
            "번호", "시작", "종료", "시간(s)", "유형", "신뢰도"
        ])
        self.stroke_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.stroke_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.stroke_table.doubleClicked.connect(self._on_table_double_click)
        
        # 열 너비 설정
        self.stroke_table.setColumnWidth(0, 50)  # 번호
        self.stroke_table.setColumnWidth(1, 70)  # 시작
        self.stroke_table.setColumnWidth(2, 70)  # 종료
        self.stroke_table.setColumnWidth(3, 70)  # 시간
        self.stroke_table.setColumnWidth(4, 90)  # 유형
        self.stroke_table.setColumnWidth(5, 70)  # 신뢰도
        
        table_layout.addWidget(self.stroke_table)
        layout.addWidget(table_group)
        
        # 버튼
        button_layout = QHBoxLayout()
        
        self.export_btn = QPushButton("CSV로 저장")
        self.export_btn.clicked.connect(self._export_csv)
        self.export_btn.setEnabled(False)
        
        self.go_to_btn = QPushButton("선택 프레임으로 이동")
        self.go_to_btn.clicked.connect(self._on_go_to_frame)
        self.go_to_btn.setEnabled(False)
        
        self.close_btn = QPushButton("닫기")
        self.close_btn.clicked.connect(self.close)
        self.close_btn.setStyleSheet("""
            QPushButton {
                background-color: #555555;
            }
            QPushButton:hover {
                background-color: #666666;
            }
        """)
        
        button_layout.addStretch()
        button_layout.addWidget(self.export_btn)
        button_layout.addWidget(self.go_to_btn)
        button_layout.addWidget(self.close_btn)
        layout.addLayout(button_layout)
    
    def set_results(self, results: Dict[int, StrokeAnalysisResult]):
        """여러 사람의 분석 결과 설정"""
        self.results = results
        self.person_combo.blockSignals(True)
        self.person_combo.clear()
        
        if not results:
            self.person_combo.addItem("데이터 없음")
            self.person_combo.setEnabled(False)
            return
            
        sorted_indices = sorted(results.keys())
        for idx in sorted_indices:
            self.person_combo.addItem(f"Person {idx}", idx)
            
        self.person_combo.setEnabled(True)
        self.person_combo.blockSignals(False)
        
        if sorted_indices:
            self.person_combo.setCurrentIndex(0)
            self._update_display(sorted_indices[0])

    def _on_person_changed(self, index):
        """사람 선택 변경 시"""
        if index < 0:
            return
        person_idx = self.person_combo.currentData()
        self._update_display(person_idx)

    def _update_display(self, person_idx: int):
        """선택된 사람의 데이터로 화면 업데이트"""
        if person_idx not in self.results:
            return
            
        self.current_person_idx = person_idx
        result = self.results[person_idx]
        
        # 요약 업데이트
        self.stroke_count_label.setText(f"스트로크 수: {result.stroke_count}")
        self.stroke_rate_label.setText(f"스트로크 레이트: {result.avg_stroke_rate:.1f} spm")
        self.frequency_label.setText(f"주파수: {result.dominant_frequency:.2f} Hz")
        
        # 플롯 업데이트
        self.signal_plot.set_data(result)
        
        # 테이블 업데이트
        self.stroke_table.setRowCount(len(result.strokes))
        fps = 30.0  # 기본값, 실제로는 result에 fps 정보가 있으면 좋은데... 일단 30.
        
        for i, stroke in enumerate(result.strokes):
            self.stroke_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.stroke_table.setItem(i, 1, QTableWidgetItem(str(stroke.frame_start)))
            self.stroke_table.setItem(i, 2, QTableWidgetItem(str(stroke.frame_end)))
            
            # 시간 계산
            duration_sec = (stroke.frame_end - stroke.frame_start) / fps
            self.stroke_table.setItem(i, 3, QTableWidgetItem(f"{duration_sec:.2f}s"))
            
            type_text = "드라이브" if stroke.stroke_type == 'drive' else "리커버리"
            self.stroke_table.setItem(i, 4, QTableWidgetItem(type_text))
            self.stroke_table.setItem(i, 5, QTableWidgetItem(f"{stroke.confidence:.2f}"))
        
        self.go_to_btn.setEnabled(len(result.strokes) > 0)
        self.export_btn.setEnabled(len(result.strokes) > 0)

    def update_current_frame(self, frame_idx: int):
        """현재 프레임 업데이트 (외부 호출용)"""
        self.signal_plot.set_current_frame(frame_idx)
        
        # 테이블에서 해당 프레임 구간 하이라이트? (선택사항)
        # 너무 번쩍거릴 수 있으니 패스하거나, 원하면 추가.

    def _export_csv(self):
        """분석 결과를 CSV로 저장"""
        if self.current_person_idx not in self.results:
            return
            
        result = self.results[self.current_person_idx]
        if not result.strokes:
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "CSV로 저장", f"stroke_analysis_person_{self.current_person_idx}.csv", 
            "CSV Files (*.csv)"
        )
        
        if not file_path:
            return
            
        try:
            with open(file_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(["번호", "시작 프레임", "종료 프레임", "시간(초)", "유형", "신뢰도"])
                
                fps = 30.0
                for i, stroke in enumerate(result.strokes):
                    duration = (stroke.frame_end - stroke.frame_start) / fps
                    writer.writerow([
                        i + 1, 
                        stroke.frame_start, 
                        stroke.frame_end, 
                        f"{duration:.2f}",
                        stroke.stroke_type,
                        stroke.confidence
                    ])
            
            QMessageBox.information(self, "저장 완료", "성공적으로 저장되었습니다.")
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"저장 중 오류 발생: {str(e)}")
    
    def set_result(self, result: StrokeAnalysisResult):
        """단일 분석 결과 설정 (호환성 유지용)"""
        self.set_results({0: result})
    
    def _on_table_double_click(self, index):
        """테이블 더블클릭 시 해당 프레임으로 이동"""
        row = index.row()
        if self.current_person_idx in self.results:
            result = self.results[self.current_person_idx]
            if row >= 0 and row < len(result.strokes):
                frame = result.strokes[row].frame_start
                self.go_to_frame.emit(frame)
    
    def _on_go_to_frame(self):
        """선택된 프레임으로 이동"""
        rows = self.stroke_table.selectedIndexes()
        if rows and self.current_person_idx in self.results:
            result = self.results[self.current_person_idx]
            row = rows[0].row()
            if row >= 0 and row < len(result.strokes):
                frame = result.strokes[row].frame_start
                self.go_to_frame.emit(frame)
