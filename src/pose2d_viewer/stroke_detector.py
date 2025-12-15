"""
Rowing Stroke Detector (조정 스트로크 감지기)

Hip 좌표를 기준으로 상체 관절의 주기적 움직임(사인 웨이브)을 분석하여
자동으로 스트로크를 감지합니다.

Logic:
- 손목이 앞-뒤 움직임 (sine wave)
- 팔꿈치가 앞-뒤 움직임 (sine wave)
- 어깨가 앞-뒤로 움직임 (sine wave)
- Hip 좌표를 기준으로 상대적 정규화 (카메라 이동 보정)
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d


@dataclass
class StrokeEvent:
    """스트로크 이벤트"""
    frame_start: int
    frame_end: int
    stroke_type: str  # 'catch', 'drive', 'finish', 'recovery'
    confidence: float


@dataclass
class StrokeAnalysisResult:
    """스트로크 분석 결과"""
    strokes: List[StrokeEvent]
    stroke_count: int
    avg_stroke_rate: float  # strokes per minute
    dominant_frequency: float
    normalized_wrist_data: Optional[np.ndarray] = None
    normalized_elbow_data: Optional[np.ndarray] = None
    normalized_shoulder_data: Optional[np.ndarray] = None
    filtered_wrist_data: Optional[np.ndarray] = None
    filtered_elbow_data: Optional[np.ndarray] = None
    filtered_shoulder_data: Optional[np.ndarray] = None


# 스켈레톤 모델별 키포인트 인덱스 매핑
KEYPOINT_INDICES = {
    "HALPE_26": {
        "LWrist": 9, "RWrist": 10,
        "LElbow": 7, "RElbow": 8,
        "LShoulder": 5, "RShoulder": 6,
        "LHip": 11, "RHip": 12,
    },
    "COCO_17": {
        "LWrist": 9, "RWrist": 10,
        "LElbow": 7, "RElbow": 8,
        "LShoulder": 5, "RShoulder": 6,
        "LHip": 11, "RHip": 12,
    },
    "COCO_133": {
        "LWrist": 9, "RWrist": 10,
        "LElbow": 7, "RElbow": 8,
        "LShoulder": 5, "RShoulder": 6,
        "LHip": 11, "RHip": 12,
    },
    "COCO_133_WRIST": {
        "LWrist": 9, "RWrist": 10,
        "LElbow": 7, "RElbow": 8,
        "LShoulder": 5, "RShoulder": 6,
        "LHip": 11, "RHip": 12,
    },
    "BODY_25": {
        "LWrist": 7, "RWrist": 4,
        "LElbow": 6, "RElbow": 3,
        "LShoulder": 5, "RShoulder": 2,
        "LHip": 12, "RHip": 9,
    },
    "BODY_25B": {
        "LWrist": 9, "RWrist": 10,
        "LElbow": 7, "RElbow": 8,
        "LShoulder": 5, "RShoulder": 6,
        "LHip": 11, "RHip": 12,
    },
    "BLAZEPOSE": {
        "LWrist": 15, "RWrist": 16,
        "LElbow": 13, "RElbow": 14,
        "LShoulder": 11, "RShoulder": 12,
        "LHip": 23, "RHip": 24,
    },
}


def get_keypoint_index(model_name: str, joint_name: str) -> Optional[int]:
    """모델에서 특정 관절의 인덱스 반환"""
    if model_name not in KEYPOINT_INDICES:
        # 기본값으로 HALPE_26 사용
        model_name = "HALPE_26"
    return KEYPOINT_INDICES.get(model_name, {}).get(joint_name)


def extract_joint_coordinates(
    frames_data: List,
    person_idx: int,
    joint_name: str,
    model_name: str = "HALPE_26"
) -> Tuple[np.ndarray, np.ndarray]:
    """프레임 데이터에서 특정 관절의 X, Y 좌표 시계열 추출"""
    joint_idx = get_keypoint_index(model_name, joint_name)
    if joint_idx is None:
        return np.array([]), np.array([])
    
    x_coords = []
    y_coords = []
    
    for frame_data in frames_data:
        if person_idx < len(frame_data.people):
            person = frame_data.people[person_idx]
            if joint_idx < len(person.keypoints):
                kp = person.keypoints[joint_idx]
                if kp.confidence > 0.3:  # 신뢰도 임계값
                    x_coords.append(kp.x)
                    y_coords.append(kp.y)
                else:
                    x_coords.append(np.nan)
                    y_coords.append(np.nan)
            else:
                x_coords.append(np.nan)
                y_coords.append(np.nan)
        else:
            x_coords.append(np.nan)
            y_coords.append(np.nan)
    
    return np.array(x_coords), np.array(y_coords)


def get_hip_center(
    frames_data: List,
    person_idx: int,
    model_name: str = "HALPE_26"
) -> Tuple[np.ndarray, np.ndarray]:
    """양쪽 Hip의 중간점 계산"""
    l_hip_x, l_hip_y = extract_joint_coordinates(frames_data, person_idx, "LHip", model_name)
    r_hip_x, r_hip_y = extract_joint_coordinates(frames_data, person_idx, "RHip", model_name)
    
    # 양쪽 Hip의 평균
    hip_center_x = (l_hip_x + r_hip_x) / 2
    hip_center_y = (l_hip_y + r_hip_y) / 2
    
    return hip_center_x, hip_center_y


def normalize_by_hip(
    joint_x: np.ndarray,
    joint_y: np.ndarray,
    hip_x: np.ndarray,
    hip_y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Hip 좌표 기준으로 정규화 (카메라 이동 보정)"""
    norm_x = joint_x - hip_x
    norm_y = joint_y - hip_y
    return norm_x, norm_y


def interpolate_nans(data: np.ndarray) -> np.ndarray:
    """NaN 값을 선형 보간으로 채움"""
    result = data.copy()
    nan_mask = np.isnan(result)
    
    if np.all(nan_mask):
        return result
    
    valid_indices = np.where(~nan_mask)[0]
    if len(valid_indices) < 2:
        return result
    
    result[nan_mask] = np.interp(
        np.where(nan_mask)[0],
        valid_indices,
        result[valid_indices]
    )
    
    return result


def smooth_signal(data: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """가우시안 스무딩 적용"""
    data_interp = interpolate_nans(data)
    if np.any(np.isnan(data_interp)):
        return data_interp
    return gaussian_filter1d(data_interp, sigma)


def find_dominant_frequency(
    data: np.ndarray,
    fps: float = 30.0,
    min_freq: float = 0.2,
    max_freq: float = 2.0
) -> Tuple[float, np.ndarray, np.ndarray]:
    """FFT를 사용하여 지배적인 주파수 찾기
    
    Returns:
        dominant_freq: 지배적 주파수 (Hz)
        freqs: 주파수 배열
        power: 파워 스펙트럼
    """
    data_clean = interpolate_nans(data)
    if np.any(np.isnan(data_clean)):
        return 0.0, np.array([]), np.array([])
    
    # 평균 제거
    data_demean = data_clean - np.mean(data_clean)
    
    # FFT 계산
    n = len(data_demean)
    fft_result = np.fft.fft(data_demean)
    power = np.abs(fft_result[:n//2]) ** 2
    freqs = np.fft.fftfreq(n, d=1/fps)[:n//2]
    
    # 관심 주파수 범위에서 피크 찾기
    mask = (freqs >= min_freq) & (freqs <= max_freq)
    if not np.any(mask):
        return 0.0, freqs, power
    
    masked_power = power.copy()
    masked_power[~mask] = 0
    
    dominant_idx = np.argmax(masked_power)
    dominant_freq = freqs[dominant_idx]
    
    return dominant_freq, freqs, power


def bandpass_filter(
    data: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4
) -> np.ndarray:
    """대역 통과 필터 적용 (DC 성분 및 고주파 노이즈 제거)"""
    # NaN 보간
    data_clean = interpolate_nans(data)
    if np.any(np.isnan(data_clean)):
        return data_clean
    
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # 필터 설계 및 적용
    try:
        b, a = signal.butter(order, [low, high], btype='band')
        # 데이터 길이가 필터 차수보다 작으면 에러 발생 가능
        if len(data_clean) <= 3 * order:
            return data_clean
        y = signal.filtfilt(b, a, data_clean)
        return y
    except Exception:
        return data_clean

def find_stroke_peaks(
    data: np.ndarray,
    dominant_freq: float,
    fps: float = 30.0,
    prominence_factor: float = 0.5  # 표준편차 배수
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """스트로크 피크(최대점)와 밸리(최소점) 찾기, 필터링된 데이터 반환
    
    Returns:
        peaks: 피크 프레임 인덱스
        valleys: 밸리 프레임 인덱스
        filtered_data: 분석에 사용된 필터링된 데이터
    """
    # 1. NaN 보간
    data_interp = interpolate_nans(data)
    if np.any(np.isnan(data_interp)):
        return np.array([]), np.array([]), data
    
    # 2. 대역 통과 필터 (0.2 ~ 3.0 Hz) - 조정 스트로크 대역
    # 주파수 성분만 남기고 트렌드(DC) 제거
    filtered_data = bandpass_filter(data_interp, 0.2, 3.0, fps)
    
    # 3. 예상 피크 간 거리 계산
    if dominant_freq > 0.15:  # 너무 낮은 주파수는 무시
        expected_distance = max(5, int(fps / dominant_freq * 0.4))
    else:
        # 주파수 감지 실패 시, 일반적인 조정 레이트(20spm = 0.33Hz) 가정
        expected_distance = int(fps * 0.5)
    
    # 4. Prominence 결정 (표준편차 기반)
    # 전체 범위(max-min)를 쓰면 이상치에 취약하므로 표준편차나 IQR 사용
    std_val = np.std(filtered_data)
    if std_val < 1e-6:
        return np.array([]), np.array([]), filtered_data
        
    prominence = std_val * prominence_factor
    
    # 5. 피크/밸리 찾기
    peaks, _ = signal.find_peaks(
        filtered_data,
        distance=expected_distance,
        prominence=prominence
    )
    
    valleys, _ = signal.find_peaks(
        -filtered_data,
        distance=expected_distance,
        prominence=prominence
    )
    
    # 6. 유효 데이터 구간 필터링
    valid_mask = ~np.isnan(data)
    # 원본 데이터가 NaN인 곳의 피크는 제거
    peaks = peaks[valid_mask[peaks]]
    valleys = valleys[valid_mask[valleys]]
    
    return peaks, valleys, filtered_data


def classify_stroke_phases(
    peaks: np.ndarray,
    valleys: np.ndarray,
    total_frames: int
) -> List[StrokeEvent]:
    """피크와 밸리를 기반으로 스트로크 단계 분류"""
    strokes = []
    
    # 이벤트 병합 및 정렬
    all_events = []
    for p in peaks:
        all_events.append((p, 'peak'))
    for v in valleys:
        all_events.append((v, 'valley'))
    
    all_events.sort(key=lambda x: x[0])
    
    if len(all_events) < 2:
        return strokes
    
    # 연속된 이벤트 분석
    for i in range(len(all_events) - 1):
        frame_start = all_events[i][0]
        frame_end = all_events[i + 1][0]
        start_type = all_events[i][1]
        end_type = all_events[i + 1][1]
        
        # 같은 타입이 연속되면 건너뜀 (피크-피크, 밸리-밸리)
        if start_type == end_type:
            continue
            
        if start_type == 'valley':
            # Valley -> Peak: Drive (보통 앞으로 뻗었다가 당기는 구간)
            stroke_type = 'drive'
        else:
            # Peak -> Valley: Recovery (당겼다가 다시 앞으로 나가는 구간)
            stroke_type = 'recovery'
        
        # 지속 시간 체크 (너무 짧거나 긴 것은 제외)
        duration = frame_end - frame_start
        if duration < 5: # 5프레임 미만은 노이즈일 가능성 큼
            continue
            
        confidence = min(1.0, duration / 15.0)  # 예: 15프레임(0.5초) 정도면 신뢰도 1
        
        strokes.append(StrokeEvent(
            frame_start=int(frame_start),
            frame_end=int(frame_end),
            stroke_type=stroke_type,
            confidence=confidence
        ))
    
    return strokes


def detect_rowing_strokes(
    frames_data: List,
    person_idx: int = 0,
    model_name: str = "HALPE_26",
    fps: float = 30.0,
    side: str = "both"  # "left", "right", "both"
) -> StrokeAnalysisResult:
    """조정 스트로크 자동 감지
    
    Args:
        frames_data: 프레임 데이터 리스트
        person_idx: 분석할 사람 인덱스
        model_name: 스켈레톤 모델 이름
        fps: 프레임 레이트
        side: 분석할 팔 ("left", "right", "both")
    
    Returns:
        StrokeAnalysisResult: 스트로크 분석 결과
    """
    if not frames_data:
        return StrokeAnalysisResult(
            strokes=[],
            stroke_count=0,
            avg_stroke_rate=0.0,
            dominant_frequency=0.0
        )
    
    # Hip 중심 좌표 추출
    hip_x, hip_y = get_hip_center(frames_data, person_idx, model_name)
    
    # 손목 좌표 추출 및 정규화
    if side in ["left", "both"]:
        l_wrist_x, l_wrist_y = extract_joint_coordinates(
            frames_data, person_idx, "LWrist", model_name
        )
        l_wrist_norm_x, _ = normalize_by_hip(l_wrist_x, l_wrist_y, hip_x, hip_y)
    
    if side in ["right", "both"]:
        r_wrist_x, r_wrist_y = extract_joint_coordinates(
            frames_data, person_idx, "RWrist", model_name
        )
        r_wrist_norm_x, _ = normalize_by_hip(r_wrist_x, r_wrist_y, hip_x, hip_y)
    
    # 분석할 손목 데이터 선택
    if side == "left":
        wrist_x = l_wrist_norm_x
    elif side == "right":
        wrist_x = r_wrist_norm_x
    else:  # both - 양쪽 평균
        wrist_x = (l_wrist_norm_x + r_wrist_norm_x) / 2
    
    # 팔꿈치 좌표 추출 및 정규화
    if side in ["left", "both"]:
        l_elbow_x, l_elbow_y = extract_joint_coordinates(
            frames_data, person_idx, "LElbow", model_name
        )
        l_elbow_norm_x, _ = normalize_by_hip(l_elbow_x, l_elbow_y, hip_x, hip_y)
    
    if side in ["right", "both"]:
        r_elbow_x, r_elbow_y = extract_joint_coordinates(
            frames_data, person_idx, "RElbow", model_name
        )
        r_elbow_norm_x, _ = normalize_by_hip(r_elbow_x, r_elbow_y, hip_x, hip_y)
    
    if side == "left":
        elbow_x = l_elbow_norm_x
    elif side == "right":
        elbow_x = r_elbow_norm_x
    else:
        elbow_x = (l_elbow_norm_x + r_elbow_norm_x) / 2
    
    # 어깨 좌표 추출 및 정규화
    if side in ["left", "both"]:
        l_shoulder_x, l_shoulder_y = extract_joint_coordinates(
            frames_data, person_idx, "LShoulder", model_name
        )
        l_shoulder_norm_x, _ = normalize_by_hip(l_shoulder_x, l_shoulder_y, hip_x, hip_y)
    
    if side in ["right", "both"]:
        r_shoulder_x, r_shoulder_y = extract_joint_coordinates(
            frames_data, person_idx, "RShoulder", model_name
        )
        r_shoulder_norm_x, _ = normalize_by_hip(r_shoulder_x, r_shoulder_y, hip_x, hip_y)
    
    if side == "left":
        shoulder_x = l_shoulder_norm_x
    elif side == "right":
        shoulder_x = r_shoulder_norm_x
    else:
        shoulder_x = (l_shoulder_norm_x + r_shoulder_norm_x) / 2
    
    # 손목 데이터로 지배적 주파수 찾기 (손목이 가장 큰 움직임)
    dominant_freq, _, _ = find_dominant_frequency(wrist_x, fps)
    
    # 피크 찾기 (내부에서 필터링 수행됨)
    peaks, valleys, filtered_wrist = find_stroke_peaks(wrist_x, dominant_freq, fps)
    
    # 다른 관절 데이터도 필터링 (시각화용)
    filtered_elbow = bandpass_filter(elbow_x, 0.2, 3.0, fps)
    filtered_shoulder = bandpass_filter(shoulder_x, 0.2, 3.0, fps)
    
    # 스트로크 분류
    strokes = classify_stroke_phases(peaks, valleys, len(frames_data))
    
    # Drive 스트로크 수 계산
    drive_count = sum(1 for s in strokes if s.stroke_type == 'drive')
    
    # 평균 스트로크 레이트 계산 (strokes per minute)
    if len(frames_data) > 0 and drive_count > 0:
        duration_seconds = len(frames_data) / fps
        avg_stroke_rate = (drive_count / duration_seconds) * 60
    elif dominant_freq > 0.1:
        # 스트로크 감지는 못했지만 주파수는 있는 경우 주파수 기반 추정치라도 제공
        avg_stroke_rate = dominant_freq * 60
    else:
        avg_stroke_rate = 0.0
    
    return StrokeAnalysisResult(
        strokes=strokes,
        stroke_count=drive_count,
        avg_stroke_rate=avg_stroke_rate,
        dominant_frequency=dominant_freq,
        normalized_wrist_data=wrist_x,
        normalized_elbow_data=elbow_x,
        normalized_shoulder_data=shoulder_x,
        filtered_wrist_data=filtered_wrist,
        filtered_elbow_data=filtered_elbow,
        filtered_shoulder_data=filtered_shoulder
    )
