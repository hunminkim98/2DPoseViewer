"""
Utility functions for pose visualization.
"""

from typing import List, Tuple, Dict
from PySide6.QtGui import QColor


# 신체 부위별 색상
BODY_PART_COLORS = {
    'right': QColor("#FF6B6B"),     # 빨강계열 - 오른쪽
    'left': QColor("#4ECDC4"),      # 청록계열 - 왼쪽
    'center': QColor("#FFE66D"),    # 노랑계열 - 중심
    'face': QColor("#DDA0DD"),      # 보라계열 - 얼굴
    'hand': QColor("#95E1D3"),      # 연청록 - 손
    'foot': QColor("#FF8E53"),      # 주황계열 - 발
}


# 모델별 추가 연결 (트리 구조에 없는 교차 연결)
EXTRA_CONNECTIONS = {
    "BODY_25": [
        ("LHip", "LShoulder", "left"),
        ("RHip", "RShoulder", "right"),
        ("LHip", "RHip", "center"),
        ("LShoulder", "RShoulder", "center"),
    ],
    "COCO_17": [
        ("LHip", "LShoulder", "left"),
        ("RHip", "RShoulder", "right"),
        ("LHip", "RHip", "center"),
        ("LShoulder", "RShoulder", "center"),
    ],
    "COCO": [
        ("LHip", "LShoulder", "left"),
        ("RHip", "RShoulder", "right"),
        ("LHip", "RHip", "center"),
        ("LShoulder", "RShoulder", "center"),
    ],
    "HALPE_26": [
        ("LHip", "LShoulder", "left"),
        ("RHip", "RShoulder", "right"),
        ("LHip", "RHip", "center"),
        ("LShoulder", "RShoulder", "center"),
    ],
    "COCO_133": [
        ("LHip", "LShoulder", "left"),
        ("RHip", "RShoulder", "right"),
        ("LHip", "RHip", "center"),
        ("LShoulder", "RShoulder", "center"),
    ],
    "COCO_133_WRIST": [
        ("LHip", "LShoulder", "left"),
        ("RHip", "RShoulder", "right"),
        ("LHip", "RHip", "center"),
        ("LShoulder", "RShoulder", "center"),
    ],
}


def get_skeleton_connections(skeleton_root, model_name: str = None) -> List[Tuple[int, int, str]]:
    """
    스켈레톤 트리에서 연결 정보를 추출
    Returns: [(parent_id, child_id, body_part), ...]
    """
    connections = []
    name_to_id = {}
    
    def traverse(node, parent_id=None):
        current_id = node.id
        
        if current_id is not None:
            name_to_id[node.name] = current_id
        
        name = node.name.lower()
        if any(x in name for x in ['eye', 'ear', 'nose', 'head', 'jaw', 'mouth', 'brow']):
            part = 'face'
        elif any(x in name for x in ['thumb', 'index', 'middle', 'ring', 'pinky']) and 'toe' not in name:
            part = 'hand'
        elif any(x in name for x in ['toe', 'heel', 'foot', 'ankle']):
            part = 'foot'
        elif name.startswith('r') or 'right' in name:
            part = 'right'
        elif name.startswith('l') or 'left' in name:
            part = 'left'
        else:
            part = 'center'
        
        if parent_id is not None and current_id is not None:
            connections.append((parent_id, current_id, part))
        
        for child in node.children:
            traverse(child, current_id if current_id is not None else parent_id)
    
    traverse(skeleton_root)
    
    # 추가 연결 (torso box) 추가
    if model_name and model_name in EXTRA_CONNECTIONS:
        for joint1_name, joint2_name, part in EXTRA_CONNECTIONS[model_name]:
            id1 = name_to_id.get(joint1_name)
            id2 = name_to_id.get(joint2_name)
            if id1 is not None and id2 is not None:
                connections.append((id1, id2, part))
    
    return connections


def get_keypoint_ids(skeleton_root) -> Dict[int, str]:
    """스켈레톤 트리에서 키포인트 ID와 이름 매핑 추출"""
    keypoint_names = {}
    
    def traverse(node):
        if node.id is not None:
            keypoint_names[node.id] = node.name
        for child in node.children:
            traverse(child)
    
    traverse(skeleton_root)
    return keypoint_names


def get_person_color(person_idx: int) -> QColor:
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
