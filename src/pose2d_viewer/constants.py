"""
Constants and utility functions for pose visualization.
"""

from typing import List, Tuple, Dict
from PySide6.QtGui import QColor

from .skeletons import (
    HALPE_26, COCO_133, COCO_133_WRIST, COCO_17, HAND_21, 
    BODY_25B, BODY_25, BODY_135, BLAZEPOSE,
    HALPE_68, HALPE_136, COCO, MPII
)


# 사용 가능한 스켈레톤 모델들
SKELETON_MODELS = {
    "BODY_25": BODY_25,
    "COCO_17": COCO_17,
    "COCO": COCO,
    "HALPE_26": HALPE_26,
    "HALPE_68": HALPE_68,
    "HALPE_136": HALPE_136,
    "BODY_25B": BODY_25B,
    "BODY_135": BODY_135,
    "COCO_133": COCO_133,
    "COCO_133_WRIST": COCO_133_WRIST,
    "BLAZEPOSE": BLAZEPOSE,
    "MPII": MPII,
}

# 신체 부위별 색상
BODY_PART_COLORS = {
    'right': QColor("#FF6B6B"),
    'left': QColor("#4ECDC4"),
    'center': QColor("#FFE66D"),
    'face': QColor("#DDA0DD"),
    'hand': QColor("#95E1D3"),
    'foot': QColor("#FF8E53"),
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
