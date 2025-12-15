"""
Data models for pose visualization.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class Keypoint:
    """단일 키포인트 데이터"""
    x: float
    y: float
    confidence: float
    
    @property
    def is_valid(self) -> bool:
        return self.x > 0 and self.y > 0 and self.confidence > 0


@dataclass
class Person:
    """한 사람의 포즈 데이터"""
    person_id: int
    keypoints: List[Keypoint]


@dataclass
class FrameData:
    """한 프레임의 전체 데이터"""
    frame_number: int
    people: List[Person]
