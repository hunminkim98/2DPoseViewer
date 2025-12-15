"""
pose2d_viewer - 2D Pose Visualization Tool
A PySide6-based tool for visualizing 2D pose estimation data.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == "Keypoint":
        from .models import Keypoint
        return Keypoint
    elif name == "Person":
        from .models import Person
        return Person
    elif name == "FrameData":
        from .models import FrameData
        return FrameData
    elif name == "SKELETON_MODELS":
        from .app import SKELETON_MODELS
        return SKELETON_MODELS
    elif name == "PoseViewerWindow":
        from .app import PoseViewerWindow
        return PoseViewerWindow
    elif name == "detect_rowing_strokes":
        from .stroke_detector import detect_rowing_strokes
        return detect_rowing_strokes
    elif name == "StrokeAnalysisResult":
        from .stroke_detector import StrokeAnalysisResult
        return StrokeAnalysisResult
    elif name == "StrokeEvent":
        from .stroke_detector import StrokeEvent
        return StrokeEvent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Keypoint",
    "Person", 
    "FrameData",
    "SKELETON_MODELS",
    "PoseViewerWindow",
    "detect_rowing_strokes",
    "StrokeAnalysisResult",
    "StrokeEvent",
    "__version__",
]


def main():
    """Entry point for the application."""
    from .app import run_app
    run_app()

