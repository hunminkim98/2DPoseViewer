# pose2d-viewer

A PySide6-based tool for visualizing 2D pose estimation data from various pose estimation frameworks.

## Features

- **Multi-model Support**: BODY_25, COCO_17, HALPE_26, COCO_133, and more
- **Rich Visualization**: Customizable keypoint size, skeleton width, and opacity
- **Playback Controls**: Frame-by-frame navigation with adjustable FPS
- **Interactive Selection**: Click on bounding boxes to select individual persons
- **Filtering**: Apply Butterworth, Gaussian, or Median filters to smooth pose data
- **Hover Effects**: Visual feedback when hovering over detected persons

## Installation

```bash
pip install pose2d-viewer
```

### From Source

```bash
git clone https://github.com/hunminkim98/2DPoseViewer.git
cd 2DPoseViewer
pip install -e .
```

## Quick Start

### Command Line

```bash
pose2d-viewer
```

### Python API

```python
from pose2d_viewer import PoseViewerWindow
from PySide6.QtWidgets import QApplication
import sys

app = QApplication(sys.argv)
window = PoseViewerWindow()
window.show()
sys.exit(app.exec())
```

## Supported Formats

The viewer supports JSON files in OpenPose format:

```json
{
  "people": [
    {
      "pose_keypoints_2d": [x1, y1, c1, x2, y2, c2, ...]
    }
  ]
}
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Play/Pause |
| `←` / `→` | Previous/Next frame |
| `Home` / `End` | First/Last frame |
| `Ctrl+O` | Open folder |
| `Ctrl+Q` | Quit |

## Skeleton Models

| Model | Keypoints | Description |
|-------|-----------|-------------|
| BODY_25 | 25 | OpenPose body model |
| COCO_17 | 17 | COCO keypoint format |
| HALPE_26 | 26 | AlphaPose HALPE model |
| COCO_133 | 133 | Whole-body with hands and face |
| BLAZEPOSE | 33 | MediaPipe BlazePose |

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src tests
isort src tests
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
