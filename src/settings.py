from pathlib import Path
import sys

# Get the absolute path of the current file
FILE = Path(__file__).resolve()

# Get the parent directory (project root)
ROOT = FILE.parent.parent

# Add the root directory to Python path if not already present
# This allows importing modules from the project root
if ROOT not in sys.path:
    sys.path.append(str(ROOT))

# Application source type constants
IMAGE = 'Image'
SOURCES_LIST = [IMAGE]

# Directory paths for project resources
IMAGES_DIR = ROOT / 'images'

# Default images for the application
DEFAULT_IMAGE = IMAGES_DIR / 'tooth-decay-image.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'detected-image.jpg'

# Model weights directory
MODEL_DIR = ROOT / 'weights'

# Pre-trained model paths
DETECTION_MODEL = MODEL_DIR / 'best.pt'  # Custom trained detection model
SEGMENTATION_MODEL = MODEL_DIR / 'best.pt'  # YOLOv8 segmentation model