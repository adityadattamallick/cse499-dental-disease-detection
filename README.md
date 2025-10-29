# Dental Disease Detection using YOLOv8

## CSE499 - Senior Design Capstone Project

A deep learning-based system for automated detection and localization of dental diseases in oral photographs using YOLOv8 object detection framework.

## Overview

This project implements a state-of-the-art dental disease detection system using PyTorch and YOLOv8. The system is capable of identifying and localizing various dental conditions from custom dental photographs, providing an automated solution to assist dental professionals in diagnosis and treatment planning.

## Features

- **Multi-disease Detection**: Detects various dental diseases including caries, gingivitis, calculus, and other oral conditions
- **Real-time Inference**: Fast detection using optimized YOLOv8 architecture
- **Custom Dataset**: Trained on a curated dataset of dental photographs
- **YOLOv8 Format**: Utilizes industry-standard YOLOv8 annotation format for training
- **Pre-trained Checkpoint**: Includes best performing model checkpoint for immediate deployment

## Technical Architecture

- **Framework**: PyTorch
- **Model**: YOLOv8 (You Only Look Once v8)
- **Input**: Custom dental photographs (RGB images)
- **Annotation Format**: YOLOv8 format (converted from JSON to TXT labels)
- **Task**: Object Detection and Localization

<!--## Project Structure-->

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/adityadattamallick/cse499-dental-disease-detection.git
cd cse499-dental-disease-detection
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the model on your custom dataset:

```bash
python train.py --data config.yaml --epochs 100 --batch 16 --img 640
```

### Prediction on Single Image

```python
from ultralytics import YOLO

# Load the trained model
model = YOLO('best.pt')

# Run inference
results = model.predict('path/to/dental/image.jpg', conf=0.3)

# Display results
results[0].show()
```

## Dataset

- **Format**: YOLOv8 annotation format
- **Initial Format**: JSON labels (converted to TXT)
- **Image Type**: Custom dental photographs
- **Classes**: Multiple dental disease categories

The dataset follows YOLOv8 annotation standards where each image has a corresponding `.txt` file with normalized bounding box coordinates:
```
class_id center_x center_y width height
```

## Dataset Details

The dataset comprises multiple classes of dental conditions and diseases. Below are the detailed class mappings used in this project:

<div align="center">
  <img src="dataset-class-mapping/dataset_class_table-1.jpg" alt="Dataset Class Mapping Table 1" width="800"/>
  <br/>
  <em>Dataset Class Mapping - Part 1</em>
  <br/><br/>
  
  <img src="dataset-class-mapping/dataset_class_table-2.jpg" alt="Dataset Class Mapping Table 2" width="800"/>
  <br/>
  <em>Dataset Class Mapping - Part 2</em>
</div>

These tables provide comprehensive information about the disease classes, their identifiers, and categorization used for training and evaluation.

## Model Performance

- **Metrics**
  - mAP@0.5
  - mAP@0.5:0.95

## Methodology

1. **Data Collection**: Curated custom dataset of dental photographs
2. **Annotation**: Initial JSON format annotations converted to YOLOv8 TXT format
3. **Preprocessing**: Image normalization and augmentation
4. **Training**: YOLOv8 model trained with PyTorch backend
5. **Validation**: Cross-validation and testing on holdout set
6. **Deployment**: Optimized checkpoint for inference

## Requirements

See `requirements.txt` for complete list of dependencies. Key libraries include:

- PyTorch
- Ultralytics (YOLOv8)
- OpenCV
- NumPy
- Matplotlib
- Pillow

## Contributing

This is a Senior Design Capstone project. For collaboration or questions, please open an issue or contact the project maintainer.

## Contact

**Project Author**: Aditya Narayan Datta Mallick  
**Institution**: North South University, Dhaka, Bangladesh<br /> 
**Course**: CSE499 - Senior Design & Capstone Project

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Aditya Narayan Datta Mallick

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

## Acknowledgments

- YOLOv8 by Ultralytics
- PyTorch team
- Saif Ahmed
- Dr Nabeel Mohammed

## References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Streamlit Cropper Tool](https://github.com/turner-anderson/streamlit-cropper)

---

