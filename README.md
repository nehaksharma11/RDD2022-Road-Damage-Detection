# RDD2022-Road-Damage-Detection 

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00D4AA.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An AI-powered road damage detection system built for the Crackathon hackathon. The model automatically detects and classifies five types of road damage using computer vision and deep learning.

## Project Overview

Road infrastructure maintenance is critical for public safety and economic efficiency. This project leverages state-of-the-art object detection to automate the identification of road damage, enabling faster and more cost-effective infrastructure monitoring.

### Key Features
-  Detects 5 types of road damage
-  Real-time inference capability
-  78.11% mAP50 accuracy
-  Trained on 47,000+ road images
-  Built with YOLOv8 (state-of-the-art)

## Performance Metrics

| Metric | Score | Description |
|--------|-------|-------------|
| mAP@0.5 | **78.11%** | Main detection accuracy |
| mAP@0.5-0.95 | **49.03%** | Strict IoU metric |
| Precision|** 74.09%** | Correct detections |
| Recall | **72.77%** | Detection coverage |

## Model Architecture

- **Framework**: YOLOv8 (Ultralytics)
- **Model Variant**: YOLOv8 Medium
- **Input Size**: 640×640 pixels
- **Parameters**: ~25M
- **Speed**: ~45 FPS on GPU

## Dataset

**RDD2022 (Road Damage Detection 2022)**
- Training Images: ~47,000
- Classes: 5 damage types
- Format: YOLO annotation format
- Source: Multi-national road images

### Damage Classes
1. **Longitudinal Crack** - Linear cracks parallel to road direction
2. **Transverse Crack** - Linear cracks perpendicular to road direction  
3. **Alligator Crack** - Interconnected cracks forming patterns
4. **Other Corruption** - Surface deterioration and wear
5. **Pothole** - Bowl-shaped depressions in pavement

## Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/nehaksharma11/RDD2022-Road-Damage-Detection.git
cd RDD2022-Road-Damage-Detection

# Install dependencies
pip install -r requirements.txt
```

### Requirements
```txt
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
matplotlib>=3.7.0
PyYAML>=6.0
```

### Inference
```python
from ultralytics import YOLO

# Load trained model
model = YOLO('models/best_model.pt')

# Predict on single image
results = model.predict('path/to/image.jpg', conf=0.25)

# Display results
results[0].show()
```

### Batch Prediction
```python
# Predict on folder of images
results = model.predict('path/to/images/', save=True)
```

##  Training Details

### Hyperparameters
- **Epochs**: 50
- **Batch Size**: 16
- **Image Size**: 640×640
- **Optimizer**: AdamW
- **Learning Rate**: 0.01 (initial)
- **Patience**: 15 (early stopping)

### Data Augmentation
- HSV Color Space Augmentation
- Horizontal Flip (50%)
- Mosaic Augmentation
- Scale & Translation
- Random Brightness/Contrast

### Training Time
- **Total Time**: ~4 hours
- **Hardware**: Kaggle GPU T4
- **Framework**: PyTorch + Ultralytics

##  Project Structure
```
RDD2022-Road-Damage-Detection/
├── README.md
├── requirements.txt
├── report.pdf
├── models/
│   └── best_model.pt           # Trained model weights (50MB)
├── notebooks/
│   └── training_notebook.ipynb # Kaggle training notebook
├── submission/
│   └── submission.zip          # Competition submission
├── docs/
│   ├── metrics.txt            # Performance metrics
│   └── data.yaml              # Dataset configuration
├── scripts/
│   ├── train.py               # Training script
│   ├── predict.py             # Inference script
└── └── evaluate.py            # Evaluation script
```

##  Methodology

### 1. Data Preprocessing
- Image resizing to 640×640
- YOLO format label conversion
- Train/validation split (80/20)

### 2. Model Training
- Transfer learning from COCO pre-trained weights
- Progressive learning rate scheduling
- Data augmentation for robustness

### 3. Evaluation
- Validation on held-out dataset
- Metrics: mAP, precision, recall
- Per-class performance analysis

### 4. Post-Processing
- Non-Maximum Suppression (NMS)
- Confidence threshold: 0.25
- IoU threshold: 0.45

## Analysis & Insights

### What Worked Well
- YOLOv8 Medium balanced speed and accuracy  
- Mosaic augmentation improved small object detection  
- Pre-trained weights accelerated convergence  
- Early stopping prevented overfitting  

### Challenges
-> Class imbalance (some damage types rarer)  
-> Variable lighting conditions  
-> Small damage instances difficult to detect  

### Future Improvements
* Ensemble multiple models for better accuracy  
* Test-time augmentation (TTA)  
* Attention mechanisms for small objects  
* Post-processing refinement  
* Multi-scale prediction  

## Usage Examples

### Example 1: Single Image Prediction
```python
from ultralytics import YOLO

model = YOLO('models/best_model.pt')
result = model.predict('road_image.jpg')

# Get bounding boxes
boxes = result[0].boxes
for box in boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    print(f"Detected: {model.names[class_id]} ({confidence:.2f})")
```

### Example 2: Video Processing
```python
# Process video frame by frame
results = model.predict('road_video.mp4', save=True, conf=0.3)
```

### Example 3: Real-time Webcam
```python
# Real-time detection from webcam
model.predict(source=0, show=True)
```

## References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [RDD2022 Dataset](https://github.com/sekilab/RoadDamageDetector)
- [Object Detection Paper](https://arxiv.org/abs/2304.00501)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Team

- Neeraj Upadhayay - Team Leader
- Neha Kumari - Team Member
- Ronak Ramsuyash Thakur - Team Member
- Mahak Yadav - Team Member

## Contact

For questions or collaboration:
- Email: neha.kumari11101@gmail.com
- LinkedIn: https://www.linkedin.com/in/neha-kumari-3184841ab/
- GitHub: [@nehaksharma11](https://github.com/nehaksharma11)

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- Thanks to the organizers of Crackathon hackathon
- RDD2022 dataset creators
- Ultralytics team for YOLOv8
- Kaggle for providing GPU resources

##  Citation

If you use this work, please cite:
```bibtex
@misc{roadwise2026,
  title={RDD2022 Road Damage Detection using YOLOv8},
  author={Neha Kumari},
  year={2026},
  publisher={GitHub},
  url={https://github.com/nehaksharma11/RDD2022-Road-Damage-Detection}
}
```

---

** If you find this project helpful, please give it a star!**

Made with ❤️ for safer roads
```
