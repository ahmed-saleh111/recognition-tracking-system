# Nursery Recognition & Tracking System

A comprehensive computer vision system designed for nursery environments that performs real-time person detection, tracking, and face recognition to identify kids and caregivers.

## 🎯 Features

- **Person Detection & Classification**: Detect and classify individuals as "Kid" or "Caregiver"
- **Multi-Object Tracking**: Track multiple people across video frames using ByteTrack
- **Face Recognition**: Identify specific individuals using Siamese neural networks
- **Real-time Processing**: Process video streams with configurable frame rates
- **Voting System**: Improve recognition accuracy using temporal voting across frames
- **Video Output**: Generate annotated videos with bounding boxes and identity labels

## 🏗️ System Architecture

The system consists of several key components:

- **Detection & Tracking**: Uses YOLO models for person detection and ByteTrack for tracking
- **Face Recognition**: Combines face detection with similarity matching using trained models
- **Person Recognition**: Integrates face detection with person identification
- **Video Processing**: Handles video input/output and frame-by-frame processing

## 📋 Requirements

### System Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended) or CPU
- OpenCV-compatible camera or video files

### Dependencies
All required packages are listed in `requirements.txt`. Key dependencies include:
- PyTorch & TorchVision
- Ultralytics (YOLO)
- OpenCV
- NumPy, PIL, Matplotlib
- Additional ML libraries (see requirements.txt for full list)

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd nursery
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv nursery_env

# Activate virtual environment
# On Windows:
nursery_env\Scripts\activate
# On macOS/Linux:
source nursery_env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
---

### 4. Download Required Models
Ensure the following model files are in the `models/` directory:
- `detection_model.pt` - Person detection model
- `face_detection_model.pt` - Face detection model
- `siamese_model.pt` - Face recognition model, you can fine tune it in your custom dataset to increase the accuracy.
- `yolo11n-cls.pt` - the model used for fine tuning in `data_prep_finetune.ipynb`

📥 **Download the pre-trained models using the links above.**

| Model File                                                                                                         | Description                              | Usage                             |
|-------------------------------------------------------------------------------------------------------------------|------------------------------------------|-----------------------------------|
| [`detection_model.pt`](https://drive.google.com/uc?export=download&id=1g2BkV51N7z1VUebRbjQDClid0ScnAIre)         | Person detection model (Kids/Caregivers) | Main detection and tracking       |
| [`face_detection_model.pt`](https://drive.google.com/uc?export=download&id=1NHsNWBaJbUF1M6ZhB0PsUvirjf3SJQrR)    | Face detection model                     | Face localization for recognition |
| [`siamese_model.pt`](https://drive.google.com/uc?export=download&id=1bNKpEhMbiiVlqE07zMOD1iW58K8IU0_W)           | Fine-tuned face recognition model        | Person identification             |
| [`yolo11n-cls.pt`](https://drive.google.com/uc?export=download&id=1kEclCloS_v-oorjh2ho3mnQSgZYOn0LT)             | YOLO classification model                | Fine-tuning on custom datasets    |

---

**📋 Instructions:**

1. Click on the links above to download each model.
2. Save all `.pt` files into the `models/` directory.
3. Ensure file names match exactly as listed.

**⚠️ Important Notes:**

* `yolo11n-cls.pt` is required if you plan to fine-tune on your custom dataset.

---

### 5. Prepare Dataset
The `dataset/` directory should contain subdirectories for each person to be recognized:
```
dataset/
├── person_1/
│   └── img_1.jpg
├── perso_2/
│   └── img_1.jpg
├── ...
```

## 🎮 Usage

### Running the Test Script

The main entry point is `app.py`, which processes a video file and generates an annotated output.

```bash
python app.py
```

### Configuration Options

You can modify the following parameters in `app.py`:

```python
# Model paths
detection_model = 'models/detection_model.pt'
yolo_face = 'models/face_detection_model.pt'
person_model = 'models/siamese_model.pt'

# Data and video paths
person_data = 'dataset'
video_path = 'videos/video.mp4'
output_video_path = 'videos/output_annotated.mp4'

# Processing parameters
threshold = 0.9              # Recognition confidence threshold
conf = 0.5                   # Detection confidence threshold
voting_frame_window = 60     # Frames for temporal voting
camera_fps = 20              # Processing frame rate
recognition_spf = 0.1        # Recognition frequency (seconds per frame)

# Display options
show = True                  # Show real-time video window
device = 'cuda'              # Use 'cpu' for CPU-only processing
```
![Result](out_put.gif)


## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Set `device = 'cpu'` in test.py
   - Reduce batch sizes or image resolution

2. **Model File Not Found**
   - Ensure all model files are in the `models/` directory
   - Check file paths in the configuration

3. **Video File Issues**
   - Verify video file exists in `videos/` directory
   - Ensure video format is supported by OpenCV

4. **Recognition Accuracy Issues**
   - Adjust `threshold` parameter (lower = more sensitive)
   - Increase `voting_frame_window` for more stable results
   - Add more reference images to the dataset

### Performance Optimization

- **GPU Usage**: Ensure CUDA is properly installed and `device='cuda'`
- **Frame Rate**: Adjust `camera_fps` and `recognition_spf` based on your hardware
- **Recognition Frequency**: Increase `recognition_spf` to reduce computational load

## 📊 Output

The system generates:
- **Annotated Video**: Video with bounding boxes and identity labels
- **Console Output**: Real-time processing information
- **Recognition Results**: Person identities and tracking information

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Dependencies Attribution

This project uses several open-source libraries:
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for object detection
- [PyTorch](https://pytorch.org/) for deep learning
- [OpenCV](https://opencv.org/) for computer vision
- [ByteTrack](https://github.com/ifzhang/ByteTrack) for object tracking

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the code comments for implementation details
3. Create an issue in the repository



