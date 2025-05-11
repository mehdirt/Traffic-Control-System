# Traffic Control System

A sophisticated computer vision-based traffic monitoring and control system leverages state-of-the-art object detection and tracking algorithms to analyze and manage traffic flow in real-time.

## 🚀 Features

- Real-time object detection and tracking using YOLO and ByteTrack
- Vehicle color classification using a custom ONNX model
- Object movement state tracking and history
- Real-time video annotation with bounding boxes and labels
- Object counting and classification
- Customizable confidence and IOU thresholds for detection
- Support for both video files and live camera feeds
- Detailed tracking history for each detected object

## 🛠️ Technologies Used

- **YOLOv8**: For high-performance object detection
- **Supervision**: For advanced object tracking and analysis
- **OpenCV**: For image processing and computer vision tasks
- **NumPy**: For numerical computations and array operations
- **Python**: As the primary programming language

## 📋 Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for better performance)
- Webcam or IP camera for real-time monitoring

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/traffic-control-system.git
cd traffic-control-system
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## 🚦 Usage

1. Run the main application:
```bash
python main.py --source_weights_path models/yolov8n.pt --source_video_path input.mp4 --target_video_path output.mp4 --confidence_threshold 0.5 --iou_threshold 0.8
```

2. Configure the system using the config files in `src/config/`

3. The system will start processing video input and provide real-time traffic analysis

## 📊 Project Structure

```
traffic-control-system/
├── src/
│   ├── main.py           # Main application entry point
│   ├── pipeline.py       # Core processing pipeline
│   └── config/           # Configuration files
├── models/               # Trained models
├── data/                 # Dataset and training data
├── assets/               # Static assets and resources
└── requirements.txt      # Project dependencies
```

## 🤝 Contributing

--
## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- YOLOv8 team for the excellent object detection model
- Supervision team for the tracking and analysis tools
- OpenCV community for the computer vision library
