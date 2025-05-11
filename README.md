# Traffic Object Detection

A sophisticated computer vision-based traffic monitoring and control system leveraging state-of-the-art object detection and tracking algorithms to analyze and manage traffic flow in real-time.

## 🚀 Features

- Real-time Object Detection: Utilizes YOLOv8 for high-performance detection of vehicles and other objects.
- Object Tracking: Employs ByteTrack via the Supervision library for accurate tracking of detected objects.
- Vehicle Color Classification: Uses a custom ONNX model to classify the color of detected vehicles.
- Movement State Tracking: Monitors and records the movement states and histories of objects.
- Video Annotation: Provides real-time annotation of video feeds with bounding boxes and labels.
- Object Counting and Classification: Counts and classifies objects based on detection results.
- Customizable Thresholds: Allows adjustment of confidence and IOU thresholds for detection.
- Input Flexibility: Supports both video files and live camera feeds.
- Tracking History: Maintains detailed tracking history for each detected object.

## 🛠️ Technologies Used

- **YOLOv8**: For object detection, provided by the Ultralytics library.
- **Supervision**: For advanced object tracking and analysis.
- **OpenCV**: For image processing and computer vision tasks.
- **NumPy**: For numerical computations and array operations.
- **Python**: As the primary programming language.

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
To see all available options, run:
```bash
python main.py --help
```

2. Configure the system using the config files in `src/config/`

3. The system will start processing video input and provide real-time traffic analysis

## Project Structure

| Directory/File       | Description                              |
|----------------------|------------------------------------------|
| `src/`               | Contains core application files          |
| `src/main.py`        | Main application entry point             |
| `src/pipeline.py`    | Core processing pipeline                 |
| `src/config/`        | Configuration files                      |
| `models/`            | Trained models                           |
| `data/`              | Dataset and training data                |
| `assets/`            | Static assets and resources              |
| `requirements.txt`   | Project dependencies                     |

## Dependencies

| Dependency              | Version       |
|-------------------------|---------------|
| ultralytics             | 8.3.84        |
| supervision             | 0.25.1        |
| opencv-python-headless  | >=4.10.0      |
| numpy                   | >=1.20.0      |
| argparse                | >=1.4.0       |

## Related Work

In addition to this project, I have trained an RF-DETR object detection model on [a custom traffic dataset](https://app.roboflow.com/cvision-pulcl/traffic-object-detection-etcoy/) that I annotated myself. The model achieved a mean Average Precision (mAP) of 52% and is maintained on the Roboflow platform. This separate effort explores alternative object detection approaches for traffic monitoring and is not included in this repository.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact Information
If you have any feedback, feel free to reach out.

[![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:mahdirafati680@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mahdi-rafati-97420a197/)
[![Medium](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@mehdirt)
[![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://x.com/itsmehdirt)
