"""Traffic Control System - Main Entry Point.

This module serves as the main entry point for the Traffic Control System.
It provides command-line interface for running the vehicle tracking and
analysis pipeline.

The system uses YOLO for object detection and ByteTrack for object tracking,
along with additional models for vehicle color classification and emergency
vehicle detection.

Example:
    To run the system:
        python main.py --source_weights_path models/yolov8n.pt \
                      --source_video_path input.mp4 \
                      --target_video_path output.mp4 \
                      --confidence_threshold 0.5 \
                      --iou_threshold 0.8
"""

import argparse

from pipeline import TrackingPipeline


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the traffic control system.

    This function sets up the argument parser with all necessary parameters
    for configuring the tracking pipeline.

    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - source_weights_path (str): Path to YOLO model weights
            - source_video_path (str): Path to input video (default: 0)
            - target_video_path (str): Path to save output video
            - confidence_threshold (float): Detection confidence threshold
            - iou_threshold (float): IOU threshold for non-max suppression
    """
    parser = argparse.ArgumentParser(
        description="Traffic Flow Analysis with YOLO and ByteTrack"
    )

    parser.add_argument(
        "--source_weights_path",
        required=True,
        help="Path to the source weights file",
        type=str,
    )
    parser.add_argument(
        "--source_video_path",
        default=0,
        help="Path to the source video file (use 0 for webcam)",
        type=str,
    )
    parser.add_argument(
        "--target_video_path",
        default=None,
        help="Path to the target video file (output)",
        type=str,
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.5,
        help="Confidence threshold for the model (default: 0.5)",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold",
        default=0.8,
        help="IOU threshold for the model (default: 0.8)",
        type=float,
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the Traffic Control System.

    This function:
    1. Parses command line arguments
    2. Initializes the tracking pipeline
    3. Processes the video stream
    4. Handles the output visualization

    The system will process the input video (or webcam stream) and display
    real-time tracking results with vehicle classifications and movement
    states.
    """
    args = parse_arguments()
    
    pipeline = TrackingPipeline(
        source_weights_path=args.source_weights_path,
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
    )
    pipeline.process_video()


if __name__ == "__main__":
    main()