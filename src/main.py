import argparse
from pipeline import TrackingPipeline

def parse_arguments():
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
        required=True,
        help="Path to the source video file",
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
        default=0.3,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold",
        default=0.7,
        help="IOU threshold for the model",
        type=float,
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    pipeline = TrackingPipeline(
        source_weights_path=args.source_weights_path,
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
    )
    pipeline.process_video()
    
    # cap = cv2.VideoCapture(args.source if args.source != "0" else 0)
    # while cap.isOpened():
    #     success, frame = cap.read()
    #     if not success:
    #         break
        
    #     processed_frame = pipeline.process_frame(frame)
    #     cv2.imshow("Object Tracking", processed_frame)
        
    #     if cv2.waitKey(1) == ord("q"):
    #         break
    
    # cap.release()
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()