from typing import Optional

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

from config.tracking_config import TRACKING_CONFIG
from config.annotations import ANNOTATION_CONFIG


class TrackingPipeline:
    def __init__(
        self,
        source_weights_path: str,
        source_video_path: str,
        target_video_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.7,
    ) -> None:
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path
        self.track_history = {}
        # Model
        self.model = YOLO(source_weights_path)
        # Tracker
        self.tracker = sv.ByteTrack(frame_rate=TRACKING_CONFIG['frame_rate'])
        # Supervision Video Information
        self.video_info = sv.VideoInfo.from_video_path(source_video_path)
        # Box Annotator
        self.box_annotator = sv.BoxAnnotator(
            color=ANNOTATION_CONFIG['color_palette'],
            thickness=ANNOTATION_CONFIG["box_thickness"],
            )
        # Label Annotator
        self.label_annotator = sv.LabelAnnotator(
            color=ANNOTATION_CONFIG['color_palette'], text_color=sv.Color.BLACK
        )
        # Trace Annotator
        # self.trace_annotator = sv.TraceAnnotator(
        #     color=ANNOTATION_CONFIG['color_palette'],
        #     position=sv.Position.CENTER,
        #     trace_length=100,
        #     thickness=ANNOTATION_CONFIG['tracking_line_thickness'],
        # )

    def process_video(self):
        """
        """
        # if self.target_video_path:
        #     with sv.VideoSink(self.target_video_path, self.video_info) as sink:
        #         for frame in tqdm(frame_generator, total=self.video_info.total_frames):
        #             annotated_frame = self.process_frame(frame)
        #             sink.write_frame(annotated_frame)
        # else:
        #     for frame in tqdm(frame_generator, total=self.video_info.total_frames):
        #         annotated_frame = self.process_frame(frame)
        #         cv2.imshow("Processed Video", annotated_frame)
        #         if cv2.waitKey(1) & 0xFF == ord("q"):
        #             break
        #     cv2.destroyAllWindows()
        
        cap = cv2.VideoCapture(self.source_video_path if self.source_video_path != "0" else 0)
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            annotated_frame = self.process_frame(frame)
            cv2.imshow("Object Tracking", annotated_frame)
            
            if cv2.waitKey(1) == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

    def annotate_frame(
        self, frame: np.ndarray, detections: sv.Detections
    ) -> np.ndarray:
        """
        """
        annotated_frame = frame.copy()

        labels = [
            f"#{track_id} {ANNOTATION_CONFIG['class_mapping'][class_id]} {conf:.2f}"
            for track_id, class_id, conf in zip(
                detections.tracker_id, 
                detections.class_id, 
                detections.confidence
            )
        ]

        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels
        )
        annotated_frame = self.box_annotator.annotate(
            scene=annotated_frame,
            detections=detections
        )
        # annotated_frame = self.trace_annotator.annotate(
        #     scene=annotated_frame,
        #     detections=detections
        # )

        return annotated_frame
    
    def update_tracking_history(self, detections: sv.Detections):
        """
        """
        for tracker_id, [x1, y1, x2, y2] in zip(detections.tracker_id, detections.xyxy):
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            self.track_history.setdefault(tracker_id, []).append(center)
            
            # Keep history length limited
            if len(self.track_history[tracker_id]) > ANNOTATION_CONFIG["tracking_history_length"]:
                self.track_history[tracker_id].pop(0)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        """
        results = self.model(
            source=frame,
            verbose=False,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=TRACKING_CONFIG["allowed_classes"],
        )[0]
        
        detections = sv.Detections.from_ultralytics(results)
         
        detections = self.tracker.update_with_detections(detections)
        
        # Debug: Print detections structure
        print("Detections:", detections)
        print("Tracker IDs:", detections.tracker_id)
        print("Class IDs:", detections.class_id)
        print("Confidence:", detections.confidence)
        print("Bounding Boxes:", detections.xyxy)

        # Update history
        self.update_tracking_history(detections)
        
        return self.annotate_frame(frame, detections)

