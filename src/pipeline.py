import logging
from typing import Optional, List

import cv2
import numpy as np
import onnxruntime as ort
import supervision as sv
from ultralytics import YOLO

from config.tracking_config import TRACKING_CONFIG
from config.annotations import ANNOTATION_CONFIG


class TrackingPipeline:
    """A pipeline for tracking and analyzing vehicles in video streams.

    This class implements a complete pipeline for:
    - Object detection using YOLO
    - Object tracking using ByteTrack
    - Vehicle color classification
    - Emergency vehicle detection
    - Movement state tracking
    - Video annotation and visualization

    Attributes:
        conf_threshold (float): Confidence threshold for detection
        iou_threshold (float): IOU threshold for non-maximum suppression
        source_video_path (str): Path to the input video file
        target_video_path (Optional[str]): Path to save the output video
        track_history (dict): History of tracked object positions
        model (YOLO): YOLO object detection model
        tracker (sv.ByteTrack): ByteTrack object tracker
        video_info (sv.VideoInfo): Information about the input video
        box_annotator (sv.BoxAnnotator): Annotator for drawing boxes
        label_annotator (sv.LabelAnnotator): Annotator for drawing labels
        color_model (ort.InferenceSession): ONNX model for color classification
        movement_status (dict): Movement state of tracked objects
        emergency_model (ort.InferenceSession): ONNX model for emergency detection
    """

    def __init__(
        self,
        source_weights_path: str,
        source_video_path: str,
        target_video_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.7,
    ) -> None:
        """Initialize the tracking pipeline.

        Args:
            source_weights_path (str): Path to the YOLO model weights
            source_video_path (str): Path to the input video file
            target_video_path (Optional[str]): Path to save the output video
            confidence_threshold (float): Confidence threshold for detection
            iou_threshold (float): IOU threshold for non-maximum suppression
        """
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path
        self.track_history = {}
        # Track movement state per object
        self.movement_status = {}
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
        # Model: Color recognition
        self.color_model = ort.InferenceSession(
            "models/custom/color_classifier.onnx"
            )
        # Model: Emergency detector
        self.emergency_model = ort.InferenceSession(
            "models/custom/emergency_detector.onnx"
            )

    def process_video(self) -> None:
        """Process the input video stream.

        This method:
        1. Opens the video capture
        2. Processes each frame
        3. Displays the annotated results
        4. Handles user input for quitting
        """
        cap = cv2.VideoCapture(
            self.source_video_path
            if self.source_video_path != "0" else 0
            )
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

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame through the pipeline.

        This method:
        1. Runs object detection
        2. Updates tracking
        3. Predicts vehicle colors
        4. Detects emergency vehicles
        5. Updates tracking history
        6. Annotates the frame

        Args:
            frame (np.ndarray): Input frame to process

        Returns:
            np.ndarray: Processed and annotated frame
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

        # Color predictions        
        colors = [
            self.predict_color(frame, box)
            if self.is_vehicle(class_id)
            else ""
            for box, class_id in zip(detections.xyxy, detections.class_id)
        ]

        # Emergency predictions
        emergency_status = [
            self.predict_emergency(frame, box)
            if self.is_emergency_vehicle(class_id)
            else ""
            for box, class_id in zip(detections.xyxy, detections.class_id)
        ]

        # Update history
        self.update_tracking_history(detections)
        
        return self.annotate_frame(frame, detections, colors, emergency_status)

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        colors: List[str],
        emergency_status: List[str]
    ) -> np.ndarray:
        """Annotate a frame with detection and tracking information.

        Args:
            frame (np.ndarray): Input frame to annotate
            detections (sv.Detections): Detection results from YOLO
            colors (List[str]): Predicted colors for each vehicle
            emergency_status (List[str]): Emergency status for each vehicle

        Returns:
            np.ndarray: Annotated frame with bounding boxes and labels
        """
        annotated_frame = frame.copy()
        labels = [
            (
                "#{} {} {} {} {:.2f}".format(
                    track_id,
                    ANNOTATION_CONFIG['class_mapping'][class_id],
                    'EMERGENCY' if emergency_status[idx] == 'emergency'
                    else color,
                    '<MOVING>' if self.movement_status.get(track_id, False)
                    else '<STABLE>',
                    conf
                )
            )
            for idx, (track_id, class_id, conf, color) in enumerate(
                zip(
                    detections.tracker_id,
                    detections.class_id,
                    detections.confidence,
                    colors
                )
            )
        ]

        print(f"colors: {colors}")
        print(f"Detections cls_id: {detections.class_id}")

        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels
        )
        annotated_frame = self.box_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
        )

        # Count and display object counts
        counts = self.count_objects(detections)
        annotated_frame = self.display_counts(annotated_frame, counts)

        return annotated_frame
    
    def update_tracking_history(self, detections: sv.Detections) -> None:
        """Update the tracking history and movement status of objects.

        This method:
        1. Stores object centroids in tracking history
        2. Limits history length to prevent memory issues
        3. Calculates movement status based on displacement
        4. Cleans up stale tracking IDs

        Args:
            detections (sv.Detections): Current frame detections
        """
        # Store object's id and centroid
        for tracker_id, [x1, y1, x2, y2] in zip(detections.tracker_id, detections.xyxy):
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            self.track_history.setdefault(tracker_id, []).append(center)
            
            # Keep history length limited
            if len(self.track_history[tracker_id]) > ANNOTATION_CONFIG["tracking_history_length"]:
                self.track_history[tracker_id].pop(0)
            
            # Calculate movement
            if len(self.track_history[tracker_id]) >= 2:
                first_point = self.track_history[tracker_id][0]
                last_point = self.track_history[tracker_id][-1]
                displacement = (
                    (last_point[0] - first_point[0]) ** 2 +
                    (last_point[0] - first_point[0]) ** 2
                ) ** 0.5
                self.movement_status[tracker_id] = displacement > TRACKING_CONFIG['displacement_threshhold']
            else:
                self.movement_status[tracker_id] = False 
        
        # Cleanup stale tracker IDs
        active_ids = set(detections.tracker_id)

        # Remove inactive entries from movement_status
        stale_ids = set(self.movement_status.keys()) - active_ids
        for tid in stale_ids:
            del self.movement_status[tid]
        
        # Remove inactive entries from movement_status
        stale_ids = set(self.track_history.keys()) - active_ids
        for tid in stale_ids:
            del self.track_history[tid]
    
    def count_objects(self, detections: sv.Detections) -> dict:
        """Count the number of objects by class in the current frame.

        Args:
            detections (sv.Detections): Current frame detections

        Returns:
            dict: Dictionary mapping class names to their counts
        """
        counts = {}
        for class_id in detections.class_id:
            class_name = ANNOTATION_CONFIG['class_mapping'][class_id]
            counts[class_name] = counts.get(class_name, 0) + 1
        return counts

    def display_counts(self, frame: np.ndarray, counts: dict) -> np.ndarray:
        """Display object counts on the frame.

        Args:
            frame (np.ndarray): Input frame
            counts (dict): Dictionary of class names and their counts

        Returns:
            np.ndarray: Frame with counts displayed
        """
        y_position = 30
        for class_name, count in counts.items():
            text = f"{class_name}: {count}"
            cv2.putText(
                frame,
                text,
                (10, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            y_position += 30
        return frame

    def predict_color(self, frame: np.ndarray, box: np.ndarray) -> str:
        """Predict the color of a vehicle in the given bounding box.

        Args:
            frame (np.ndarray): Input frame
            box (np.ndarray): Bounding box coordinates [x1, y1, x2, y2]

        Returns:
            str: Predicted color class
        """
        x1, y1, x2, y2 = box.astype(int)
        vehicle_roi = frame[y1:y2, x1:x2]
        
        if vehicle_roi.size == 0:
            return "unknown"
        
        # Convert to PIL-style RGB
        processed = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2RGB)
        processed = cv2.resize(processed, (224, 224))
        
        # Convert to float32 and normalize
        processed = processed.astype(np.float32) / 255.0  # Critical: float32 conversion
        
        # ImageNet normalization (ensure values are float32)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        processed = (processed - mean) / std
        
        # Change dimension order to CHW + add batch dimension
        processed = processed.transpose(2, 0, 1)  # HWC to CHW
        processed = np.expand_dims(processed, axis=0)
        
        outputs = self.color_model.run(
            output_names=[self.color_model.get_outputs()[0].name],
            input_feed={self.color_model.get_inputs()[0].name: processed}
        )
        return ANNOTATION_CONFIG["color_classes"][np.argmax(outputs[0])]
    
    def is_vehicle(self, class_id: int) -> bool:
        """Check if a detected object is a vehicle.

        Args:
            class_id (int): YOLO class ID

        Returns:
            bool: True if the object is a vehicle, False otherwise
        """
        return class_id in [2, 3, 5, 7]  # cars, motorcycles, buses, trucks
    
    def predict_emergency(self, frame: np.ndarray, box: np.ndarray) -> str:
        """Predict if a vehicle is an emergency vehicle.

        Args:
            frame (np.ndarray): Input frame
            box (np.ndarray): Bounding box coordinates [x1, y1, x2, y2]

        Returns:
            str: Emergency status ('emergency' or '')
        """
        x1, y1, x2, y2 = box.astype(int)
        vehicle_roi = frame[y1:y2, x1:x2]

        if vehicle_roi.size == 0:
            return ""
        
        # Convert to PIL-style RGB
        processed = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2RGB)
        processed = cv2.resize(processed, (224, 224))
        
        # Convert to float32 and normalize
        processed = processed.astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        processed = (processed - mean) / std
        
        # Change dimension order to CHW + add batch dimension
        processed = processed.transpose(2, 0, 1)  # HWC to CHW
        processed = np.expand_dims(processed, axis=0)
        
        outputs = self.emergency_model.run(
            output_names=[self.emergency_model.get_outputs()[0].name],
            input_feed={self.emergency_model.get_inputs()[0].name: processed}
        )
        return ANNOTATION_CONFIG["emergency_classes"][np.argmax(outputs[0])]    

    def is_emergency_vehicle(self, class_id: int) -> bool:
        """Check if a vehicle class could be an emergency vehicle.

        Args:
            class_id (int): YOLO class ID

        Returns:
            bool: True if the vehicle type could be emergency, False otherwise
        """
        return class_id in [2, 5, 7]  # cars, bus-like ones, trucks
