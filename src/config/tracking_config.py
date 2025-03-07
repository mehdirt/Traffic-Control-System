TRACKING_CONFIG = {
    # Tracking params
    # "tracker_type": "bytetrack",
    # "track_thresh": 0.5, #!
    # "track_buffer": 100,
    # "match_thresh": 0.8,
    # "frame_rate": 30,

    # Detection params
    "conf_thresh": 0.4,      # New: Minimum detection confidence
    "iou_thresh": 0.45,  
    "allowed_classes": [0, 1, 2, 3, 5, 7]
}