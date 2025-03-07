import supervision as sv

ANNOTATION_CONFIG = {
    "box_thickness": 2,
    "text_scale": 0.5,
    "text_thickness": 1,
    "color_palette": sv.ColorPalette.DEFAULT,
    "tracking_line_thickness": 2,
    "tracking_history_length": 30,
    "class_mapping": {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }
}