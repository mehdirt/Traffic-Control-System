import supervision as sv

ANNOTATION_CONFIG = {
    "box_thickness": 2,
    "text_scale": 0.5,
    "text_thickness": 1,
    "color_palette": sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1", "#d13cb8", "#3cd1cf"]),
    "tracking_line_thickness": 2,
    "tracking_history_length": 30,
    "class_mapping": {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    },
    "color_classes": [
        "beige", "black", "blue", "brown", "gold",
        "green", "grey", "orange", "pink", "purple",
        "red", "silver", "tan", "white", "yellow",
    ],
}