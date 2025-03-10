import supervision as sv

ANNOTATION_CONFIG = {
    "box_thickness": 2,
    "text_scale": 0.5,
    "text_thickness": 1,
    "color_palette": sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1", "#d13cb8", "#3cd1cf"]),
    "tracking_line_thickness": 2,
    "tracking_history_length": 30,
    "class_mapping": {
        0: "Person",
        1: "Bicycle",
        2: "Car",
        3: "Motorcycle",
        5: "Bus",
        7: "Truck",
    },
    "color_classes": [
        "BEIGE", "BLACK", "BLUE", "BROWN", "GOLD",
        "GREEN", "GREY", "ORANGE", "PINK", "PURPLE",
        "RED", "SILVER", "TAN", "WHITE", "YELLOW",
    ],
}