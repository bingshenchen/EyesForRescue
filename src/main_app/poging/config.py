import os
from dotenv import load_dotenv

load_dotenv()

YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH")
POSE_MODEL_PATH = os.getenv("yolov11n")
CLASSIFIER_PATH = os.getenv("CLASSIFIER_PATH")

CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.3

TRACKER_MAX_MISS = 5
TRACKER_MIN_HITS = 3

DISPLAY_RESULTS = True
SAVE_VIDEO = False
OUTPUT_VIDEO_PATH = os.getenv("OUTPUT_VIDEO_PATH", "output.mp4")

ENABLE_TRACKING = True
ENABLE_CLASSIFICATION = True

CLASS_COLORS = {
    "person": (0, 255, 0),
    "falling_person": (0, 0, 255),
    "sitting_person": (255, 255, 0),
    "lying_person": (255, 0, 255)
}