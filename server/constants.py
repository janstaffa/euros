ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

UPLOAD_FOLDER = "./data/uploads"
RESULTS_FOLDER = "./data/results"

IDENTICAL_IOU_THRESHOLD = 0.9
DETECTION_CONF_THRESHOLD = 0.6

INPUT_IMG_SIZE = (720, 960)
INPUT_IMG_RATIO = INPUT_IMG_SIZE[0] / INPUT_IMG_SIZE[1]

COIN_VALUES = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2]
CLASS_NAMES = [
    "1_cent",
    "2_cent",
    "5_cent",
    "10_cent",
    "20_cent",
    "50_cent",
    "1_euro",
    "2_euro",
]
COLOURS = ["red", "green", "blue", "yellow", "orange", "magenta", "brown", "white"]

DETECTION_THRESHOLD = 0.5

MODEL_PATH = "./models/yolov8n_760x920_data_split.pt"