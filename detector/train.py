from ultralytics import YOLO

model = YOLO("yolov8n.pt")
# model = YOLO("yolov8s.pt")
# model = YOLO("runs/detect/train/weights/last.pt")

results = model.train(
    data="config.yaml",
    name="yolov8n_720x960_larger_data_split",
    epochs=100,
    patience=10,
    plots=True,
    fliplr=0,
    mosaic=0.6,
    hsv_s=0.5,
    flipud=0,
    # degrees=90,
    # imgsz=960
    # resume=True,
    # device="cpu",
)
