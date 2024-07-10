import os
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageColor, ImageFont


# ref: https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


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

COLOURS = ["red", "green", "blue", "yellow", "orange", "pink", "brown", "white"]


IDENTICAL_IOU_THRESHOLD = 0.9
DETECTION_CONF_THRESHOLD = 0.6

# model = YOLO("runs/detect/yolov8n_720x960_larger_rotation/weights/best.pt")
model = YOLO("runs/detect/yolov8n_720x960_larger_data_split/weights/best.pt")
# model = YOLO("yolov8n.pt")

path = "test_images"
# images = list(path + "/" + x for x in os.listdir(path))
images = ["test_new.jpg"]
results = model(images, conf=DETECTION_CONF_THRESHOLD)

# Process results list
for i in range(len(results)):
    result = results[i]
    img_path = images[i]

    confs = result.boxes.conf
    clss = result.boxes.cls
    bboxs = result.boxes.xyxy
    im = Image.open(img_path)

    draw = ImageDraw.Draw(im)

    for j in range(len(confs)):
        box = list(bboxs[j])
        conf = confs[j]
        cls = clss[j]

        if conf < DETECTION_CONF_THRESHOLD:
            continue

        skip = False
        for k in range(len(confs)):
            if k == j:
                continue

            box2 = list(bboxs[k])
            conf2 = confs[k]
            iou = bb_intersection_over_union(box, box2)
            if iou >= IDENTICAL_IOU_THRESHOLD:
                if conf2 > conf:
                    skip = True
                    break
        if skip:
            continue

        cls_name = CLASS_NAMES[int(cls)]
        cls_color = COLOURS[int(cls)]
        draw.rectangle(xy=box, outline=cls_color, width=4)
        det_label = cls_name + " - " + str(round(float(conf) * 100, 1)) + "%"
        draw.text(
            xy=(box[0], box[1] - 35),
            text=det_label,
            font=ImageFont.load_default(size=30),
            fill=cls_color,
        )

    im.show("Result")
    # print(confs, clss, result.boxes)
    # result.show()  # display to screen
    # result.save(filename="result.jpg")  # save to disk
