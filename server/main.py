from constants import (
    CLASS_NAMES,
    COIN_VALUES,
    COLOURS,
    DETECTION_CONF_THRESHOLD,
    DETECTION_THRESHOLD,
    IDENTICAL_IOU_THRESHOLD,
    INPUT_IMG_RATIO,
    INPUT_IMG_SIZE,
    MODEL_PATH,
    RESULTS_FOLDER,
    UPLOAD_FOLDER,
)
from utils import allowed_file, bb_intersection_over_union
from werkzeug.utils import secure_filename
from flask import (
    Flask,
    flash,
    request,
    redirect,
    url_for,
    render_template,
    send_from_directory,
)
import os
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont, ExifTags
import glob

app = Flask(__name__)

model = YOLO(MODEL_PATH)


for ORIENTATION_KEY in ExifTags.TAGS.keys():
    if ExifTags.TAGS[ORIENTATION_KEY] == "Orientation":
        break


@app.route("/", methods=["GET", "POST"])
def root():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(path)

            img = Image.open(path)

            exif = img.getexif()
            
            # Rotate
            if len(exif) > 0:
                if exif[ORIENTATION_KEY] == 3:
                    img = img.rotate(180, expand=True)
                elif exif[ORIENTATION_KEY] == 6:
                    img = img.rotate(270, expand=True)
                elif exif[ORIENTATION_KEY] == 8:
                    img = img.rotate(90, expand=True)
            # Resize
            if img.size != INPUT_IMG_SIZE:
                # Crop to aspect ratio
                if img.size[0] / img.size[1] != INPUT_IMG_RATIO:
                    if img.size[0] < img.size[1]:
                        new_h = img.size[0] * 1 / INPUT_IMG_RATIO
                        img = img.crop((0, 0, img.size[0], new_h))
                    else:
                        new_w = img.size[1] * INPUT_IMG_RATIO
                        img = img.crop((0, 0, new_w, img.size[1]))

                resized = img.resize(INPUT_IMG_SIZE)
                resized.save(path)

            # Run inference
            result = model(path, conf=DETECTION_THRESHOLD)[0]

            confs = result.boxes.conf
            clss = result.boxes.cls
            bboxs = result.boxes.xyxy

            img = Image.open(path)
            draw = ImageDraw.Draw(img)

            sum_val = 0
            for j in range(len(confs)):
                box = list(bboxs[j])
                conf = float(confs[j])
                cls = int(clss[j])

                if conf < DETECTION_CONF_THRESHOLD:
                    continue

                skip = False
                for k in range(len(confs)):
                    if k == j:
                        continue

                    box2 = list(bboxs[k])
                    conf2 = float(confs[k])
                    iou = bb_intersection_over_union(box, box2)
                    if iou >= IDENTICAL_IOU_THRESHOLD:
                        if conf2 > conf:
                            skip = True
                            break
                if skip:
                    continue

                sum_val += COIN_VALUES[cls]

                cls_name = CLASS_NAMES[cls]
                cls_color = COLOURS[cls]
                draw.rectangle(xy=box, outline=cls_color, width=4)
                det_label = cls_name + " - " + str(round(float(conf) * 100, 1)) + "%"
                draw.text(
                    xy=(box[0], box[1] - 35),
                    text=det_label,
                    font=ImageFont.load_default(size=30),
                    fill=cls_color,
                )
            img.save(os.path.join(RESULTS_FOLDER, filename))

            return render_template(
                "result.html", value=round(sum_val, 2), filename=filename
            )
    return render_template("index.html")


@app.route("/result/<filename>")
def send_file(filename):
    return send_from_directory(RESULTS_FOLDER, filename)


if __name__ == "__main__":
    to_remove = [RESULTS_FOLDER, UPLOAD_FOLDER]
    for rm in to_remove:
        for f in os.listdir(rm):
            os.remove(os.path.join(rm, f))

    print("- Removed old uploaded/result files")

    app.run(host="127.0.0.1", port=8080, debug=True)
