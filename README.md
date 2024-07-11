# Euro coin counter
This app uses the YOLOv8 object detection model to detect and classify different Euro coins. 

<div>
  <img src="/assets/app.png" height="300"/>
  <img src="/assets/app2.png" height="300"/>
</div>

## Data
The model was trained on a [custom dataset](https://www.kaggle.com/datasets/janstaffa/euro-coins-dataset) featuring 150 images with annotated Euro coins.

## Results
**mAP:**
- mAP@.5 = 0.91
- mAP@.5-.95 = 0.89

## How to run
1. clone this repo
2. cd into `/server`
3. start Flask application (`python main.py` or `python -m flask --app main.py run`)
