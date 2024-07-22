# %%
from ultralytics import YOLO
from PIL import Image

model_path = '/home/julian/work/ultralytics/models/yolov8m.pt'
model = YOLO(model_path)
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam

# results = model.predict(source="0")
# results = model.predict(source="folder", show=True) # Display preds. Accepts all YOLO predict arguments

# from PIL
# image_path = '/home/julian/work/ultralytics/downtown-Seattle.jpg'
image_path = '/home/julian/work/ultralytics/traffic-light-sample.jpg'
image = Image.open(image_path)
results = model.predict(source=image, save=True)  # save plotted images

# %%
