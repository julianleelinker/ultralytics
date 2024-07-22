
from ultralytics import YOLO


data_path = '/mnt/data-home/julian/sports-image-classification/dataset'
model = YOLO('yolov8n-cls.pt') # load a pretrained model (recommended for training)
model.train(data=data_path, epochs=5)