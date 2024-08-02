import torch
from ultralytics.nn.tasks import yaml_model_load, parse_model
from ultralytics.utils.torch_utils import initialize_weights
from ultralytics import YOLO
import copy


data_yaml = '/home/julian/work/dinov2/ultralytics/ultralytics/cfg/datasets/tiip-scratch_yolo-s1-s22.yaml'
yolo_path = '/home/julian/work/dinov2/ultralytics/ultralytics/cfg/models/v8/yolov8m-linear-detect-3head.yaml'
yolo_yaml = yaml_model_load(yolo_path) 
ch = 3

# model, _ = parse_model(copy.deepcopy(yolo_yaml), ch=ch, verbose=True)
# model = model.to(device)
# sample = torch.rand(2, 3, 640, 640).to(device)

model = YOLO(yolo_path)
results = model.train(data=data_yaml, epochs=100, imgsz=640, name='tiip-linear-detect-3head', batch=64)
