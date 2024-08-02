import torch
import os
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
import utils
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

cfg_path = '/home/julian/work/dinov2/ultralytics/ultralytics/cfg/models/v8/yolov8m-neck.yaml'
model = DetectionModel(cfg=cfg_path).cuda()
image = torch.rand(1, 3, 640, 640).cuda()
weight_path = '/home/julian/work/dinov2/yolov8m.pt'
weight_model = YOLO("yolov8m.yaml").load(weight_path)
ssl_state_dict = weight_model.model.state_dict()
utils.update_ssl_backbone(model.model, ssl_state_dict, prefix='model.')
import ipdb; ipdb.set_trace()