import torch
import copy
from ultralytics.nn.tasks import yaml_model_load, parse_model


cfg = '/home/julian/work/dinov2/ultralytics/ultralytics/cfg/models/v8/yolov8n-ssl.yaml'
yaml = yaml_model_load(cfg) 
ch = 3
# model = ClassificationModel(cfg=cfg, ch=3, verbose=True)
model, save = parse_model(copy.deepcopy(yaml), ch=ch, verbose=True)  # model, savelist
image = torch.rand(2, ch, 640, 640).cuda()
model.cuda()
result = model(image)
import ipdb; ipdb.set_trace()