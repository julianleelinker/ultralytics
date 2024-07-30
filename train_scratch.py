from ultralytics import YOLO

# data_yaml = '/home/julian/work/ultralytics/ultralytics/cfg/datasets/tiip-scratch_yolo-s1-s19_s21.yaml' # leo
data_yaml = '/home/julian/work/dinov2/ultralytics/ultralytics/cfg/datasets/tiip-scratch_yolo-s1-s22.yaml'

# load the model
# model = torch.load(model_path)
model = YOLO("yolov8m.yaml")

results = model.train(data=data_yaml, epochs=100, imgsz=640, name='scratch-yolov8m-0722', batch=32)
