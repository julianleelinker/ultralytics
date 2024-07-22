from ultralytics import YOLO


# backbone_path = '/mnt/data-home/julian/tiip/dinov2/test0712/model_final.pth' # 197

# backbone_path = '/home/julian/data/ssl/tiip-yolov8s-v0.1.1-0719/model_final.pth'
backbone_path = '/home/julian/data/ssl/tiip-yolov8m-v0.1.1-0720/model_final.pth'
# backbone_path = '/home/julian/data/ssl/tiip-v0.1-0716/model_final.pth'
data_yaml = '/home/julian/work/ultralytics/ultralytics/cfg/datasets/tiip-scratch_yolo-s1-s19_s21.yaml'

# the model_path was saved by following code
#with self.path_manager.open(save_file, "wb") as f:
#    torch.save(data, cast(IO[bytes], f))

# load the model
# model = torch.load(model_path)
model = YOLO("yolov8m.yaml")
model.model.update_backbone(backbone_path)

results = model.train(data=data_yaml, epochs=100, imgsz=640, name='tiip-yolov8m-v0.1.1-0721', batch=64, freeze=9)
