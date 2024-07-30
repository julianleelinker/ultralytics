from ultralytics import YOLO


# backbone_path = '/home/julian/data/ssl/tiip-v0.1-0716/model_final.pth' 
backbone_path = '/mnt/data-home/julian/tiip/dinov2/tiip-yolov8m-v0.1.2-0722-2/model_final.pth'
# backbone_path = '/mnt/data-home/julian/tiip/dinov2/resize_640_global_512_local_224_0726/model_final.pth'
data_yaml = '/home/julian/work/dinov2/ultralytics/ultralytics/cfg/datasets/tiip-scratch_yolo-s1-s22.yaml'
# data_yaml = '/home/julian/work/dinov2/ultralytics/ultralytics/cfg/datasets/tiip-scratch_yolo-s1-s19_s21.yaml'
model = YOLO("yolov8m.yaml")
model.model.update_backbone(backbone_path)

results = model.train(data=data_yaml, epochs=100, imgsz=640, name='tiip-yolov8m-v0.1.2-0722-2-larger-head', batch=64, freeze=9, optimizer='SGD', lr0=0.1)
# results = model.train(data=data_yaml, epochs=100, imgsz=640, name='tiip-yolov8m-v0.1.2-0722-2-larger-head', batch=64, freeze=9)
