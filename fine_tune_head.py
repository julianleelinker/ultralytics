from ultralytics import YOLO


data_yaml = '/home/julian/work/dinov2/ultralytics/ultralytics/cfg/datasets/tiip-scratch_yolo-s1-s22.yaml'
# backbone_path = '/home/julian/data/ssl/tiip-v0.1-0716/model_final.pth' 
# backbone_path = '/mnt/data-home/julian/tiip/dinov2/tiip-yolov8m-v0.1.2-0722-2/model_final.pth'
# backbone_path = '/mnt/data-home/julian/tiip/dinov2/resize_640_global_512_local_224_0726/model_final.pth'
# backbone_path = '/mnt/data-home/julian/tiip/dinov2/distill-vits-yolov8m-tiip/model_0069999.rank_0.pth'
backbone_path = '/mnt/data-home/julian/tiip/dinov2/tiip-yolov8m-nocls-0906/model_0299999.rank_0.pth'
suffix = '-headlr0.1'
# suffix = '-head'

pretrain_name = backbone_path.split('/')[-2]
finetune_name = pretrain_name + suffix

model = YOLO("yolov8m.yaml")
# prefix = 'teacher.backbone.'
prefix = 'student.backbone.'
model.model.update_backbone(backbone_path, prefix=prefix)

results = model.train(data=data_yaml, epochs=100, imgsz=640, name=finetune_name, batch=64, freeze=9, optimizer='SGD', lr0=0.1)
# results = model.train(data=data_yaml, epochs=100, imgsz=640, name=finetune_name, batch=64, freeze=9)
