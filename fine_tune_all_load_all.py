from ultralytics import YOLO


data_yaml = '/home/julian/work/dinov2/ultralytics/ultralytics/cfg/datasets/tiip-scratch_yolo-s1-s22.yaml'
# pretrain_path = '/home/julian/work/ultralytics/runs/detect/tiip-yolov8m-v0.1.1-0721-finetune/weights/best.pt'
# pretrain_path = '/home/julian/work/ultralytics/runs/detect/tiip-yolov8m-v0.1.2-0722-2/weights/best.pt'
# pretrain_path = '/home/julian/work/ultralytics/runs/detect/tiip-yolov8m-v0.1.2-0726-resize-freeze/weights/best.pt'
# pretrain_path = '/home/julian/work/ultralytics/runs/detect/tiip-yolov8m-v0.1.2-0726-resize-freeze/weights/best.pt'
# pretrain_path = '/home/julian/work/ultralytics/runs/detect/tiip-yolov8m-v0.1.2-0722-2-larger-head/weights/best.pt'
# pretrain_path = '/home/julian/work/ultralytics/runs/detect/distill-vits-yolov8m-tiip-head/weights/best.pt'
# pretrain_path = '/home/julian/work/ultralytics/runs/detect/distill-vits-yolov8m-tiip-headlr0.1/weights/best.pt'
# pretrain_path = '/home/julian/work/ultralytics/runs/detect/tiip-yolov8m-nocls-0906-head/weights/best.pt'
pretrain_path = '/home/julian/work/ultralytics/runs/detect/tiip-yolov8m-nocls-0906-headlr0.1/weights/best.pt'
pretrain_name = pretrain_path.split('/')[-3]
suffix = '-all0.005'
# suffix = '-all'
finetune_name = pretrain_name + suffix

# load the model
model = YOLO("yolov8m.yaml").load(pretrain_path)
# results = model.train(data=data_yaml, epochs=100, imgsz=640, name=finetune_name, batch=48)
results = model.train(data=data_yaml, epochs=100, imgsz=640, name=finetune_name, batch=48, optimizer='SGD', lr0=0.005)
