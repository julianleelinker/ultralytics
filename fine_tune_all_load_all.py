from ultralytics import YOLO


data_yaml = '/home/julian/work/dinov2/ultralytics/ultralytics/cfg/datasets/tiip-scratch_yolo-s1-s22.yaml'
# pretrain_path = '/home/julian/work/ultralytics/runs/detect/tiip-yolov8m-v0.1.1-0721-finetune/weights/best.pt'
# pretrain_path = '/home/julian/work/ultralytics/runs/detect/tiip-yolov8m-v0.1.2-0722-2/weights/best.pt'
# pretrain_path = '/home/julian/work/ultralytics/runs/detect/tiip-yolov8m-v0.1.2-0726-resize-freeze/weights/best.pt'
# pretrain_path = '/home/julian/work/ultralytics/runs/detect/tiip-yolov8m-v0.1.2-0726-resize-freeze/weights/best.pt'
# pretrain_path = '/home/julian/work/ultralytics/runs/detect/tiip-yolov8m-v0.1.2-0722-2-larger-head/weights/best.pt'
pretrain_path = '/home/julian/work/ultralytics/runs/detect/distill-vits-yolov8m-tiip-head/weights/best.pt'
pretrain_name = pretrain_path.split('/')[-3]
finetune_name = pretrain_name + '-all'

# load the model
model = YOLO("yolov8m.yaml").load(pretrain_path)
results = model.train(data=data_yaml, epochs=100, imgsz=640, name=finetune_name, batch=48)
