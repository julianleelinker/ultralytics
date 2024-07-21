import yaml
import tqdm
import pathlib
import random


def xyxy_to_normxcycwh(xmin, ymin, xmax, ymax, img_shape):
    '''
    there may be reverse max/min, so taking abs for w, h
    img_shape: (height, width)
    '''
    height, width = img_shape
    xc = 0.5* (xmin + xmax) / width
    yc = 0.5* (ymin + ymax) / height
    w = abs(xmax-xmin) / width
    h = abs(ymax-ymin) / height
    return xc, yc, w, h


def get_yolo_label_str(frame, img_shape, class_index):
    lines = []
    for box in frame['boxes']:
        xc, yc, w, h  = xyxy_to_normxcycwh(box['x_min'], box['y_min'], box['x_max'], box['y_max'], img_shape=img_shape)
        line = f"{class_index:d} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n"
        lines.append(line)
    return lines


def has_out_of_range_label(frame, img_shape):
    height, width = img_shape
    for box in frame['boxes']:
        if min(box['x_min'], box['y_min'], box['x_max'], box['y_max']) < 0:
            return True
        if max(box['x_max'], box['x_min']) >= width:
            return True
        if max(box['y_max'], box['y_min']) >= height:
            return True
    return False


def main():
    org_data_path = '/home/julian/Downloads/bosch-traffic-light/original'
    test_path = f'{org_data_path}/test.yaml'
    train_path = f'{org_data_path}/train.yaml'
    img_shape = (720, 1280)
    class_index = 0
    yolo_root = pathlib.Path('/home/julian/Downloads/bosch-traffic-light/yolo')
    yolo_img_root = yolo_root / 'images'
    yolo_lbl_root = yolo_root / 'labels'

    for split in ['train', 'val', 'test']:
        pathlib.Path(yolo_img_root / split).mkdir(parents=True, exist_ok=True)
        pathlib.Path(yolo_lbl_root / split).mkdir(parents=True, exist_ok=True)

    with open(test_path) as f:
        test_yaml = yaml.load(f, Loader=yaml.FullLoader)

    with open(train_path) as f:
        train_yaml = yaml.load(f, Loader=yaml.FullLoader)

    cnt = {
        'out_of_range': 0,
        'train': 0,
        'val': 0,
        'test': 0,
    }
    for frame in tqdm.tqdm(train_yaml):
        if has_out_of_range_label(frame, img_shape=img_shape):
            print(f'frame {frame["path"]} has out of range label')
            cnt['out_of_range'] += 1
            continue
        
        if random.uniform(0, 1) > 0.1:
            split = 'train'
            cnt['train'] += 1
        else:
            split = 'val'
            cnt['val'] += 1

        lines = get_yolo_label_str(frame, img_shape, class_index)
        file_name = frame['path'].split('/')[-1].split('.')[0] 

        file_path = f'{yolo_lbl_root}/{split}/{file_name}.txt'
        # print(f'writing to {file_path}')
        with open(file_path, 'w') as f:
            f.writelines(lines)

        # ../../../original/
        target_path = pathlib.Path(f'{yolo_img_root}/{split}/{file_name}.png')
        depth = 3
        src_path = target_path.parents[depth]/'original'/frame["path"]
        assert pathlib.Path(src_path).exists(), f'{src_path} does not exist'
        target_path.symlink_to(f'{"../"*depth}original/{frame["path"]}')

    for frame in tqdm.tqdm(test_yaml):
        if has_out_of_range_label(frame, img_shape=img_shape):
            print(f'frame {frame["path"]} has out of range label')
            cnt['out_of_range'] += 1
            continue

        cnt['test'] += 1
        lines = get_yolo_label_str(frame, img_shape, class_index)
        file_name = frame['path'].split('/')[-1].split('.')[0] 

        file_path = f'{yolo_lbl_root}/test/{file_name}.txt'
        # print(f'writing to {file_path}')
        with open(file_path, 'w') as f:
            f.writelines(lines)

        target_path = pathlib.Path(f'{yolo_img_root}/test/{file_name}.png')
        depth = 3
        src_path = target_path.parents[depth]/'original'/'rgb'/'test'/f'{file_name}.png'
        assert pathlib.Path(src_path).exists(), f'{src_path} does not exist'
        target_path.symlink_to(f'{"../"*depth}original/rgb/test/{file_name}.png')

    print(cnt)


if __name__ == '__main__':
    main()