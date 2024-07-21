import pathlib
import os
import shutil

def main():
    root_path = '/home/julian/Downloads/bosch-traffic-light/yolo/'
    src_path_list = [f'{root_path}/images/val', f'{root_path}/labels/val']
    dst_path_list = [path.replace('val', 'val-coco') for path in src_path_list]

    for src_path, dst_path in zip(src_path_list, dst_path_list):
        shutil.copytree(src_path, dst_path)

    lbl_files = sorted(pathlib.Path(dst_path_list[1]).rglob('*.txt'))
    print(f'Number of label files: {len(lbl_files)}')

    dst_index = '9'
    for file in lbl_files:
        # open file, replace first char in each line with 9, overwrite same file
        with open(file, 'r') as f:
            lines = f.readlines()
            lines = [line.replace('0', dst_index, 1) for line in lines]
        with open(file, 'w') as f:
            f.writelines(lines)


if __name__ == '__main__':
    main()