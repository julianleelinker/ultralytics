# %%
import torch
import os
import copy
from ultralytics import YOLO
from ultralytics.nn.tasks import yaml_model_load, parse_model



# %% 
use_vit = False
yolo_od_model = False
show_all_diagrams = True
if use_vit:
    background_threshold = 0
    smaller = True
    whiten = False
    # settings for ViT
    feat_dim_dict = {'s': 384, 'b': 768, 'l': 1024, 'g': 1536}
    # model_size in ['s', 'b', 'l', 'g']
    model_size = 's' 
    feat_dim = feat_dim_dict[model_size]
    backbone = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_size}14').cuda()
    image_size = 518
    patch_size = backbone.patch_size # patchsize=14

else:
    background_threshold = 0
    smaller = True
    whiten = True
    # settings for YOLO
    yolo_path = '/home/julian/work/dinov2/ultralytics/ultralytics/cfg/models/v8/yolov8m-ssl.yaml'
    # ssl_path = '/mnt/data-home/julian/tiip/dinov2/imangenet-v0.1-0712/model_final.pth'
    # ssl_path = '/home/julian/yolov8n-scratch.pt'
    # ssl_path = '/home/julian/work/dinov2/yolov8n.pt'
    # ssl_path = '/mnt/data-home/julian/tiip/dinov2/tiip-v0.1.1-0719/model_final.pth'
    # ssl_path = '/mnt/data-home/julian/tiip/dinov2/tiip-yolov8m-v0.1.1-0720/model_final.pth'
    # ssl_path = '/mnt/data-home/julian/tiip/dinov2/tiip-yolov8m-v0.1.1-0722/model_final.pth'
    # ssl_path = '/mnt/data-home/julian/tiip/dinov2/tiip-yolov8m-v0.1.2-0722-2/model_0008699.pth'
    # ssl_path = '/mnt/data-home/julian/tiip/dinov2/tiip-yolov8m-v0.1.2-0722-2/model_final.pth'
    # ssl_path = '/mnt/data-home/julian/tiip/dinov2/resize_640/model_0005549.pth'
    ssl_path = '/mnt/data-home/julian/tiip/dinov2/resize_640_global_512_local_224/model_0001349.pth'
    # ssl_path = '/mnt/data-home/julian/tiip/dinov2/resize_640_global_512_local_224/model_final.pth'
    # ssl_path = '/home/julian/work/dinov2/scratch-yolov8m-07222-last.pt'
    yolo_yaml = yaml_model_load(yolo_path) 
    feat_dim = 576
    image_size = 640
    patch_size = 32

    def update_ssl_backbone(yolo_model, ssl_state_dict, prefix):
        print(f"updating backbone {ssl_path}")
        updated_count = 0
        unupdated_count = 0
        for key in yolo_model.state_dict():
            ssl_key = prefix + key
            if ssl_key in ssl_state_dict:
                # print(f'{ssl_key} in ssl model')
                updated_count += 1
                yolo_model.state_dict()[key].copy_(ssl_state_dict[ssl_key])
            else:
                print(f'{ssl_key} not in ssl model')
                unupdated_count += 1
        print(f'{updated_count=} {unupdated_count=}')

    
    ch = 3
    # model, save = parse_model(copy.deepcopy(yaml), ch=ch, verbose=True)  # model, savelist
    yolo_model, _ = parse_model(copy.deepcopy(yolo_yaml), ch=ch, verbose=True)
    ssl_model = torch.load(ssl_path)


    ssl_model = torch.load(ssl_path)
    if yolo_od_model:
        ssl_model = YOLO("yolov8m.yaml").load(ssl_path)
        ssl_state_dict = ssl_model.model.state_dict()
        prefix = 'model.'
    else:
        ssl_state_dict = ssl_model['model']
        prefix = 'teacher.backbone.'
    update_ssl_backbone(yolo_model, ssl_state_dict, prefix=prefix)
    backbone = yolo_model
    backbone.cuda()



#  cell 6
# this threshold depends on model size, and not sure the criterion is < or >
# currently tested < for b, g, > for l, s
# smaller = True
# folder_path = '/home/julian/work/Dino_V2/harryported_giffin_images'
folder_path = '/mnt/data-home/julian/ssl/pca-examples/tiip-s1'
# folder_path = '/mnt/data-home/julian/ssl/pca-examples/fashion'
# folder_path = '/mnt/data-home/julian/ssl/pca-examples/imagenet'


from PIL import Image
from torchvision import transforms
# transform = transforms.Compose([           
#                                 transforms.Resize(256),                    
#                                 transforms.CenterCrop(224),               
#                                 transforms.ToTensor(),                    
#                                 transforms.Normalize(                      
#                                 mean=[0.485, 0.456, 0.406],                
#                                 std=[0.229, 0.224, 0.225]              
#                                 )])


transform1 = transforms.Compose([           
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size), #should be multiple of model patch_size                 
                                transforms.ToTensor(),                    
                                transforms.Normalize(mean=0.5, std=0.2)
                                ])


#  cell 7

patch_h  = image_size//patch_size
patch_w  = image_size//patch_size


total_features  = []
with torch.no_grad():
    for img_path in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_path)
        img = Image.open(img_path).convert('RGB')
        img_t = transform1(img)
        if use_vit:
            features_dict = backbone.forward_features(img_t.unsqueeze(0).cuda())
        else:
            features_dict = backbone(img_t.unsqueeze(0).cuda())
        features = features_dict['x_norm_patchtokens']
        total_features.append(features)


total_features = torch.cat(total_features, dim=0).cpu().detach()
print('*'*50)
print(f'total features shape: {total_features.shape}')


# cell 8
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# First PCA to Seperate Background
# sklearn expects 2d array for traning
total_features = total_features.reshape(4 * patch_h * patch_w, feat_dim) 

pca = PCA(n_components=3, whiten=whiten)
pca.fit(total_features)
pca_features = pca.transform(total_features)
print('*'*50)
print(f'PCA features shape: {pca_features.shape}')


# visualize PCA components for finding a proper threshold
# 3 histograms for 3 components
if show_all_diagrams:
    fig, axis = plt.subplots(1, 4, figsize=(12, 3))
    fig.suptitle('PCA largest 3 components histograms', fontsize=12)
    for i in range(3):
        axis[i].hist(pca_features[:, i])
    plt.show()


# cell 9

# min_max scale
# pca_features[:, 0] = (pca_features[:, 0] - pca_features[:, 0].min()) / \
#                      (pca_features[:, 0].max() - pca_features[:, 0].min())
#pca_features = sklearn.processing.minmax_scale(pca_features)

if show_all_diagrams:
    fig, axis = plt.subplots(1, 4, figsize=(12, 3))
    fig.suptitle('PCA first component', fontsize=12)
    for i in range(4):
        axis[i].imshow(pca_features[i*patch_h*patch_w : (i+1)*patch_h*patch_w, 0].reshape(patch_h, patch_w))
    plt.show()


#  cell 10
# segment/seperate the backgound and foreground using the first component
if smaller:
    pca_features_bg = pca_features[:, 0] < background_threshold # from first histogram
else:
    pca_features_bg = pca_features[:, 0] > background_threshold # from first histogram
pca_features_fg = ~pca_features_bg

# if show_all_diagrams:
#     fig, axis = plt.subplots(1, 4, figsize=(12, 3))
#     fig.suptitle('background filtered by PCA first component', fontsize=12)
#     for i in range(4):
#         axis[i].imshow(pca_features_bg[i*patch_h*patch_w : (i+1)*patch_h*patch_w].reshape(patch_h, patch_w))
#     plt.show()


# cell 11

# 2nd PCA for only foreground patches
pca.fit(total_features[pca_features_fg]) 
pca_features_left = pca.transform(total_features[pca_features_fg])

for i in range(3):
    # min_max scaling
    pca_features_left[:, i] = (pca_features_left[:, i] - pca_features_left[:, i].min()) / (pca_features_left[:, i].max() - pca_features_left[:, i].min())

pca_features_rgb = pca_features.copy()
# for black background
pca_features_rgb[pca_features_bg] = 0
# new scaled foreground features
pca_features_rgb[pca_features_fg] = pca_features_left

# reshaping to numpy image format
pca_features_rgb = pca_features_rgb.reshape(4, patch_h, patch_w, 3)

fig, axis = plt.subplots(1, 4, figsize=(12, 3))
fig.suptitle('vitb-tiip', fontsize=12)
for i, img_path in enumerate(os.listdir(folder_path)):
    axis[i].imshow(pca_features_rgb[i])
plt.show()


fig, axis = plt.subplots(1, 4, figsize=(10, 2.5))
fig.suptitle('original image', fontsize=12)
for i, img_path in enumerate(os.listdir(folder_path)):
    img_path = os.path.join(folder_path, img_path)
    img = Image.open(img_path).convert('RGB').resize((image_size, image_size))
    axis[i].imshow(img)
plt.show()
# %%
