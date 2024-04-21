#%%
import os
import argparse
import cv2
import random
import colorsys
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image

import utils
import vision_transformer as vit_o
import glob

seed = 2228
ind_to_label = {
    0:'aeroplane',
    1:'bicycle',
    2:'bird',
    3:'boat',
    4:'bottle',
    5:'bus',
    6:'car',
    7:'cat',
    8:'chair',
    9:'cow',
    10:'diningtable',
    11:'dog',
    12:'horse',
    13:'motorbike',
    14:'person',
    15:'pottedplant',
    16:'sheep',
    17:'sofa',
    18:'train',
    19:'tvmonitor'
}
label_to_ind = {v:k for k,v in ind_to_label.items()}

def load_sample_imgs(img_paths, img_size=(256,256), patch_size=8):
    sample_imgs = []
    for img_path in img_paths:
        img = cv2.imread(img_path, 1)
        img = cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_LINEAR)
        img = Image.fromarray(img)
        img = pth_transforms.Compose(
            [
                utils.GaussianBlurInference(),
                pth_transforms.ToTensor()
            ]
        )(img) # ( 3, img_size[0], img_size[1] )
    
        # make the image divisible by patch size
        w, h = img.shape[1]-img.shape[1]%patch_size, img.shape[2]-img.shape[2]%patch_size
        img = img[:, :w, :h].unsqueeze(0)
        sample_imgs.append(img)
    sample_imgs_tensor = torch.cat(sample_imgs)
    
    w_featmap = img_size[0] // patch_size
    h_featmap = img_size[1] // patch_size

    return sample_imgs_tensor, w_featmap, h_featmap


#%%
@torch.no_grad()
def save_attention_maps(model: utils.MultiCropWrapper,
                        img_paths,
                        save_dir,
                        iter):
    model = model.to('cuda')
    
    sample_imgs, w_featmap, h_featmap = load_sample_imgs(img_paths)
    sample_imgs = sample_imgs.to('cuda')
    attentions = model.backbone.get_last_selfattention(sample_imgs)
    attentions = attentions.detach()
    n_imgs = attentions.shape[0]
    nh = attentions.shape[1]
    
    # we keep only the output patch attention
    attentions = attentions[:, :, 0, 1:].reshape(n_imgs, nh, -1)
    attentions = attentions.reshape(n_imgs, nh, w_featmap, h_featmap)
    attentions = attentions.detach().cpu().numpy()
    
    for img_path, attention_map in zip(img_paths, attentions):
        img_name = img_path.split('/')[-1].split('.')[0]
        np.save(os.path.join(save_dir, img_name+f"_{iter}"), attention_map)
    
    return attentions

# %%
# PRETRAINED_WEIGHTS = 'outputs/useall-100-100/fold0/checkpoint.pth'
# # PRETRAINED_WEIGHTS = 'outputs/split-10-30-30-30/voc_fold2/checkpoint.pth'
# CHECKPOINT_KEY = 'student'
# patch_size, out_dim, n_classes = 8, 65536, 20
# model = vit_o.__dict__['vit_small'](patch_size=patch_size)
# embed_dim = model.embed_dim
# model = utils.MultiCropWrapper(
#     model,
#     vit_o.DINOHead(in_dim=embed_dim, out_dim=out_dim),
#     vit_o.CLSHead(in_dim=384, hidden_dim=256, num_classes=n_classes)
# )

# sd = torch.load(PRETRAINED_WEIGHTS, map_location='cpu')
# sd = sd[CHECKPOINT_KEY]
# sd = {k.replace("module.", ""): v for k, v in sd.items()}

# msg = model.load_state_dict(sd)
# print(msg)
# model = model.to('cuda')

# voc_imgs_root = '/media/wonjun/HDD8TB/voc12/VOCdevkit/VOC2012/JPEGImages'
# img_paths = [
#     os.path.join(voc_imgs_root, '2007_000063.jpg'),
#     os.path.join(voc_imgs_root, '2007_000129.jpg'),
#     os.path.join(voc_imgs_root, '2007_000799.jpg'),
#     os.path.join(voc_imgs_root, '2007_000925.jpg'),
#     os.path.join(voc_imgs_root, '2007_001678.jpg')
# ]

# attentions = save_attention_maps(model, img_paths)

# for attn_map in attentions:
#     fig, ax = plt.subplots(2, 3, figsize=(9, 6))
#     for i, map in enumerate(attn_map):
#         ax[i//3, i%3].imshow(map)
#     plt.show()
# %%
