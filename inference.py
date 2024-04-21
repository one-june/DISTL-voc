#%%
import os#; os.environ['CUDA_VISIBLE_DEVICES']='2'
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

#%%
def load_img(img_path, img_size=(256,256), patch_size=8):
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
    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    return img, w_featmap, h_featmap

#%%
# PRETRAINED_WEIGHTS = 'outputs/useall-100-100/fold0/checkpoint.pth'
PRETRAINED_WEIGHTS = 'outputs/split-10-30-30-30/fold2/checkpoint.pth'
PRETRAINED_WEIGHTS = 'outputs/test/fold2/checkpoint.pth'
CHECKPOINT_KEY = 'student'
patch_size, out_dim, n_classes = 8, 65536, 20
model = vit_o.__dict__['vit_small'](patch_size=patch_size)
embed_dim = model.embed_dim
model = utils.MultiCropWrapper(
    model,
    vit_o.DINOHead(in_dim=embed_dim, out_dim=out_dim),
    vit_o.CLSHead(in_dim=384, hidden_dim=256, num_classes=n_classes)
)

sd = torch.load(PRETRAINED_WEIGHTS, map_location='cpu')
sd = sd[CHECKPOINT_KEY]
sd = {k.replace("module.", ""): v for k, v in sd.items()}

msg = model.load_state_dict(sd)
print(msg)
model = model.to('cuda')

# %%
voc12_root = '/home/wonjun/data/voc2012'
ids_path = os.path.join(voc12_root, 'train_id.txt')
with open(ids_path, 'r') as handle:
    ids = handle.readlines()
ids = [s.strip() for s in ids]

label_file_path = os.path.join(voc12_root, 'cls_labels.npy')
cls_labels_dict = np.load(label_file_path, allow_pickle=True).item()


n_correct, n_total = 0, 0
for img_name in tqdm(ids, desc="Calculating accuracy", colour='green'):
    img_path = os.path.join(voc12_root, 'VOCdevkit', 'VOC2012', 'JPEGImages',
                            img_name+'.jpg')
    img, _, _ = load_img(img_path)
    preds = model(img.to('cuda'))
    preds = [torch.sigmoid(p).detach().cpu().numpy()[0][0] for p in preds]
    preds = [1 if p >= 0.5 else 0 for p in preds]
    
    label = cls_labels_dict[img_name]
    n_correct += sum([1 for true, pred in zip(label, preds) if true==pred])
    n_total += len(preds)

accuracy = n_correct / n_total
print(f"overall accuracy for {PRETRAINED_WEIGHTS}:\n {accuracy*100:.4f}%")
# %%
