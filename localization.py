#%%
import os; os.environ['CUDA_VISIBLE_DEVICES']='3'
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
IND_TO_LABEL = {
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
LABEL_TO_IND = {v:k for k,v in IND_TO_LABEL.items()}


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

def get_layer_selfattention(model: utils.MultiCropWrapper,
                             img: torch.Tensor,
                             layer_index: int,
                             patch_size=8):
    attentions = model.backbone.get_layer_selfattention(img, layer_index)
    attentions = attentions.detach()
    nh = attentions.shape[1]
    
    # keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(
        attentions.unsqueeze(0), scale_factor=patch_size, mode='nearest'
    )[0].cpu().numpy()
    
    im = np.transpose(img[0].detach().cpu().numpy(), (1,2,0))
    fig, ax = plt.subplots(2,3, figsize=(9,6))
    for j in range(nh):
        ax[j//3, j%3].imshow(im, alpha=0.3)
        ax[j//3, j%3].imshow(attentions[j], alpha=0.7)
    fig.suptitle(f"attention maps for layer {layer_index}")
    plt.show()
    
    return attentions

def get_head_selfattention(model: utils.MultiCropWrapper,
                           img: torch.Tensor,
                           layer_index:int,
                           head_index:int,
                           patch_size=8):
    attentions = model.backbone.get_layer_selfattention(img, layer_index)
    attentions = attentions.detach()
    nh = attentions.shape[1]
    
    # keep only the output patch attention
    attentions = attentions[0,:,0,1:].reshape(nh,-1)
    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attention = attentions[head_index]
    attention = nn.functional.interpolate(
        attention.unsqueeze(0).unsqueeze(0),
        scale_factor=patch_size,
        mode='nearest'
    )[0][0].cpu().numpy()
    
    im = np.transpose(img[0].detach().cpu().numpy(), (1,2,0))
    fig, ax = plt.subplots(1,1, figsize=(4,4))
    ax.imshow(im, alpha=0.5)
    ax.imshow(attention, alpha=0.5)
    fig.suptitle(f"layer {layer_index} head {head_index} attention map")
    plt.show()
    
    return attention

def get_head_importance(classes_of_interest: list,
                        model,
                        img):
    
    head_importance_by_class = {}
    fig, ax = plt.subplots(1, len(classes_of_interest), figsize=(5*len(classes_of_interest), 5))
    if len(classes_of_interest)==1: ax=[ax]

    for i, _class in enumerate(classes_of_interest):
        class_ind = LABEL_TO_IND[_class]
        pred = model(img)
        pred[class_ind].backward()
        head_importance = []
        for block in model.backbone.blocks:
            ctx = block.attn.context_layer_val
            grad_ctx = ctx.grad
            dot = torch.einsum("bhli,bhli->bhl", [grad_ctx, ctx])
            head_importance.append(dot.abs().sum(-1).sum(0).detach())
        head_importance = torch.vstack(head_importance)

        # Normalize attention values by layer
        exponent = 2
        norm_by_layer = torch.pow(torch.pow(head_importance, exponent).sum(-1), 1/exponent)
        head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20
        
        head_importance = head_importance.detach().cpu().numpy().astype(np.float32)
        ax[i].imshow(head_importance, cmap='plasma')
        ax[i].set_title(_class)
        
        head_importance_by_class[_class] = head_importance

        del pred
    
    plt.show()
    
    return head_importance_by_class

#%% Try for pretrained weights (before DISTL pretraining)
# PRETRAINED_WEIGHTS = 'pretrained_weights/pretrain.ckpt'
# state_dict = torch.load(PRETRAINED_WEIGHTS, map_location='cpu')['state_dict']
# state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
# state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
# state_dict = {k.replace("dino.", ""): v for k, v in state_dict.items()}
# model = vit_o.__dict__['vit_small'](patch_size=8)
# msg = model.load_state_dict(state_dict, strict=False)
# pprint(msg)
# model = model.to('cuda')

# %% Show single example

PRETRAINED_WEIGHTS = 'outputs/useall-100-100/fold0/checkpoint.pth'
# PRETRAINED_WEIGHTS = 'outputs/split-10-30-30-30/fold2/checkpoint.pth'
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


#%% Load an image
voc12_img_root = '/home/wonjun/data/voc12/VOCdevkit/VOC2012/JPEGImages'
img_id = '2007_002597'
img_path = os.path.join(voc12_img_root, img_id+'.jpg')

img, w_featmap, h_featmap = load_img(
    img_path=img_path, img_size=(256,256)
)
img = img.to('cuda')

#%%
_ = get_layer_selfattention(model, img, layer_index=-1)
#%%

# =========== Get model preds ============
pred = model(img)
pred = np.array([t.detach().cpu().numpy()[0][0] for t in pred])

predicted_classes = np.argsort(pred)[::-1] # list items in order of decreasing value
predicted_classes = [ind for ind in predicted_classes if ind in np.where(pred>=0)[0]]
predicted_logits = [pred[i] for i in predicted_classes]
predicted_classes = [IND_TO_LABEL[ind] for ind in predicted_classes]
print(predicted_classes)
print(predicted_logits)

head_importance_by_class = get_head_importance(predicted_classes,
                                               model,
                                               img)
for _cls, head_importance in head_importance_by_class.items():
    print(f"Most important attention head for class {_cls.upper()}")
    
    # most important head in a certain layer
    layer_ind = -2
    best_head_ind = np.argmax(head_importance[layer_ind])
    attn_map = get_head_selfattention(model, img, layer_ind, best_head_ind)

    attn_map_q25 = (attn_map > np.percentile(attn_map, 25)).astype(np.uint8)
    attn_map_q50 = (attn_map > np.percentile(attn_map, 50)).astype(np.uint8)
    attn_map_q75 = (attn_map > np.percentile(attn_map, 75)).astype(np.uint8)
    attn_map_q95 = (attn_map > np.percentile(attn_map, 95)).astype(np.uint8)
    attn_map_q98 = (attn_map > np.percentile(attn_map, 98)).astype(np.uint8)

    fig, ax = plt.subplots(1,6, figsize=(24, 4))
    ax[0].imshow(attn_map, cmap='plasma')
    ax[1].imshow(attn_map_q25, cmap='plasma')
    ax[2].imshow(attn_map_q50, cmap='plasma')
    ax[3].imshow(attn_map_q75, cmap='plasma')
    ax[4].imshow(attn_map_q95, cmap='plasma')
    ax[5].imshow(attn_map_q98, cmap='plasma')
    plt.show()

#%%
# ========= segmentation maps from attention maps =========
c = predicted_classes[0]
hi = head_importance_by_class[c]
hi
#%%
hi[-1:].shape
#%%
head_ind = np.unravel_index(np.argmax(hi[-1:]), hi.shape)
most_important_head = hi[-1:][head_ind]
# %%
most_important_head