import os
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torchvision import transforms
from torch.utils.data import Dataset


def load_img_name_list(dataset_path):
    img_gt_name_list = open(dataset_path).readlines()
    img_name_list = [img_gt_name.strip() for img_gt_name in img_gt_name_list]

    return img_name_list


def load_image_label_list_from_npy(img_name_list, label_file_path):
    cls_labels_dict = np.load(label_file_path, allow_pickle=True).item()
    label_list = []
    for id in img_name_list:
        if id not in cls_labels_dict.keys():
            img_name = id + '.jpg'
        else:
            img_name = id
        label_list.append(cls_labels_dict[img_name])
    return label_list
    # return [cls_labels_dict[img_name] for img_name in img_name_list ]

class VOC12Dataset(Dataset):
    def __init__(self,
                 images_root='/home/wonjun/data/voc2012/VOCdevkit/VOC2012/JPEGImages',
                 data_folds_path='data_preparation',
                 pretrain=True,
                 num_folds=0,
                 transforms=None):
        
        self.images_root = images_root
        self.pretrain = pretrain
        self.transforms = transforms
        
        dfs = []
        if self.pretrain:
            assert num_folds==0
            df_path = os.path.join(data_folds_path, "voc12_pretrain.csv")
            dfs.append(pd.read_csv(df_path))
            
        elif not self.pretrain:
            assert num_folds in [1, 2, 3]
            for fold in range(num_folds):
                df_path = os.path.join(data_folds_path, f"voc12_fold_{fold}.csv")
                dfs.append(pd.read_csv(df_path))
        self.df = pd.concat(dfs, axis=0)

    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_root, row.ids+'.jpg')
        image = cv2.imread(img_path, 1)
        image = cv2.resize(image, dsize=(256,256), interpolation=cv2.INTER_LINEAR)
        image = Image.fromarray(image)
        
        images = self.transforms(image)
        
        if self.pretrain:
            label = torch.Tensor(row[[str(i) for i in range(20)]].values.astype(int))
        else:
            label = []
            
        return images, label


# class VOC12Dataset(Dataset):
#     def __init__(self, data_root, label_file_path="voc12/cls_labels.npy", train=True, transform=None, gen_attn=False):
#         img_name_list_path = os.path.join("voc12", f'{"train_aug" if train or gen_attn else "val"}_id.txt')
#         self.img_name_list = load_img_name_list(img_name_list_path)
#         self.label_list = load_image_label_list_from_npy(self.img_name_list, label_file_path)
#         data_root = Path(data_root) / "voc12" if "voc12" not in data_root else data_root
#         self.data_root = Path(data_root) / "VOCdevkit" / "VOC2012"
#         self.gt_dir = self.data_root / "SegmentationClass"
#         self.transform = transform

#     def __getitem__(self, idx):
#         name = self.img_name_list[idx]
#         img = PIL.Image.open(os.path.join(self.data_root, 'JPEGImages', name + '.jpg')).convert("RGB")
#         label = torch.from_numpy(self.label_list[idx])
#         if self.transform:
#             img = self.transform(img)

#         return img, label

#     def __len__(self):
#         return len(self.img_name_list)