#%%
import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import os
import glob
import cv2
from tqdm import tqdm
import random
import argparse
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, KFold
from skmultilearn.model_selection.iterative_stratification import IterativeStratification

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
# %%
voc_root = '/media/wonjun/HDD8TB/voc12'
train_ids_path = os.path.join(voc_root, 'train_aug_id.txt')
with open(train_ids_path, 'r') as handle:
    img_gt_name_list = handle.readlines()
img_name_list = [s.strip() for s in img_gt_name_list]

labels_file_path = os.path.join(voc_root, 'cls_labels.npy')
cls_labels_dict = np.load(labels_file_path, allow_pickle=True).item()

label_list = [cls_labels_dict[img_name] for img_name in img_name_list]

# %% Show an example
# i = 9293

# img_name = img_name_list[i]
# img_path = os.path.join(voc_root, 'VOCdevkit', 'VOC2012', 'JPEGImages',
#                         img_name+'.jpg')
# img = PIL.Image.open(img_path).convert("RGB")
# display(img)

# labels = label_list[i]
# inds = np.where(labels==1)[0]
# labels = [ind_to_label[ind] for ind in inds]
# print(labels)

#%%
df = pd.DataFrame(label_list)

#%% Select about 10% of cases to be used as pretraining set
# We want each class to be represented in the pretraining set
# so we look for positive cases of the class and take about 10% from them
selected_indices = []
for i in range(20):
    positives_inds = list(df[df[i]==1].index)
    selected_positives_inds = random.sample(positives_inds, int(0.1*len(positives_inds)))
    selected_indices.extend(selected_positives_inds)
selected_indices_set = set(selected_indices) # remove duplicates

pretrain_df = df.iloc[list(selected_indices_set)]

pretrain_df['ids'] = pretrain_df.index.to_series().apply(
    lambda name: img_name_list[name]
)

#%%
# # Show an example
# i = 888
# row = pretrain_df.iloc[i]
# img_path = os.path.join(voc_root, 'VOCdevkit', 'VOC2012', 'JPEGImages',
#                         row.ids+'.jpg')
# img = Image.open(img_path).convert("RGB")
# display(img)
# labels = [ind_to_label[ind] for ind in np.where(pretrain_df.iloc[i].values==1)[0]]
# print(labels)


#%% Split the rest into three folds
distl_df = df.iloc[~df.index.isin(list(selected_indices_set))]
distl_df['ids'] = distl_df.index.to_series().apply(
    lambda name: img_name_list[name]
)
distl_df = distl_df.sample(frac=1, random_state=2228)

split_size = len(distl_df) // 3
fold0 = distl_df.iloc[:split_size]
fold1 = distl_df.iloc[split_size: 2*split_size]
fold2 = distl_df.iloc[2*split_size:]

# %%
pretrain_df.to_csv('voc12_pretrain.csv', index=False)
fold0.to_csv('voc12_fold_0.csv', index=False)
fold1.to_csv('voc12_fold_1.csv', index=False)
fold2.to_csv('voc12_fold_2.csv', index=False)

#%% Load saved df's for inspection
# pretrain = pd.read_csv('voc12_pretrain.csv')
# f0 = pd.read_csv('voc12_fold_0.csv')
# f1 = pd.read_csv('voc12_fold_1.csv')
# f2 = pd.read_csv('voc12_fold_2.csv')

#%%
# # Show an example
# i = 1800
# row = f2.iloc[i]
# img_path = os.path.join(voc_root, 'VOCdevkit', 'VOC2012', 'JPEGImages',
#                         row.ids+'.jpg')
# img = Image.open(img_path).convert("RGB")
# display(img)
# labels = [ind_to_label[ind] for ind in np.where(row.values==1)[0]]
# print(labels)