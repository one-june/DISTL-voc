#%%
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# %%
img_name = '2007_000129'
npys_dir = 'attention_maps_progression/split-10-30-30-30/fold0'
save_dir = 'attention_maps_progression/split-10-30-30-30/fold0-pngs'
#%%
fnames = os.listdir(npys_dir)
fnames = sorted([file for file in fnames if img_name in file], key=lambda x: (len(x), x))
# arrs = [np.load(os.path.join(npys_dir, npy)) for npy in fnames]

#%%
for j, fname in enumerate(tqdm(fnames)):
    
    arr = np.load(os.path.join(npys_dir, fname))
    
    fig, axs = plt.subplots(2,3, figsize=(9,6))
    axs = axs.flatten()

    for h in range(6):
        axs[h].imshow(arr[h])
        axs[h].axis('off')
    plt.suptitle(fname[:-4])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, fname[:-4]+'.png'),
                bbox_inches='tight',
                pad_inches=0)
    plt.close()
# %%