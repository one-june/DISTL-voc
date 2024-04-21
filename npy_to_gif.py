#%%
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# %%
img_name = '2007_000129'
npys_dir = 'training_attention_maps/split-10-30-30-30/fold0'
fnames = os.listdir(npys_dir)
fnames = sorted([file for file in fnames if img_name in file], key=lambda x: (len(x), x))
arrs = [np.load(os.path.join(npys_dir, npy)) for npy in fnames]

head = 0
arrs = [arr[head] for arr in arrs]

#%%
def update(i):
    plt.clf()
    plt.imshow(arrs[i], cmap='viridis')
    plt.title(fnames[i])

fig, ax = plt.subplots()
animation = FuncAnimation(fig, update, frames=range(len(arrs)), interval=400, repeat=False)
animation.save(f'{img_name}.gif', writer='imagemagick')
