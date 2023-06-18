# %%
# %%
rootdir = '/home/lscsc/caizhijie/0420-wamera-benchmark/data/pic'

# %%
import glob
jpgs = glob.glob('%s/*/*.jpg' % rootdir)

import pandas as pd
import pickle as pk

df_jpgs = pd.DataFrame.from_dict({'jpg': jpgs})

df_jpgs['env'] = df_jpgs['jpg'].apply(lambda x: x.split('_')[0][-1])
df_jpgs['subj'] = df_jpgs['jpg'].apply(lambda x: x.split('_')[1][-1])
df_jpgs['group'] = df_jpgs['jpg'].apply(lambda x: x.split('_')[2][-1])
df_jpgs['angle'] = df_jpgs['jpg'].apply(lambda x: x.split('_')[3][-1])
df_jpgs['cam'] = df_jpgs['jpg'].apply(lambda x: x.split('_')[4][-1])
df_jpgs['t'] = df_jpgs['jpg'].apply(lambda x: x.split('_')[5][-1])

# %%
import sys

openposepath = '/home/lscsc/caizhijie/ref-rep/pytorch-openpose'
sys.path.append(openposepath)
from src import util

import cv2
import torch
import numpy as np

class prep:
    def __init__(self):
        self.scale_search = [0.5,]
        self.boxsize = 368
        self.stride = 8
        self.padValue = 128
        self.thre1 = 0.1
        self.thre2 = 0.05
    
    def __call__(self, oriImg):
        self.multiplier = [x * self.boxsize / oriImg.shape[0] for x in self.scale_search]
        self.heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
        self.paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))
        for m in range(len(self.multiplier)):
            scale = self.multiplier[m]
            imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            imageToTest_padded, pad = util.padRightDownCorner(imageToTest, self.stride, self.padValue)
            im = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 - 0.5
            im = np.ascontiguousarray(im)

            data = torch.from_numpy(im).float()

        return data

# %%
from torch.utils.data import Dataset, DataLoader

preper = prep()

class _dataset(Dataset):
    def __init__(self, df):
        self.df = df
        
    def __getitem__(self, index):
        return self.df.iloc[index]
    
    def __len__(self):
        return len(self.df)
    
def collate_fn(batch):
    imgs = [preper(cv2.imread(_['jpg'])) for _ in batch]
    paths = [_['jpg'] for _ in batch]
    return np.stack(imgs), paths

inference_dataset = _dataset(df_jpgs.sort_values('jpg', ignore_index=True))
inference_loader = DataLoader(inference_dataset, 128, shuffle=False, collate_fn=collate_fn)

print(len(inference_dataset))

# %%
from src import translator

gpuid = 1
device = 'cuda:%d' % gpuid

model_path = openposepath + '/model/body_pose_model.pth'
teacher = translator.bodypose_transparent()
model_dict = util.transfer(teacher, torch.load(model_path))
teacher.load_state_dict(model_dict)
teacher.to(device)
teacher.eval()
teacher.half()

# %% [markdown]
# single picture

# %%
import glob
import pandas as pd

# darkdir = '/home/lscsc/caizhijie/0420-wamera-benchmark/darktest/'
# jpglist = glob.glob(darkdir + 'p*.jpg')
# df = pd.DataFrame.from_dict({'jpg': jpglist})

# inference_dataset = _dataset(df)
# inference_loader = DataLoader(inference_dataset, 2, shuffle=False, collate_fn=collate_fn)

# %%
import tqdm
import torch.nn.functional as F
batch1 = None

openpose_dstrootdir = '/home/lscsc/caizhijie/0420-wamera-benchmark/data/openpose/'

for k in range(0, 60):
    kptlist = list()
    afflist = list()
    namelist = list()

    # dataset = _dataset(df_jpgs[k * 80000:(k + 1) * 80000])
    inference_dataset = _dataset(df_jpgs[k * 5000:(k + 1) * 5000])
    inference_loader = DataLoader(inference_dataset, 128, shuffle=False, num_workers=12, collate_fn=collate_fn, drop_last=False)


    for i, batch in tqdm.tqdm(enumerate(inference_loader), total=len(inference_loader)):
        kpt, aff, _ = teacher(torch.tensor(batch[0]).squeeze().to(device).half())
        kpt = F.interpolate(kpt, (54, 96))
        aff = F.interpolate(aff, (54, 96))
        kptlist.extend(kpt.detach().cpu().numpy())
        afflist.extend(aff.detach().cpu().numpy())
        namelist.extend(batch[1])
        # batch1 = batch
        # pk.dump([kpt.detach().cpu().numpy(), aff.detach().cpu().numpy(), batch[1]], open(openpose_dstrootdir + '%d_%d_batch.pk' % (k,i), 'wb'))
        
    # pk.dump([kptlist, afflist, namelist], open(openpose_dstrootdir + '%d.pk' % k, 'wb'))
    import os
    
    for i in tqdm.trange(len(namelist)):
        try:
            pk.dump({'aff':afflist[i], 'kpt':kptlist[i]}, open(openpose_dstrootdir + ('/'.join(namelist[i].split('/')[-3:])[:-4] + '.pk'), 'wb'))
        except FileNotFoundError:
            if not os.path.exists(openpose_dstrootdir + '/'.join(namelist[i].split('/')[-3:-2])):
                os.mkdir(openpose_dstrootdir + '/'.join(namelist[i].split('/')[-3:-2]))
            if not os.path.exists(openpose_dstrootdir + '/'.join(namelist[i].split('/')[-3:-1])):
                os.mkdir(openpose_dstrootdir + '/'.join(namelist[i].split('/')[-3:-1]))
            pk.dump({'aff':afflist[i], 'kpt':kptlist[i]}, open(openpose_dstrootdir + ('/'.join(namelist[i].split('/')[-3:])[:-4] + '.pk'), 'wb'))
    
    del kptlist, afflist, namelist

# %%



