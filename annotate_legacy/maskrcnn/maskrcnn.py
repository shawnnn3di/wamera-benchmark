# %%
# %%
rootdir = '/home/lscsc/caizhijie/shar-data/archive'

# %%
import glob
jpgs = glob.glob('%s/*/*/*.jpg' % rootdir)
pks = glob.glob('%s/*/*/*.pk_2_1' % rootdir)

import pandas as pd
import pickle as pk

df_jpgs = pd.DataFrame.from_dict({'jpg': jpgs})
df_jpgs['subdir'] = df_jpgs['jpg'].apply(lambda x: x.split('/')[-2])
df_jpgs['dir'] = df_jpgs['jpg'].apply(lambda x: x.split('/')[-3])
df_jpgs = df_jpgs[df_jpgs['subdir'].apply(lambda x: x[0] == 't')]

df_jpgs['mov'] = df_jpgs['dir'].apply(lambda x: x.split('_')[2])
df_jpgs['obj'] = df_jpgs['dir'].apply(lambda x: x.split('_')[3])
df_jpgs['ang'] = df_jpgs['dir'].apply(lambda x: x.split('_')[4])
df_jpgs['rep'] = df_jpgs['subdir'].apply(lambda x: x.split('_')[0].split('t')[-1])



# %%
# df_jpgs

# %%
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import torch

import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library

yamlfile = 'configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml'
# yamlfile = 'configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml'

cfg.merge_from_file(yamlfile)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(yamlfile[11:])
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(yamlfile[8:])

# %%
from torch.utils.data import Dataset, DataLoader

import cv2
import torch
import tqdm
import numpy as np

class _dataset(Dataset):
    def __init__(self, df):
        self.df = df
        
    def __getitem__(self, index):
        return self.df.iloc[index]
    
    def __len__(self):
        return len(self.df)
    
def collate_fn(batch):
    imgs = [cv2.imread(_['jpg']) for _ in batch]
    paths = [_['jpg'] for _ in batch]
    return np.stack(imgs), paths


class packPredictor(DefaultPredictor):
    def __call__(self, pack):
        # 'NHWC'
        assert len(pack.shape) == 4, 'A default predictor is qualified'
        with torch.no_grad():
            packimage = list()
            packheight = list()
            packwidth = list()
            
            for i in range(pack.shape[0]):
                thisimage = pack[i, ...]
                if self.input_format == 'RGB':
                    thisimage = thisimage[..., ::-1]
                    
                # print(thisimage.shape)
                height, width = thisimage.shape[:2]
                image = self.aug.get_transform(thisimage).apply_image(thisimage)
                image = torch.as_tensor(image.astype('float32').transpose(2, 0, 1))
                
                packimage.append(image)
                packheight.append(height)
                packwidth.append(width)
            
            inputs = [{'image': packimage[_], 'height': packheight[_], 'width': packwidth[_]} for _ in range(len(packimage))]
            predictions = self.model(inputs)
            return predictions

# %%
# %%
openpose_dstrootdir = '/home/lscsc/caizhijie/archive_0118/maskrcnn/'

import os

for k in range(4):
    dataset = _dataset(df_jpgs[k * 20000:(k + 1) * 20000])
    inference_loader = DataLoader(dataset, 32, shuffle=False, collate_fn=collate_fn, num_workers=8, pin_memory=True)
    ppredictor = packPredictor(cfg)
    
    outputlist = list()
    namelist = list()
    for i, batch in tqdm.tqdm(enumerate(inference_loader), total=len(inference_loader)):
        output = ppredictor(batch[0])
        outputlist.extend([_['instances'].to('cpu') for _ in output])
        
        # outputlist.extend([1 for _ in range(len(batch[0]))])
        
        namelist.extend(batch[1])
        
    for i in tqdm.trange(len(namelist)):
        try:
            pk.dump(outputlist[i], open(openpose_dstrootdir + ('/'.join(namelist[i].split('/')[-3:])[:-4] + '.pk'), 'wb'))
        except FileNotFoundError:
            if not os.path.exists(openpose_dstrootdir + '/'.join(namelist[i].split('/')[-3:-2])):
                os.mkdir(openpose_dstrootdir + '/'.join(namelist[i].split('/')[-3:-2]))
            if not os.path.exists(openpose_dstrootdir + '/'.join(namelist[i].split('/')[-3:-1])):
                os.mkdir(openpose_dstrootdir + '/'.join(namelist[i].split('/')[-3:-1]))
            pk.dump(outputlist[i], open(openpose_dstrootdir + ('/'.join(namelist[i].split('/')[-3:])[:-4] + '.pk'), 'wb'))
    
    print(len(outputlist))
    print(len(namelist))
    del outputlist

# %%



