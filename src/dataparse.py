from torch.utils.data import Dataset

import cv2
import torch
import pandas as pd
import pickle as pk
import numpy as np

import glob
import tqdm
import time


class dataset(Dataset):
    def __init__(self, jpgpk, trainvalid):
        self.df = pk.load(open(jpgpk, 'rb'))
        if trainvalid == 'train':
            self.df = self.df.loc[:int(len(self.df) * 0.8)]
        else:
            self.df = self.df.loc[int(len(self.df) * 0.8):]
        
    def __getitem__(self, index):
        openposedir = '/home/lscsc/caizhijie/archive_0118/openpose/'
        maskrcnndir = '/home/lscsc/caizhijie/archive_0118/maskrcnn/'
        csislicedir = '/home/lscsc/caizhijie/archive_0118/csislice/'
        
        picid = self.df.iloc[index]['jpg']
        jpg = cv2.imread(picid)
        
        openpose = pk.load(open(openposedir + '/'.join(picid.split('/')[-3:])[:-4] + '.pk', 'rb'))
        maskrcnn = pk.load(open(maskrcnndir + '/'.join(picid.split('/')[-3:])[:-4] + '.pk', 'rb'))
        csi = pk.load(open((csislicedir + '/'.join([picid.split('/')[-3], picid.split('/')[-1]])[:-4]).replace('pic', 'csislice') + '.pk', 'rb'))
        
        return {'openpose': openpose, 'maskrcnn': maskrcnn, 'csi': csi, 'pic': jpg}
        
    def __len__(self):
        return len(self.df)
    
    
class fastdataset(Dataset):
    def __init__(self, jpgpk, trainvalid):
        self.df = pk.load(open(jpgpk, 'rb'))#[:1000]
        
        openposedir = '/home/lscsc/caizhijie/archive_0118/openpose/'
        self.maskrcnndir = '/home/lscsc/caizhijie/archive_0118/maskrcnn/'
        csislicedir = '/home/lscsc/caizhijie/archive_0118/csislice/'
        
        self.annotate = pk.load(open('/home/lscsc/caizhijie/0420-wamera-benchmark/annotate/annotate_%s.pk' % trainvalid, 'rb'))#[:1000]
        
        self.black = np.zeros((512, 512, 3), dtype=np.int8) - 0.5
        
    def __getitem__(self, index):
        picid = self.annotate.iloc[index]['jpg']
        # jpg = cv2.imread(picid)
        jpg = self.black
        csi = self.annotate.iloc[index]['csi']
        openpose = self.annotate.iloc[index]['openpose']
        maskrcnn = self.annotate.iloc[index]['maskrcnn']
        
        # maskrcnn = pk.load(open(self.maskrcnndir + '/'.join(picid.split('/')[-3:])[:-4] + '.pk', 'rb'))
        
        return {'openpose': openpose, 'maskrcnn': maskrcnn, 'csi': csi, 'pic': jpg}
        
    def __len__(self):
        return len(self.df)
        
    
def collate_fn(batch):
    jhm = list()
    aff = list()
    box = list()
    mask = list()
    csi = list()
    pic = list()
    
    for b in batch:
        jhm.append(b['openpose']['aff'])
        aff.append(b['openpose']['kpt'])
        # _b = b['maskrcnn'].get_fields()
        # box.append(_b['pred_boxes'][0].tensor.float())
        # mask.append(torch.tensor(cv2.resize(_b['pred_masks'][0].float().numpy(), (46, 46))))
        box.append(b['maskrcnn'][0].tensor)
        mask.append(torch.tensor(b['maskrcnn'][1]))
        csi.append(torch.tensor(b['csi']).abs().squeeze().permute(1, 0, 2).reshape((-1, 3)).float())
        pic.append(b['pic'])
        
        
    jhm = np.stack(jhm)
    aff = np.stack(aff)
    box = torch.stack(box)
    mask = torch.stack(mask)
    csi = torch.stack(csi)
    
    return dict(zip(['jhm', 'paf', 'box', 'sm', 'csi', 'img'], [jhm, aff, box, mask, csi, pic]))