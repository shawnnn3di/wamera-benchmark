from torch.utils.data import Dataset

import cv2
import torch
import pandas as pd
import pickle as pk
import numpy as np

import glob
import tqdm
import time


class fastdataset_normcsi(Dataset):
    def __init__(self, jpgpk, trainvalid):
        self.df = pk.load(open(jpgpk, 'rb'))
        self.maskrcnndir = '/home/lscsc/caizhijie/archive_0118/maskrcnn'
        self.annotate = pk.load(open('/home/lscsc/caizhijie/0420-wamera-benchmark/annotate/annotate_%s.pk' % trainvalid, 'rb'))
        self.black = np.zeros((512, 512, 3), dtype=np.int8) - 0.5
        
    def __getitem__(self, index):
        picid = self.annotate.iloc[index]['jpg']
        jpg = self.black
        csi = pk.load(open(picid[:-4] + '.pk_2_1', 'rb'))
        openpose = self.annotate.iloc[index]['openpose']
        maskrcnn = self.annotate.iloc[index]['maskrcnn']
        
        return {'openpose': openpose, 'maskrcnn': maskrcnn, 'csi': csi, 'pic': jpg}
        
    def __len__(self):
        return len(self.df)
    

def collate_fn(batch):
    jhm = list()
    aff = list()
    box = list()
    mask = list()
    amp = list()
    pha = list()
    pic = list()
    
    for b in batch:
        jhm.append(b['openpose']['aff'])
        aff.append(b['openpose']['kpt'])
        # _b = b['maskrcnn'].get_fields()
        # box.append(_b['pred_boxes'][0].tensor.float())
        # mask.append(torch.tensor(cv2.resize(_b['pred_masks'][0].float().numpy(), (46, 46))))
        box.append(b['maskrcnn'][0].tensor)
        mask.append(torch.tensor(b['maskrcnn'][1]))
        amp.append(b['csi'][0][:12, ...])
        pha.append(b['csi'][1][:12, ...])
        pic.append(b['pic'])
        
        
    jhm = np.stack(jhm)
    aff = np.stack(aff)
    box = torch.stack(box)
    mask = torch.stack(mask)
    amp = torch.tensor(np.stack(amp))
    pha = torch.tensor(np.stack(pha))
    csi = [amp, pha]
    
    
    return dict(zip(['jhm', 'paf', 'box', 'sm', 'csi', 'img'], [jhm, aff, box, mask, csi, pic]))