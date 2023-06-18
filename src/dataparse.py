from torch.utils.data import Dataset, DataLoader

import cv2
import glob
import numpy as np
import pandas as pd
import pickle as pk
import torch

class bigdataset(Dataset):
    def __init__(self, df, ratio=1):
        self.df = df
        self.black = np.zeros((512, 512, 3), dtype=np.int8) - 0.5
        
    def __getitem__(self, index):
        # jpg = cv2.imread(self.df.iloc[index]['jpg'])
        jpg = self.black
        # openpose = [pk.load(open(self.df.iloc[index]['piclist_cam%d' % cam].replace('../../data', 'data/openpose')[:-4] + '.pk', 'rb')) for cam in range(4)]
        # maskrcnn = [pk.load(open(self.df.iloc[index]['piclist_cam%d' % cam].replace('../../data', 'data/maskrcnn')[:-4] + '.pk', 'rb')) for cam in range(3)]
        
        maskrcnn = None
        # csi = [self.df.iloc[index]['csi_rx%d' % rx] for rx in range(3)]
        # csi = pk.load(open(self.df.iloc[index]['piclist_cam0'][6:-3].replace('pic', 'csislice/pic') + 'pk', 'rb'))
        
        # self.df['piclist_cam0']
        
        a, b, c, d, e, f, g = pk.load(open(self.df.iloc[index]['name'], 'rb'))
        openpose = [a, b, c, d]
        csi = [e, f, g]
        return {'openpose': openpose, 'maskrcnn': maskrcnn, 'csi': csi, 'pic': jpg, 'name': self.df.iloc[index]['name']}
    
    def __len__(self):
        return len(self.df)
    
    
def collate_fn(batch):
    jhm = list()
    aff = list()
    box = list()
    mask = list()
    csi = list()
    pic = list()
    name = list()
    
    for b in batch:
        # jhm.append(b['openpose']['aff'])
        jhm.append(np.vstack([b['openpose'][_]['aff'] for _ in range(4)]))
        # aff.append(b['openpose']['kpt'])
        aff.append(np.vstack([b['openpose'][_]['kpt'] for _ in range(4)]))
        # box.append(b['maskrcnn'][0].tensor)
        box.append(torch.tensor([0, 0, 0, 0]))
        # mask.append(torch.tensor(b['maskrcnn'][1]))
        mask.append(torch.tensor([0, 0, 0, 0]))
        # csi.append(torch.tensor(b['csi']).abs().squeeze().permute(1, 0, 2).reshape((-1, 3)).float())
        csi.append(torch.tensor(np.vstack(b['csi']).reshape((150, 90, 3)).transpose(2, 0, 1)).abs())
        pic.append(b['pic'])
        name.append(b['name'])
    
    jhm = np.stack(jhm)
    aff = np.stack(aff)
    box = torch.stack(box)
    mask = torch.stack(mask)
    csi = torch.stack(csi)
    
    return dict(zip(['jhm', 'paf', 'box', 'sm', 'csi', 'img', 'name'], [jhm, aff, box, mask, csi, pic, name]))


def build_loader(args, validonly=False, ratio=1):
    # df = pk.load(open('/home/lscsc/caizhijie/0420-wamera-benchma
    
    # traindf = pd.DataFrame.from_dict({'name': glob.glob('/home/lscsc/caizhijie/0420-wamera-benchmark/data/frames/' + 'pic_*subj[2356]*.pk')}).sample(frac=1.0)
    # validdf = pd.DataFrame.from_dict({'name': glob.glob('/home/lscsc/caizhijie/0420-wamera-benchmark/data/frames/' + 'pic_*subj0*.pk')}).sample(frac=1.0)
    
    traindf = pd.DataFrame.from_dict({'name': glob.glob('/home/lscsc/caizhijie/0420-wamera-benchmark/data/frames/' + 'pic_*subj[2356]*.pk')}).sample(frac=1.0)
    validdf = pd.DataFrame.from_dict({'name': glob.glob('/home/lscsc/caizhijie/0420-wamera-benchmark/data/frames/' + 'pic_*subj*.pk')})
    validdf['time'] = validdf['name'].apply(lambda x: x[-26:])
    validdf.sort_values('time', inplace=True, ignore_index=True)
    validdf = validdf.iloc[::20]
    
    # traindf = df.iloc[:int(len(df) * 0.8)]
    # validdf = df.iloc[int(len(df) * 0.8):]
        
    validset = bigdataset(validdf, ratio)
    train_loader = None
        
    if not validonly:
        trainset = bigdataset(traindf, ratio)
        train_loader = DataLoader(
            trainset,
            batch_size=args.batchsize,
            shuffle=args.shuffle_train,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
    
    valid_loader = DataLoader(
        validset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    return train_loader, valid_loader