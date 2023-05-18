from torch.utils.data import Dataset, DataLoader

import numpy as np
import pickle as pk
import torch


class fastdataset(Dataset):
    def __init__(self, pkpath):
        self.annotate = pk.load(open(pkpath, 'rb'))
        self.black = np.zeros((512, 512, 3), dtype=np.int8) - 0.5
        
    def __getitem__(self, index):
        picid = self.annotate.iloc[index]['jpg']
        jpg = self.black
        csi = self.annotate.iloc[index]['csi']
        openpose = self.annotate.iloc[index]['openpose']
        maskrcnn = self.annotate.iloc[index]['maskrcnn']
        
        return {'openpose': openpose, 'maskrcnn': maskrcnn, 'csi': csi, 'pic': jpg}
        
    def __len__(self):
        return len(self.annotate)
    
    
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
    
    
def build_loader(args):
    traindf = '%s_train.pk' % args.prefix
    validdf = '%s_valid.pk' % args.prefix
    
    trainset = fastdataset(traindf)
    validset = fastdataset(validdf)
    
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