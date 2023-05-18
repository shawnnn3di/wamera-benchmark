import torch
import modeling
import numpy as np
import pandas as pd
import pickle as pk
import tqdm
import colorsys
import tensorboardX

from torch.utils.data import Dataset, DataLoader

from previewer import previewbatch


class fastdataset(Dataset):
    def __init__(self, jpgpk, trainvalid):
        self.df = pk.load(open(jpgpk, 'rb'))#[:1000]
        
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
        return len(self.annotate)
    
def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    j = 0
    step = 360.0 / num
    while j < num:
        h = i
        s = 90 #+ random.random() * 10
        l = 50 #+ random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step
        j += 1
    return hls_colors

def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])
    return rgb_colors, hls_colors


class Recorder_(object): # save the object value
    def __init__(self):
        self.last=0
        self.values=[]
        self.nums=[]
    def update(self,val,n=1):
        self.last=val
        self.values.append(val)
        self.nums.append(n)
    def avg(self):
        sum=np.sum(np.asarray(self.values) * np.expand_dims(np.asarray(self.nums), (1, 2)), 0)
        count=np.sum(np.asarray(self.nums))
        return sum/count


class Recorder(object): # save the object value
    def __init__(self):
        self.last=0
        self.values=[]
        self.nums=[]
    def update(self,val,n=1):
        self.last=val
        self.values.append(val)
        self.nums.append(n)
    def avg(self):
        # sum=np.sum(np.asarray(self.values) / 100000)
        # count=np.sum(np.asarray(self.nums))
        # return sum/count * 100000
        return np.asarray(self.values).mean()

    
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
    
    
if __name__ == '__main__':
    
    batch_size = 32
    device = 'cuda:0'
    
    comment = '0517_full_deconv_1_debug_l2'
    
    colors, hls = ncolors(19)
    colors[18] = [0, 0, 0]
    
    writer = tensorboardX.SummaryWriter('./tensorboard/%s' % comment, 'iterate')
    print('writer created..')
    
    traindf = '/home/lscsc/caizhijie/0420-wamera-benchmark/annotate/annotate_train.pk'
    validdf = '/home/lscsc/caizhijie/0420-wamera-benchmark/annotate/annotate_valid.pk'
    train_set = fastdataset(
        jpgpk='/home/lscsc/caizhijie/ref-rep/pytorch-openpose/dataparse/pathlist_train.pk', 
        # csipk='annotate/csi/box_csi.pk', 
        trainvalid='train',
    )
    valid_set = fastdataset(
        jpgpk='/home/lscsc/caizhijie/ref-rep/pytorch-openpose/dataparse/pathlist_valid.pk', 
        # csipk='annotate/csi/box_csi.pk', 
        trainvalid='valid',
    )
    print('dataset ready')
    
    train_loader = DataLoader(
        train_set,
        batch_size=32,
        shuffle=True,
        num_workers=12,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=12,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    print('dataloader ready')
    
    # student = modeling.student_0().to(device)
    # student_ = modeling.student_0(38).to(device)
    
    # preprocessor = modeling.preprocess_deconv().to(device)
    preprocessor = modeling.preprocess_deconv().to(device)
    student = modeling.student_0(nchannel_in=32).to(device)
    student_ = modeling.student_0(38, nchannel_in=32).to(device)
    
    optimizer = torch.optim.Adam(
        [
            {'params': preprocessor.parameters()},
            {'params': student.parameters()},
            {'params': student_.parameters()},
        ],
        lr=1e-3,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer,
        gamma=0.99,
    )
    
    num_epoch = 1000
    for j in range(num_epoch):
        lossrecorder = Recorder()
        losslist = list()
        if j % 2 == 1:
            pcks_recorder = Recorder_()
            with torch.no_grad():
                pbar = tqdm.tqdm(enumerate(valid_loader), total=len(valid_loader))
                lenpbar = len(pbar)
                for i, batch in pbar:
                    
                    x = batch['csi'].to(device)
                    y_sm = batch['sm'].to(device).float()
                    y_jhm = torch.tensor(batch['jhm']).to(device).float()
                    y_paf = torch.tensor(batch['paf']).to(device).float()
                    
                    x = preprocessor(x)
                    jhm, sm = student(x)
                    paf, _ = student_(x)
                    
                    loss_sm = (torch.abs(sm - y_sm) * (1 + y_sm.abs())).sum() * 0.1
                    loss_jhm = (torch.abs(jhm - y_jhm) * (1 + y_jhm.abs())).sum()
                    loss_paf = (torch.abs(paf - y_paf) * (0.3 + y_paf.abs())).sum()

                    loss = loss_sm + loss_jhm + loss_paf
                    lossrecorder.update(float(loss.cpu().detach().numpy().squeeze()))
                    losslist.append(float(loss))
                    
                    if i % 100 == 0:
                        arr = np.clip(np.matmul(jhm.cpu().detach().numpy().transpose(0, 3, 2, 1), np.array(colors)[:19, :]), 0, 255)
                        writer.add_images('v_jhm_pd', arr, j, dataformats='NHWC')

                        writer.add_images('v_sm_pd', sm.unsqueeze(-1), j, dataformats='NHWC')
                        
                        arr = np.clip(np.matmul(y_jhm.cpu().detach().numpy().transpose(0, 3, 2, 1), np.array(colors)[:19, :]), 0, 255)
                        writer.add_images('v_jhm_gt', arr, j, dataformats='NHWC')
                        
                        writer.add_images('v_sm_gt', y_sm.unsqueeze(-1), j, dataformats='NHWC')
                    
                    pbar.set_description('%s, epoch: %d/%d, batch: %d/%d, loss: %.4f' % ('valid', j, num_epoch, i, lenpbar, loss))
                    with open('log.txt', 'a+') as f:
                        f.write('loss: %.4f\n' % (loss))
                    
                    if i % 20 == 0:
                        img_batch, pcks = previewbatch(batch['img'], (batch['jhm'], batch['paf']), (jhm, paf), batch['box'])
                        pcks_recorder.update(pcks)
                    
        # writer.add_scalar('validloss', lossrecorder.avg(), j)
            writer.add_scalar('validloss', sum(losslist) / len(pbar), j)
            v_pcks = list(pcks_recorder.avg()[0, ...])
            k_pcks = ['%.2f' % (float(_) / 20) for _ in range(20)]
            writer.add_scalars(
                'pcks', dict(zip(k_pcks, list(v_pcks))), global_step=j)
            print(v_pcks)
            pk.dump([paf.detach().cpu().numpy(), y_paf.detach().cpu().numpy()], open('temppaf.pk', 'wb'))
        
        lossrecorder = Recorder()            
        losslist = list()
        pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
        lenpbar = len(pbar)
        for i, batch in pbar:
            
            x = batch['csi'].to(device)
            y_sm = batch['sm'].to(device).float()
            y_jhm = torch.tensor(batch['jhm']).to(device).float()
            y_paf = torch.tensor(batch['paf']).to(device).float()
            
            x = preprocessor(x)
            jhm, sm = student(x)
            paf, _ = student_(x)
            
            loss_sm = ((sm - y_sm) ** 2 * (1 + y_sm.abs())).sum() * 0.1
            loss_jhm = ((jhm - y_jhm) ** 2 * (1 + y_jhm.abs())).sum()
            # loss_paf = (torch.abs(paf - y_paf) * (0.3 + y_paf.abs())).sum()
            
            loss_paf = ((paf - y_paf) ** 2 * (0.3 + y_paf.abs())).sum()
            
            # paf_x_sum = paf[:, 0::2, ...]
            # y_paf_x_sum = y_paf[:, 0::2, ...]
            # paf_y_sum = paf[:, 1::2, ...]
            # y_paf_y_sum = y_paf[:, 1::2, ...]
            # loss_paf_assist = (
            #     (torch.abs(paf_x_sum - y_paf_x_sum) * (1 + y_paf_x_sum.abs())).sum() + 
            #     (torch.abs(paf_y_sum - y_paf_y_sum) * (1 + y_paf_y_sum.abs())).sum()
            # )
            
            # paf_sum = torch.sum((paf ** 2).sum(1)).squeeze()
            
            # loss_paf_assist = (torch.abs(paf_sum - torch.sum((y_paf ** 2).sum(1)).squeeze() * y_sm) * (1 + y_sm.abs())).to(device).float().sum()
            
            loss_paf_assist = torch.tensor(0.0).to(device)

            loss = loss_sm + loss_jhm + loss_paf + loss_paf_assist
            lossrecorder.update(float(loss.cpu().detach().numpy().squeeze()))
            losslist.append(float(loss))
            
            if i % 10 == 0:
                arr = np.clip(np.matmul(jhm.cpu().detach().numpy().transpose(0, 3, 2, 1), np.array(colors)[:19, :]), 0, 255)
                writer.add_images('t_jhm_pd', arr, j, dataformats='NHWC')
                
                writer.add_images('t_sm_pd', sm.unsqueeze(-1), j, dataformats='NHWC')
                
                arr = np.clip(np.matmul(y_jhm.cpu().detach().numpy().transpose(0, 3, 2, 1), np.array(colors)[:19, :]), 0, 255)
                writer.add_images('t_jhm_gt', arr, j, dataformats='NHWC')
                
                writer.add_images('t_sm_gt', y_sm.unsqueeze(-1), j, dataformats='NHWC')
            
            pbar.set_description('%s, epoch: %d/%d, batch: %d/%d, loss: %.4f' % ('train', j, num_epoch, i, lenpbar, loss))
            with open('log.txt', 'a+') as f:
                f.write('loss: %.4f\n' % (loss))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # writer.add_scalar('trainloss', lossrecorder.avg(), j)
        writer.add_scalar('trainloss', sum(losslist) / len(pbar), j)
            # if i % 1600 == 199:
        scheduler.step()
        
        saveperiod = 20
        if j % saveperiod == 0:
            torch.save(student, '%d_student_%s_.checkpoint' % (j, comment))
            torch.save(student, '%d_student__%s_.checkpoint' % (j, comment))
            torch.save(preprocessor, '%d_preprocessor_%s_.checkpoint' % (j, comment))
            
            