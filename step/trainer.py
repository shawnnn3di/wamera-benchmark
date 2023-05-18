import argparse
import torch
import tqdm
import numpy as np

import torch.nn.functional as F
from torchvision.transforms import Resize
from torch.optim import Adam, lr_scheduler, RMSprop
from torch.utils.data import DataLoader
from itertools import cycle
from tensorboardX import SummaryWriter

# from dataparse import dataset, collate_fn
from dataparse import collate_fn
from dataparse import fastdataset_normcsi as dataset


from modeling import student_0, combine_translator
from previewer import previewbatch
from utils import Recorder, add_args, matthewloss, l2loss, l1loss, dice_loss

class trainer:
    def __init__(
        self,
        student0,
        translator,
        batch_size=32,
        shuffle_train=True,
        num_worker=8,
        lr=1e-3,
        weight_decay=1e-4,
        criterion_fn=None,
        num_epoch=300,
        half=True,
        valid_gap=1,
        preview_gap=100,
        writer=None,
        gpuid=0,
        len_epoch_train=2000,
        len_epoch_valid=100,
        accum_step=1,
        prefix=None,
    ):
        self.cuda_available = torch.cuda.is_available()
        self.device = 'cuda:%d' % gpuid if self.cuda_available else 'cpu'
        print(self.device)
        
        assert student0 is not None,     'direct mode needs a student'
        student0 = student0.to(self.device)
        student0.train()
        translator = translator.to(self.device)
        translator.train()
        
        self.resizer = Resize((46, 46))
        
        if half:
            student0 = student0.half()
            translator = translator.half()
            
        self.optimizer = Adam(
            [
                {'params': student0.parameters()},
                {'params': translator.parameters()},
            ],
            lr=lr,
            weight_decay=weight_decay,
        )
        self.scheduler = lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=0.9999,
        )
        
        assert criterion_fn is not None,    'either mode needs a criterion function'
        self.criterion_fn = criterion_fn
        
        self.halving = lambda x: torch.stack([_.to(self.device) for _ in x]).half() if half else torch.stack([_.to(self.device) for _ in x])
        print('model ready')
        
        train_set = dataset(
            jpgpk='/home/lscsc/caizhijie/ref-rep/pytorch-openpose/dataparse/pathlist_train.pk', 
            trainvalid='train',
        )
        valid_set = dataset(
            jpgpk='/home/lscsc/caizhijie/ref-rep/pytorch-openpose/dataparse/pathlist_valid.pk', 
            trainvalid='valid',
        )
        print('dataset ready')
        
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_worker,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        valid_loader = DataLoader(
            valid_set,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_worker,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        print('dataloader ready')
        
        len_valid = len(valid_loader)
        len_train = len(train_loader)
        
        self.train_lastend = 0
        self.valid_lastend = 0
        
        varnamelist = [
            'student0',
            # 'student1',
            'translator',
            # 'student2',
            'batch_size',
            'shuffle_train',
            'num_worker',
            'lr',
            'weight_decay',
            'criterion_fn',
            'num_epoch',
            'half',
            'valid_gap',
            'preview_gap',
            'gpuid',
            'len_epoch_train',
            'len_epoch_valid',
            'accum_step',
            'prefix',
            'len_valid',
            'len_train',
            'train_loader',
            'valid_loader',
            'writer',
        ]
        for varname in varnamelist:
            setattr(self, varname, eval(varname))
            
        self.sm_mean, self.sm_std = (0.063757055, 0.24431965)
            
    def run(self):
        for i in range(self.num_epoch):
            if i % self.valid_gap == 0:
                self.valid(i)
            self.train(i)
            
    def train(self, epoch):
        self.iterate(train=True, epoch=epoch)
        
    def valid(self, epoch):
        with torch.no_grad():
            self.iterate(train=False, epoch=epoch)
            
    def iterate(self, train=True, epoch=-1):
        loss_recorder = Recorder()
        pcks_recorder = Recorder()
        
        trainvalid = 'train' if train else 'valid'
        if train:
            len_loader = self.len_epoch_train
            loader = self.train_loader
            pbar = tqdm.trange(self.len_epoch_train)
            lastend = self.train_lastend
        else:
            len_loader = self.len_epoch_valid
            loader = self.valid_loader
            pbar = tqdm.trange(self.len_epoch_valid)
            lastend = self.valid_lastend
        
        for i, batch in tqdm.tqdm(enumerate(loader), total=len(loader)):

            amp, pha = self.halving([_.float() for _ in batch['csi']])
            input = self.translator([amp, pha])
            
            sm = self.student0(input)
            
            sm_ = self.halving(batch['sm'].float())
            sm_ = sm_ - self.sm_mean
            sm_ = sm_ / self.sm_std
            
            # jhm_ = self.halving(torch.tensor(batch['jhm'], dtype=jhm.dtype, device=jhm.device))
            # jhm_ = jhm_ - self.jhm_mean
            # jhm_ = jhm_ / self.jhm_std
            
            # paf_ = self.halving(torch.tensor(batch['paf'], dtype=jhm.dtype, device=jhm.device))
            # paf_ = paf_ - self.paf_mean
            # paf_ = paf_ / self.paf_std
            
            loss1 = self.criterion_fn(self.resizer(sm).squeeze(), self.halving(batch['sm'].float()), k=1, b=0.3)
            # loss1 = F.binary_cross_entropy_with_logits(sm.squeeze(), self.resizer(self.halving(batch['sm'])).float())
            loss = loss1
            loss_recorder.update(loss.detach().cpu().numpy())
            
            pbar.set_description('%s, epoch: %d/%d, batch: %d/%d, loss: %.4f' % (trainvalid, epoch, self.num_epoch, i, len_loader, loss))
            
            if i % self.preview_gap == 0:
                print('previewing..')                
                if epoch % 1 == 0:
                    self.writer.add_images('%s_pdimage' % trainvalid, 255 * sm.cpu().detach().numpy().astype(np.uint8), epoch, dataformats='NCHW')
                    self.writer.add_images('%s_gtimage' % trainvalid, 255 * batch['sm'].unsqueeze(1).cpu().detach().numpy(), epoch, dataformats='NCHW')
                
            if train:
                loss.backward()
                if (i + 1) % self.accum_step == 0:
                    self.optimizer.zero_grad()
                    self.optimizer.step()
            
        self.writer.add_scalars(
            'loss', {'%s_loss_epoch' % trainvalid: np.mean(loss_recorder.avg())}, global_step=epoch)
        
        v_pcks = list(pcks_recorder.avg()[0, ...])
        k_pcks = ['%.2f' % (float(_) / 20) for _ in range(20)]
        self.writer.add_scalars(
            'pcks', dict(zip(k_pcks, list(v_pcks))), global_step=epoch)
        
        if train:
            self.writer.add_scalar('lr', self.scheduler.get_last_lr()[0], epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()

    task_save_root = '%s-%s-%s-%s-%s' % (args.batch_size, args.lr, args.weight_decay, args.len_epoch_train, args.comment)
    writer = SummaryWriter('tensorboard/'+task_save_root)
    
    s_0 = student_0()
    combine_translator = combine_translator()

    _trainer = trainer(
        s_0,
        combine_translator,
        args.batch_size,
        args.shuffle_train,
        args.num_workers,
        args.lr,
        args.weight_decay,
        matthewloss,
        # l1loss,
        # dice_loss,
        args.num_epoch,
        args.half,
        args.valid_gap,
        args.preview_gap,
        writer,
        args.gpuid,
        args.len_epoch_train,
        args.len_epoch_valid,
        args.accum_step,
        args.prefix
    )
    
    _trainer.run()