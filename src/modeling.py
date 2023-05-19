import torch
import torch.nn as nn

from torchvision.transforms import Resize
from torch.optim import Adam, lr_scheduler


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )
    
    
class preprocess_deconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # 32x in size
        self.preprocess = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, 3, 2, 1, 1),
            nn.BatchNorm2d(120),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels, in_channels, 3, 2, 1, 1),
            nn.BatchNorm2d(120),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels, in_channels, 3, 2, 1, 1),
            nn.BatchNorm2d(120),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels, in_channels, 3, 2, 1, 1),
            nn.BatchNorm2d(120),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels, in_channels, 3, 2, 1, 1),
            nn.BatchNorm2d(120),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
        )
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
    def forward(self, x):
        x = x.reshape(x.shape[0], self.in_channels, 3, 3)
        return self.preprocess(x)
    

class preprocess_reslinear(nn.Module):
    def __init__(self, in_channels, mid_channels, outsize=(96, 96)):
        super().__init__()
        
        self.preprocess_0 = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.ReLU(),
        )
        
        self.preprocess_1 = nn.Sequential(
            nn.Linear(mid_channels, mid_channels),
            nn.ReLU(),
            nn.Linear(mid_channels, mid_channels),
            nn.ReLU(),
            nn.Linear(mid_channels, mid_channels),
            nn.ReLU(),
            nn.Linear(mid_channels, mid_channels),
            nn.ReLU(),
        )
        
        self.preprocess_2 = nn.Sequential(
            nn.Linear(mid_channels, mid_channels),
            nn.ReLU(),
            nn.Linear(mid_channels, outsize[0] * outsize[1]),
        )
        
        self.outsize = outsize
        
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x0 = self.preprocess_0(x)
        x1 = self.preprocess_1(x0)
        x = self.preprocess_2(x0 + x1)
        x = x.reshape(x.shape[0], 1, self.outsize[0], self.outsize[1])
        return x
    

class unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=19):
        super().__init__()
        
        c = [2 ** _ for _ in range(5, 11)]
        
        self.startconv = double_conv(in_channels, c[0])
        
        self.dconv_down0 = double_conv(c[0], c[1])
        self.dconv_down1 = double_conv(c[1], c[2])
        self.dconv_down2 = double_conv(c[2], c[3])
        self.dconv_down3 = double_conv(c[3], c[4])
        self.dconv_down4 = double_conv(c[4], c[5])
        
        self.avgpool = nn.AvgPool2d(2)
        self.upsample4 = nn.ConvTranspose2d(c[5], c[4], 3, stride=2, padding=1, output_padding=1)
        self.upsample3 = nn.ConvTranspose2d(c[4], c[3], 3, stride=2, padding=1, output_padding=1)
        self.upsample2 = nn.ConvTranspose2d(c[3], c[2], 3, stride=2, padding=1, output_padding=1)
        self.upsample1 = nn.ConvTranspose2d(c[2], c[1], 3, stride=2, padding=1, output_padding=1)
        self.upsample0 = nn.ConvTranspose2d(c[1], c[0], 3, stride=2, padding=1, output_padding=1)
        
        self.dconv_up3 = double_conv(c[5], c[4])
        self.dconv_up2 = double_conv(c[4], c[3])
        self.dconv_up1 = double_conv(c[3], c[2])
        self.dconv_up0 = double_conv(c[2], c[1])
        
        self.endconv = nn.Conv2d(c[0], c[0], 1)
        
        self.resizer0 = Resize((96, 96))
        self.resizer1 = Resize((46, 46))
        
        self.postprocess_0 = nn.Sequential(
            nn.Conv2d(32, 32, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, 1, 1),
            nn.BatchNorm2d(out_channels),
        )
        
    def forward(self, x):
        x = self.startconv(x)
        
        # encode
        conv0 = self.dconv_down0(x)
        x = self.avgpool(conv0)
        
        conv1 = self.dconv_down1(x)
        x = self.avgpool(conv1)
        
        conv2 = self.dconv_down2(x)
        x = self.avgpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.avgpool(conv3)

        x = self.dconv_down4(x)
        
        # decode
        x = self.upsample4(x)

        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample3(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample2(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)
        x = self.upsample1(x)
        x = torch.cat([x, conv0], dim=1)

        x = self.dconv_up0(x)
        x = self.upsample0(x)

        out = self.endconv(x)
        
        out = self.postprocess_0(out)
        out = self.resizer1(out)
        
        return out
    
    
def forward_pipeline(batch, device, model, half):
    x = batch['csi'].to(device)
    y_sm = batch['sm'].to(device)
    y_jhm = torch.tensor(batch['jhm']).to(device)
    y_paf = torch.tensor(batch['paf']).to(device)
    if not half:
        y_sm = y_sm.float()
        y_jhm = torch.tensor(batch['jhm']).to(device).float()
        y_paf = torch.tensor(batch['paf']).to(device).float()
    x = model[0](x)
    jhm = model[1](x)
    paf = model[2](x)
    sm = jhm[:, :-1, ...].sum(1)
    
    loss_sm = ((sm - y_sm) ** 2 * (1 + y_sm.abs())).sum() * 0.1
    loss_jhm = ((jhm - y_jhm) ** 2 * (1 + y_jhm.abs())).sum()    
    loss_paf = ((paf - y_paf) ** 2 * (0.3 + y_paf.abs())).sum()
    
    return loss_sm, loss_jhm, loss_paf, sm, jhm, paf, y_sm, y_jhm, y_paf, batch['img'], batch['box']


def build_model(args, device, checkpoint=None):
    if args.style == 'pipeline':
        if checkpoint == None:
            print('initializing new model..')
            preprocessor = preprocess_deconv(120, 32).to(device)
            student_jhm = unet(32, 19).to(device)
            student_paf = unet(32, 38).to(device)
        
        else:
            print('loading checkpoint from %s' % checkpoint)
            [preprocessor, student_jhm, student_paf] = [_.to(device) for _ in torch.load(checkpoint)]
            
        model = [preprocessor, student_jhm, student_paf]
        optimizer = Adam(
                [
                    {'params': preprocessor.parameters()},
                    {'params': student_jhm.parameters()},
                    {'params': student_paf.parameters()},
                ],
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
            
        scheduler = lr_scheduler.ExponentialLR(
            optimizer=optimizer, 
            gamma=args.lr_gamma,
        )
        forward = forward_pipeline
        
    return model, optimizer, scheduler, forward