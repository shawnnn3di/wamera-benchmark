import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms import Resize


class Conv(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(

            nn.Conv2d(C_in, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            # 防止过拟合
            nn.Dropout(0.3),
            nn.LeakyReLU(),

            nn.Conv2d(C_out, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            # 防止过拟合
            nn.Dropout(0.4),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)


# 下采样模块
class DownSampling(nn.Module):
    def __init__(self, C):
        super(DownSampling, self).__init__()
        self.Down = nn.Sequential(
            # 使用卷积进行2倍的下采样，通道数不变
            nn.Conv2d(C, C, 3, 2, 1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.Down(x)


# 上采样模块
class UpSampling(nn.Module):

    def __init__(self, C):
        super(UpSampling, self).__init__()
        # 特征图大小扩大2倍，通道数减半
        self.Up = nn.Conv2d(C, C // 2, 1, 1)

    def forward(self, x, r):
        # 使用邻近插值进行下采样
        up = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.Up(up)
        # 拼接，当前上采样的，和之前下采样过程中的
        return torch.cat((x, r), 1)


# 主干网络
class _student_0(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.preprocess = nn.Sequential(
            nn.Linear(1080, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2304)
        )

        # 4次下采样
        self.C1 = Conv(1, 64)
        self.D1 = DownSampling(64)
        self.C2 = Conv(64, 128)
        self.D2 = DownSampling(128)
        self.C3 = Conv(128, 256)
        self.D3 = DownSampling(256)
        self.C4 = Conv(256, 512)
        self.D4 = DownSampling(512)
        self.C5 = Conv(512, 1024)

        # 4次上采样
        self.U1 = UpSampling(1024)
        self.C6 = Conv(1024, 512)
        self.U2 = UpSampling(512)
        self.C7 = Conv(512, 256)
        self.U3 = UpSampling(256)
        self.C8 = Conv(256, 128)
        self.U4 = UpSampling(128)
        self.C9 = Conv(128, 64)

        self.Th = torch.nn.Sigmoid()
        self.pred = torch.nn.Conv2d(64, 19, 3, 1, 1)
        
        self.resizer = Resize((96, 96))
        self.resizer1 = Resize((46, 46))

    def forward(self, x):
        x = self.preprocess(x)
        x = x.reshape((x.shape[0], 48, 48)).unsqueeze(1)
        x = self.resizer(x)
        
        # 下采样部分
        R1 = self.C1(x)
        R2 = self.C2(self.D1(R1))
        R3 = self.C3(self.D2(R2))
        R4 = self.C4(self.D3(R3))
        Y1 = self.C5(self.D4(R4))

        # 上采样部分
        # 上采样的时候需要拼接起来
        O1 = self.C6(self.U1(Y1, R4))
        O2 = self.C7(self.U2(O1, R3))
        O3 = self.C8(self.U3(O2, R2))
        O4 = self.C9(self.U4(O3, R1))

        # 输出预测，这里大小跟输入是一致的
        # 可以把下采样时的中间抠出来再进行拼接，这样修改后输出就会更小
        
        x = self.resizer1(self.pred(O4))
        
        return torch.clip(x, 0, 1), x[:, :18, ...].sum(1)
    

def double_conv(in_channels, out_channels):  
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),  
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class preprocess_deconv(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.preprocess_0 = nn.Sequential(
            nn.ConvTranspose2d(120, 120, 3, 2, 1, 1),
            nn.BatchNorm2d(120),
            nn.ReLU(),
            nn.ConvTranspose2d(120, 120, 3, 2, 1, 1),
            nn.BatchNorm2d(120),
            nn.ReLU(),
            nn.ConvTranspose2d(120, 120, 3, 2, 1, 1),
            nn.BatchNorm2d(120),
            nn.ReLU(),
            nn.ConvTranspose2d(120, 120, 3, 2, 1, 1),
            nn.BatchNorm2d(120),
            nn.ReLU(),
            nn.ConvTranspose2d(120, 120, 3, 2, 1, 1),
            nn.BatchNorm2d(120),
            nn.ReLU(),
            nn.Conv2d(120, 32, 1),
            nn.BatchNorm2d(32),
            # nn.ReLU(),
        )
        
    def forward(self, x):
        x = x.reshape(x.__len__(), 120, 3, 3)
        return self.preprocess_0(x)
    
class preprocess_reslinear(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.preprocess_0 = nn.Sequential(
            nn.Linear(1080, 2048),
            nn.ReLU(),
        )
        
        self.preprocess_1 = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
        )
        
        self.preprocess_2 = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 9216),
        )
        
    def forward(self, x):
        x = x.reshape(x.__len__(), -1)
        x0 = self.preprocess_0(x)
        x1 = self.preprocess_1(x0)
        x = self.preprocess_2(x0 + x1)
        x = x.reshape(x.shape[0], 1, 96, 96)
        return x
    
    
class student_0(nn.Module):
    def __init__(self, nchannel=19, nchannel_in=1):
        super().__init__()
        
        c = [32, 64, 128, 256, 512, 1024]
        
        self.startconv = double_conv(nchannel_in, c[0])

        self.dconv_down0 = double_conv(c[0], c[1])
        self.dconv_down1 = double_conv(c[1], c[2])
        self.dconv_down2 = double_conv(c[2], c[3])
        self.dconv_down3 = double_conv(c[3], c[4])
        self.dconv_down4 = double_conv(c[4], c[5])

        self.maxpool = nn.AvgPool2d(2)
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.ConvTranspose2d(c[5], c[4], 3, stride=2, padding=1, output_padding=1)
        self.upsample3 = nn.ConvTranspose2d(c[4], c[3], 3, stride=2, padding=1, output_padding=1)
        self.upsample2 = nn.ConvTranspose2d(c[3], c[2], 3, stride=2, padding=1, output_padding=1)
        self.upsample1 = nn.ConvTranspose2d(c[2], c[1], 3, stride=2, padding=1, output_padding=1)
        self.upsample0 = nn.ConvTranspose2d(c[1], c[0], 3, stride=2, padding=1, output_padding=1)

        self.dconv_up3 = double_conv(c[5], c[4])
        self.dconv_up2 = double_conv(c[4], c[3])
        self.dconv_up1 = double_conv(c[3], c[2])
        self.dconv_up0 = double_conv(c[2], c[1])

        self.conv_last = nn.Conv2d(c[0], c[0], 1)
        
        self.resizer0 = Resize((96, 96))
        self.resizer1 = Resize((46, 46))
        
        self.postprocess_0 = nn.Sequential(
            nn.Conv2d(32, 32, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, nchannel, 1, 1),
            nn.BatchNorm2d(nchannel),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        
        
        # x1 = self.preprocess_1(x0)
        # x = self.preprocess_2(x0 + x1)
        
        # x = x.reshape((x.shape[0], 48, 48))
        # x = self.resizer0(x).unsqueeze(1)
        # x = x.reshape((x.shape[0], 96, 96))
        # x = x.unsqueeze(1)

        x = self.startconv(x)
        conv0 = self.dconv_down0(x)
        x = self.maxpool(conv0)

        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

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

        out = self.conv_last(x)
        
        out = self.postprocess_0(out)
        out = self.resizer1(out)
        
        return out, out[:, :18, ...].sum(1)