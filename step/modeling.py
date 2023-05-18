import torch.nn as nn
import torch
import torch.nn.functional as F

from torchvision.transforms import Resize


class combine_translator(nn.Module):
    def __init__(self):
        super(combine_translator, self).__init__()
        self.amp_head = nn.Sequential(
            nn.Linear(1080, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 4608),
        )

        self.pha_head = nn.Sequential(
            nn.Linear(1080, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 4608),
        )

    @staticmethod
    def upconvblock(in_c, out_c, k, s):
        assert k % 2 == 1
        layers = [
            nn.ConvTranspose2d(in_c, out_c, k, s, k // 2, k // 2),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
        ]
        return nn.Sequential(*layers)
    
    def forward(self, x):
        amp_ = self.amp_head(x[0].reshape(x[0].shape[0], -1))
        pha_ = self.pha_head(x[1].reshape(x[1].shape[0], -1))
        x = torch.cat([amp_, pha_], -1).view([x[0].shape[0], 96, 96]).unsqueeze(1)
        # x = self.head(x)
        # x = self.resizer(x)
        # x = torch.sigmoid(x)
        # x = x - x.mean()
        # x = x / x.std()
        return x
    
    
def double_conv(in_channels, out_channels):  
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),  
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
    

class _student_0(nn.Module):
    def __init__(self, nchannel=19):
        super().__init__()
        
        c = [32, 64, 128, 256, 512, 1024]

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
        
        self.resizer = Resize((184, 184))
        
        self.postprocess_0 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, 2, 1),
            nn.BatchNorm2d(1),
            # nn.Sigmoid(),
        )

    def forward(self, x):

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
        
        out = self.resizer(out)
        sm = self.postprocess_0(out)
        
        sm = sm - sm.mean()
        sm = sm / sm.std()

        return sm#, jhm


def conv_bn_relu(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class __student_0(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(self).__init__()
        self.conv1 = conv_bn_relu(in_channels, 64)
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = conv_bn_relu(64, 128)
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        self.conv3 = conv_bn_relu(128, 256)
        self.pool3 = nn.AvgPool2d(kernel_size=2)
        self.conv4 = conv_bn_relu(256, 512)
        self.pool4 = nn.AvgPool2d(kernel_size=2)
        self.conv5 = conv_bn_relu(512, 1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.conv6 = conv_bn_relu(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.conv7 = conv_bn_relu(512, 256)
        self.up8 =nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv8 =conv_bn_relu(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv9 = conv_bn_relu(128, 64)
        self.conv10 = nn.Conv2d(64, 1, 1)
        self.resizer = Resize((46, 46))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        conv5 = self.conv5(pool4)
        up6 = self.up6(conv5)
        up6 = torch.cat([up6, conv4], dim=1)
        conv6 = self.conv6(up6)
        up7 = self.up7(conv6)
        up7 = torch.cat([up7, conv3], dim=1)
        conv7 = self.conv7(up7)
        up8 = self.up8(conv7)
        up8 = torch.cat([up8, conv2], dim=1)
        conv8 = self.conv8(up8)
        up9 = self.up9(conv8)
        up9 = torch.cat([up9, conv1], dim=1)
        conv9 = self.conv9(up9)
        out = self.conv10(conv9)
        # out = self.sigmoid(self.resizer(out))
        out = self.resizer(out)
        return out
        
        
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
class student_0(nn.Module):

    def __init__(self):
        super().__init__()

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
        self.pred = torch.nn.Conv2d(64, 1, 3, 1, 1)
        
        self.resizer = nn.MaxPool2d((2,2))

    def forward(self, x):
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
        
        x = self.resizer(self.pred(O4))
        
        # x = x - x.mean()
        # x = x / (x.std() + 1e-12)
        return x
    

class ____student_0(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 256, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 64, 3, 2, 1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 1, 3, 2, 1)
        self.bn5 = nn.BatchNorm2d(1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        return x