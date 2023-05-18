import torch.nn as nn
import torch

from torchvision.transforms import Resize


class combine_translator(nn.Module):
    def __init__(self):
        super(combine_translator, self).__init__()
        self.amp_head = nn.Sequential(
            nn.Linear(1080, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 800),
        )

        self.pha_head = nn.Sequential(
            nn.Linear(1080, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 800),
        )

        head = [
            self.upconvblock(1, 32, 3, 2),
            # self.upconvblock(3, 3, 3, 2),
            # self.upconvblock(32, 32, 5, 3),
            self.upconvblock(32, 32, 3, 2),
            self.upconvblock(32, 32, 3, 2),
            nn.Conv2d(32, 32, 1, 1, 0),
            nn.BatchNorm2d(32),
            # nn.Sigmoid(),
            # nn.AdaptiveAvgPool2d((184 * 2, 184 * 2)),
        ]
        self.head = nn.Sequential(*head)
        self.resizer = Resize((96, 96))

    @staticmethod
    def upconvblock(in_c, out_c, k, s):
        assert k % 2 == 1
        layers = [
            nn.ConvTranspose2d(in_c, out_c, k, s, k // 2, k // 2),
            nn.BatchNorm2d(out_c),
            # nn.Softplus(),
            nn.ReLU(),
            # nn.ConvTranspose2d(out_c, out_c, k, s, k // 2),
            # nn.BatchNorm2d(out_c),
            # # nn.Softplus(),
            # nn.ReLU(),
        ]
        return nn.Sequential(*layers)
    
    def forward(self, x):
        amp_ = self.amp_head(x[0].reshape(x[0].shape[0], -1))
        pha_ = self.pha_head(x[1].reshape(x[1].shape[0], -1))
        x = torch.cat([amp_, pha_], -1).view([x[0].shape[0], 40, 40]).unsqueeze(1)
        x = self.head(x)
        x = self.resizer(x)
        x = torch.sigmoid(x)

        # x = torch.cat([(x[:, [_], ...] - self.mean[_]) / self.std[_] for _ in range(3)], dim=1)
        
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

    
class student_0(nn.Module):
    def __init__(self, nchannel=19):
        super().__init__()
        
        c = [32, 64, 128, 256, 512, 1024]

        self.dconv_down0 = double_conv(c[0], c[1])
        self.dconv_down1 = double_conv(c[1], c[2])
        self.dconv_down2 = double_conv(c[2], c[3])
        self.dconv_down3 = double_conv(c[3], c[4])
        self.dconv_down4 = double_conv(c[4], c[5])

        self.maxpool = nn.MaxPool2d(2)
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

        # self.conv_last = nn.Conv2d(nchannel, 1, 1)
        
        self.postprocess_0 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, 2, 1),
            nn.BatchNorm2d(1),
            # nn.ReLU(),
            # nn.Conv2d(32, 32, 1, 1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            # nn.Conv2d(32, 1, 1, 1),
            # nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        
        self.postprocess_1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.Conv2d(32, 32, 3, 2, 1),
            # nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 32, 1, 1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
            nn.Conv2d(32, nchannel, 3, 2, 1),
            nn.BatchNorm2d(nchannel),
            # nn.Sigmoid(),
        )

    def forward(self, x):

        # x = self.preprocess(x)
        # encode
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
        jhm = self.postprocess_1(out)
        
        sm = nn.functional.relu(sm - 0.6)

        return sm, jhm
    
    
    
class student_1(nn.Module):
    def __init__(self, nchannel=19):
        super().__init__()
        
        c = [32, 64, 128, 256, 512, 1024]

        self.dconv_down0 = double_conv(c[0], c[1])
        self.dconv_down1 = double_conv(c[1], c[2])
        self.dconv_down2 = double_conv(c[2], c[3])
        self.dconv_down3 = double_conv(c[3], c[4])
        self.dconv_down4 = double_conv(c[4], c[5])

        self.maxpool = nn.MaxPool2d(2)
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

        # self.conv_last = nn.Conv2d(nchannel, 1, 1)
        
        self.postprocess_1 = nn.Sequential(
            # nn.AdaptiveAvgPool2d((184, 184)),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.Conv2d(32, 32, 3, 2, 1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            # nn.Conv2d(32, 32, 1, 1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            # nn.Conv2d(32, nchannel, 1, 1),
            nn.Conv2d(32, nchannel, 3, 2, 1),
            nn.BatchNorm2d(nchannel),
            # nn.AdaptiveMaxPool2d((46, 46)),
            # nn.Sigmoid(),
        )

    def forward(self, x):

        # x = self.preprocess(x)
        # encode
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
        return self.postprocess_1(out)
