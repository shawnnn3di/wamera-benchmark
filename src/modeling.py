import torch
import torch.nn as nn

from torchvision.transforms import Resize


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
        
        self.preprocess = nn.Sequential(
            # nn.AdaptiveAvgPool3d((120, 3, 3)),
            nn.ConvTranspose2d(120, 120, 3, 2, 1, 1),
            nn.BatchNorm2d(120),
            nn.ReLU(),
            nn.ConvTranspose2d(120, 120, 3, 2, 1, 1),
            nn.BatchNorm2d(120),
            nn.ReLU(),
            nn.ConvTranspose2d(120, 128, 3, 2, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.AdaptiveAvgPool3d((120, 3, 3)),
            # nn.Upsample((96, 96)),
            # nn.Conv2d(120, 32, 1, 1)
            # nn.ReLU(),
        )
        
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

        self.conv_last = nn.Conv2d(c[0], nchannel, 1)
        
        self.resizer = Resize((46, 46))

        # self.conv_last = nn.Conv2d(nchannel, 1, 1)

    def forward(self, x):

        x = self.preprocess(x)
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
        
        out = self.resizer(out) / torch.tensor([[_.max() for _ in  j] for j in out]).unsqueeze(2).unsqueeze(2).to(out.device)
        
        outmean = out[:, :18, ...].sum(1)
        
        # sm = outmean - outmean.mean()
        # sm = sm / (sm.std() + 1e-8)
        
        # jhm = out - out.mean()
        # jhm = jhm / (jhm.std() + 1e-8)
        
        return outmean / torch.tensor([torch.max(_) for _ in outmean]).unsqueeze(1).unsqueeze(2).to(outmean.device), out
    

class student_2(nn.Module):
    def __init__(self, nchannel=19):
        super().__init__()
        
        self.preprocess = nn.Sequential(
            nn.AdaptiveAvgPool3d((120, 3, 3)),
            nn.ConvTranspose2d(120, 120, 3, 2, 1, 1),
            nn.BatchNorm2d(120),
            nn.ReLU(),
            nn.ConvTranspose2d(120, 120, 3, 2, 1, 1),
            nn.BatchNorm2d(120),
            nn.ReLU(),
            nn.ConvTranspose2d(120, 128, 3, 2, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.AdaptiveAvgPool3d((120, 3, 3)),
            # nn.Upsample((96, 96)),
            # nn.Conv2d(120, 32, 1, 1)
            # nn.ReLU(),
        )
        
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


    def forward(self, x):

        x = self.preprocess(x)
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
        # sm = self.postprocess_0(out)
        
        # out = out - out.mean()
        # out = out / (out.std() + 1e-8)

        return out


class student_1(nn.Module):
    def __init__(self, nchannel=19):
        super().__init__()
        
        self.preprocess = nn.Sequential(
            nn.AdaptiveAvgPool3d((120, 3, 3)),
            nn.ConvTranspose2d(120, 120, 3, 2, 1, 1),
            nn.BatchNorm2d(120),
            nn.ReLU(),
            nn.ConvTranspose2d(120, 120, 3, 2, 1, 1),
            nn.BatchNorm2d(120),
            nn.ReLU(),
            nn.ConvTranspose2d(120, 128, 3, 2, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.AdaptiveAvgPool3d((120, 3, 3)),
            # nn.Upsample((96, 96)),
            # nn.Conv2d(120, 32, 1, 1),
        )
        
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

        x = self.preprocess(x)
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
