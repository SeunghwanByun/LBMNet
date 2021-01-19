import cv2
import math
import numpy as np

import torch.nn.functional as F

from resnet_ import *

from coordconv import CoordConv
from deformconv import *

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

# LBM channel wise sum ver
class LBM_SUM(nn.Module):
    def __init__(self, input_channels, n_resblocks=2, neighbor_size=4, conv=default_conv):
        super(LBM_SUM, self).__init__()
        m_body = [
            ResBlock(conv, input_channels, kernel_size=3, act=nn.ReLU(True), res_scale=0.1)
            for _ in range(n_resblocks)
        ]

        self.body = nn.Sequential(*m_body)

        self.output = nn.Sequential(nn.Conv2d(input_channels, neighbor_size ** 2, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(neighbor_size ** 2),
                                    nn.ReLU(inplace=True))

        self.lrn = nn.PixelShuffle(neighbor_size)
        self.one = nn.Conv2d(1, 1, kernel_size=3, stride=neighbor_size, padding=0, bias=False)

        KernelOne = torch.ones(3, 3)
        KernelOneExpand = KernelOne.expand(self.one.weight.size())
        self.one.weight = torch.nn.Parameter(KernelOneExpand, requires_grad=False)
        self.act_fin = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.body(x)
        x = self.output(x)

        x = self.lrn(x)
        x = self.one(x)

        x = self.act_fin(x)

        return x

# LBM 1x1 convolution ver
class LBM_1x1Conv(nn.Module):
    def __init__(self, input_channels, n_resblocks=2, neighbor_size=4, conv=default_conv):
        super(LBM_1x1Conv, self).__init__()
        m_body = [
            ResBlock(conv, input_channels, kernel_size=3, act=nn.ReLU(True), res_scale=0.1)
            for _ in range(n_resblocks)
        ]

        self.body = nn.Sequential(*m_body)

        self.output = nn.Sequential(nn.Conv2d(input_channels, neighbor_size ** 2, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(neighbor_size ** 2),
                                    nn.ReLU(inplace=True))

        self.conv1x1 = nn.Conv2d(neighbor_size ** 2, 1, kernel_size=1, stride=1, padding=0)
        self.act_fin = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.body(x)
        x = self.output(x)
        
        x = self.conv1x1(x)
        x = self.act_fin(x)

        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Inspired by Binary Information Localization Module
class BILM(nn.Module):
    def __init__(self):
        super(BILM, self).__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, feat, side_feat_map):
        pos_sig = torch.sigmoid(side_feat_map)
        neg_sig = -1 * pos_sig

        pos_sig = self.maxpool1(pos_sig)
        neg_sig = self.maxpool2(neg_sig)
        sum_sig = pos_sig + neg_sig

        multi_with_sig = sum_sig * feat
  
        x = multi_with_sig + side_feat_map

        return x

class eASPP(nn.Module):
    def __init__(self, in_channels=1024, out_channels=256, rate=1, bn_mom=0.1):
        super(eASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=3 * rate, dilation=3 * rate, bias=False),
            nn.BatchNorm2d(out_channels, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6 * rate, dilation=6 * rate, bias=False),
            nn.BatchNorm2d(out_channels, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12 * rate, dilation=12 * rate, bias=False),
            nn.BatchNorm2d(out_channels, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )
        self.branch5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )

        self.output = nn.Conv2d(out_channels * 5, in_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x_size = (x.shape[2], x.shape[3])
        br1 = self.branch1(x)
        br2 = self.branch2(x)
        br3 = self.branch3(x)
        br4 = self.branch4(x)
        br5 = self.branch5(x)
        br5 = F.interpolate(br5, size=x_size, mode='bilinear', align_corners=True)

        output = torch.cat([br1, br2, br3, br4, br5], dim=1)
        output = self.output(output)

        return output

# First LBMNet
class LBMNet50(nn.Module):
    def __init__(self, layers=101, input_channels=6, classes=1, conv=default_conv, criterion=nn.BCEWithLogitsLoss()):#criterion=nn.BCELoss()):
        super(LBMNet50, self).__init__()

        self.criterion = criterion
        self.classes = classes

        if layers == 50:
            model = resnest50_2s2x40d()
        elif layers == 101:
            model = resnest101_2s2x64d()

        self.header = nn.Sequential(CoordConv(input_channels, 32, with_r=True),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(inplace=True))
        self.layer0 = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool

        self.lbam = LBM_SUM(input_channels=2048)
        self.lbam0 = LBM_SUM(input_channels=1024)
        self.lbam1 = LBM_SUM(input_channels=512)
        self.lbam2 = LBM_SUM(input_channels=256)
        self.lbam3 = LBM_SUM(input_channels=64)
        self.lbam4 = LBM_SUM(input_channels=32)

        self.conv_32 = nn.Sequential(nn.ConvTranspose2d(4096, 1024, kernel_size=4, stride=2, padding=1),
            # nn.Conv2d(4096, 1024, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(1024),
                                     nn.ReLU(inplace=True))
        self.conv_16 = nn.Sequential(nn.ConvTranspose2d(3072, 512, kernel_size=4, stride=2, padding=1),
            # nn.Conv2d(3072, 512, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(512),
                                     nn.ReLU(inplace=True))
        self.conv_8 = nn.Sequential(nn.ConvTranspose2d(1536, 256, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True))
        self.conv_4 = nn.Sequential(nn.Conv2d(768, 64, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.conv_2 = nn.Sequential(Upsampler(conv, 2, 192, act=False),
                                    nn.ConvTranspose2d(192, 32, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(inplace=True))

        self.output = nn.Sequential(nn.Conv2d(96, 48, kernel_size=1, stride=1, padding=0),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout2d(p=0.2),
                                    nn.Conv2d(48, classes, kernel_size=1, stride=1, padding=0))

    def forward(self, x, y=None):
        x1 = self.header(x)  # x 1, 32
        x2 = self.layer0(x1)  # x 1/4, 64
        x4 = self.layer1(x2)  # x 1/4, 256
        x8 = self.layer2(x4)  # x 1/8, 512
        x16 = self.layer3(x8) # x 1/16, 1024
        x32 = self.layer4(x16) # x 1/32, 2048

        input = x.detach().cpu().numpy()[0].transpose(1, 2, 0).reshape(64, 512, 6) * 255.0
        img = input[:,:,:3]
        z = input[:,:,3]
        intensity = input[:, :, 4]
        depth = input[:, :, 5]

        lbam32 = self.lbam(x32)
        
        lbam16 = self.lbam0(x16)
        
        lbam8 = self.lbam1(x8)
        
        lbam4 = self.lbam2(x4)
        
        lbam2 = self.lbam3(x2)
        
        lbam1 = self.lbam4(x1)
        
        x32_lbam32 = x32 * lbam32
        x32 = torch.cat([x32, x32_lbam32], dim=1)
        x32 = self.conv_32(x32)

        x16_lrn16 = x16 * lbam16
        x16 = torch.cat([x16, x16_lrn16, x32], dim=1)
        x16 = self.conv_16(x16)

        x8_lbam8 = x8 * lbam8
        x8 = torch.cat([x8, x8_lbam8, x16], dim=1)
        x8 = self.conv_8(x8)

        x4_lbam4 = x4 * lbam4
        x4 = torch.cat([x4, x4_lbam4, x8], dim=1)
        x4 = self.conv_4(x4)

        x2_lbam2 = x2 * lbam2
        x2 = torch.cat([x2, x2_lbam2, x4], dim=1)
        x2 = self.conv_2(x2)

        x1_lbam1 = x1 * lbam1
        x1 = torch.cat([x1, x1_lbam1, x2], dim=1)
        x_output = self.output(x1)

        x_output = torch.sigmoid(x_output)

        if self.training:
            loss = self.criterion(x_output, y)

            return x_output, loss
        else:
            return x_output

# Improvement LBMNet
# Feed-forward LBM feature version
class LBMNet50_Improv(nn.Module):
    def __init__(self, layers=50, input_channels=6, classes=1, conv=default_conv, criterion=nn.BCEWithLogitsLoss()):
        super(LBMNet50_Improv, self).__init__()

        self.criterion = criterion
        self.classes = classes

        if layers == 50:
            model = resnet50()
        elif layers == 101:
            model = resnet101()

        self.header = nn.Sequential(CoordConv(input_channels, 32, with_r=True),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(inplace=True))
        self.layer0 = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool

        self.lbm = LBM_SUM(input_channels=2048)
        self.lbm0 = LBM_SUM(input_channels=1024)
        self.lbm1 = LBM_SUM(input_channels=512)
        self.lbm2 = LBM_SUM(input_channels=256)
        self.lbm3 = LBM_SUM(input_channels=64)
        self.lbm4 = LBM_SUM(input_channels=32)

        self.conv_32 = nn.Sequential(nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1),
            # nn.Conv2d(4096, 1024, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(1024),
                                     nn.ReLU(inplace=True))
        self.conv_16 = nn.Sequential(nn.ConvTranspose2d(2048, 512, kernel_size=4, stride=2, padding=1),
            # nn.Conv2d(3072, 512, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(512),
                                     nn.ReLU(inplace=True))
        self.conv_8 = nn.Sequential(nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True))
        self.conv_4 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.conv_2 = nn.Sequential(Upsampler(conv, 2, 128, act=False),
                                    nn.ConvTranspose2d(128, 32
                                                       , kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(inplace=True))

        self.output = nn.Sequential(nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout2d(p=0.2),
                                    nn.Conv2d(32, classes, kernel_size=1, stride=1, padding=0))

    def forward(self, x, y=None):
        x1 = self.header(x)  # x 1, 32
        lbm1 = self.lbm4(x1)
        x1_lbm1 = x1 * lbm1

        x2 = self.layer0(x1_lbm1)  # x 1/4, 64
        lbm2 = self.lbm3(x2)
        x2_lbm2 = x2 * lbm2

        x4 = self.layer1(x2_lbm2)  # x 1/4, 256
        lbm4 = self.lbm2(x4)
        x4_lbm4 = x4 * lbm4

        x8 = self.layer2(x4_lbm4)  # x 1/8, 512
        lbm8 = self.lbm1(x8)
        x8_lbm8 = x8 * lbm8

        x16 = self.layer3(x8_lbm8) # x 1/16, 1024
        lbm16 = self.lbm0(x16)
        x16_lbm16 = x16 * lbm16

        x32 = self.layer4(x16_lbm16) # x 1/32, 2048
        lbm32 = self.lbm(x32)
        x32_lbm32 = x32 * lbm32

        x32 = self.conv_32(x32_lbm32)

        x16 = torch.cat([x16_lbm16, x32], dim=1)
        x16 = self.conv_16(x16)

        x8 = torch.cat([x8_lbm8, x16], dim=1)
        x8 = self.conv_8(x8)

        x4 = torch.cat([x4_lbm4, x8], dim=1)
        x4 = self.conv_4(x4)

        x2 = torch.cat([x2_lbm2, x4], dim=1)
        x2 = self.conv_2(x2)

        x1 = torch.cat([x1_lbm1, x2], dim=1)
        x_output = self.output(x1)

        x_output = torch.sigmoid(x_output)

        if self.training:
            loss = self.criterion(x_output, y)

            return x_output, loss
        else:
            return x_output
