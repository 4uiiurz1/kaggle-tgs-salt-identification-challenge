# -*- coding: utf-8 -*-
import numpy as np

from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision
import pretrainedmodels

import resnet, senet, densenet


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, act_func=nn.ReLU(inplace=True)):
        super(VGGBlock, self).__init__()
        self.act_func = act_func
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_func(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act_func(out)

        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, act_func=nn.ReLU(inplace=True)):
        super(ResBlock, self).__init__()
        self.act_func = act_func
        self.conv0 = nn.Conv2d(in_channels, out_channels, 1)
        self.bn0 = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = self.conv0(x)
        residual = self.bn0(residual)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_func(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.act_func(out)

        return out


class ELUp1(nn.Module):
    def __init__(self, alpha=1., inplace=False):
        super(ELUp1, self).__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, input):
        return F.elu(input, self.alpha, self.inplace) + 1

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return 'alpha={}{}'.format(self.alpha, inplace_str)


class SCSEBlock(nn.Module):
    def __init__(self, in_channels, act_func=nn.ReLU(inplace=True)):
        super(SCSEBlock, self).__init__()
        self.in_channels = in_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cse = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels//2, 1),
            act_func,
            nn.Conv2d(self.in_channels//2, self.in_channels, 1),
            nn.Sigmoid())
        self.sse = nn.Sequential(
            nn.Conv2d(self.in_channels, 1, 1),
            nn.Sigmoid())


    def forward(self, x):
        cse = self.avg_pool(x)
        cse = self.cse(x)

        sse = self.sse(x)

        return torch.mul(x, cse) + torch.mul(x, sse)


class SABlock(nn.Module):
    def __init__(self, in_channels, key_channels, value_channels, out_channels, act_func=nn.ReLU(inplace=True)):
        super(SABlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels),
            act_func,
        )
        self.f_query = self.f_key
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
            kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
            kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)

        value = self.f_value(x).view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)
        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_key(x).view(batch_size, self.key_channels, -1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        context = self.W(context)

        return context


class SCSECatResUNet34(nn.Module):
    def __init__(self, args):
        """
        :param pretrained:
            False - no pre-trained network used
            True - encoder pre-trained with ResNet34
        """
        super().__init__()

        self.args = args
        if self.args.act_func == 'ReLU':
            self.act_func = nn.ReLU(inplace=True)
        elif self.args.act_func == 'ELUp1':
            self.act_func = ELUp1(inplace=True)

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4x = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8x = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up16x = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

        self.encoder = resnet.resnet34(pretrained=args.pretrained, first_stride=args.first_stride)

        self.conv1 = nn.Sequential(
            self.encoder.conv1,
            self.encoder.bn1,
            self.encoder.relu)
        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.center = VGGBlock(512, 512, 256, self.act_func)

        self.dec5 = VGGBlock(256+512, 512, 64, self.act_func)
        self.dec4 = VGGBlock(64+256, 256, 64, self.act_func)
        self.dec3 = VGGBlock(64+128, 128, 64, self.act_func)
        self.dec2 = VGGBlock(64+64, 64, 64, self.act_func)
        self.dec1 = VGGBlock(64+64, 64, 64, self.act_func)

        self.scse5 = SCSEBlock(64, self.act_func)
        self.scse4 = SCSEBlock(64, self.act_func)
        self.scse3 = SCSEBlock(64, self.act_func)
        self.scse2 = SCSEBlock(64, self.act_func)
        self.scse1 = SCSEBlock(64, self.act_func)

        self.concat = VGGBlock(320, 64, 64, self.act_func)

        self.final = nn.Conv2d(64, 1, kernel_size=1)


    def freeze_bn(self):
        for m in self.encoder.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False


    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.up(self.center(self.pool(conv5)))

        dec5 = self.scse5(self.dec5(torch.cat([center, conv5], 1)))
        dec4 = self.scse4(self.dec4(torch.cat([self.up(dec5), conv4], 1)))
        dec3 = self.scse3(self.dec3(torch.cat([self.up(dec4), conv3], 1)))
        dec2 = self.scse2(self.dec2(torch.cat([self.up(dec3), conv2], 1)))
        dec1 = self.scse1(self.dec1(torch.cat([self.up(dec2), conv1], 1)))

        concat = self.concat(torch.cat([
            dec1,
            self.up(dec2),
            self.up4x(dec3),
            self.up8x(dec4),
            self.up16x(dec5)
        ], 1))

        x_out = self.final(concat)

        return x_out


class SCSECatSEResNeXt32x4dUNet50(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        if self.args.act_func == 'ReLU':
            self.act_func = nn.ReLU(inplace=True)
        elif self.args.act_func == 'ELUp1':
            self.act_func = ELUp1(inplace=True)
        if self.args.pretrained:
            pretrained = 'imagenet'
        else:
            pretrained = None

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4x = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8x = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up16x = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

        self.encoder = senet.se_resnext50_32x4d(pretrained=pretrained, first_stride=args.first_stride)

        self.conv1 = nn.Sequential(
            self.encoder.layer0.conv1,
            self.encoder.layer0.bn1,
            self.encoder.layer0.relu1)
        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.center = VGGBlock(2048, 1024, 512, self.act_func)

        self.dec5 = VGGBlock(512+2048, 1024, 64, self.act_func)
        self.dec4 = VGGBlock(64+1024, 512, 64, self.act_func)
        self.dec3 = VGGBlock(64+512, 256, 64, self.act_func)
        self.dec2 = VGGBlock(64+256, 128, 64, self.act_func)
        self.dec1 = VGGBlock(64+64, 64, 64, self.act_func)

        self.scse5 = SCSEBlock(64, self.act_func)
        self.scse4 = SCSEBlock(64, self.act_func)
        self.scse3 = SCSEBlock(64, self.act_func)
        self.scse2 = SCSEBlock(64, self.act_func)
        self.scse1 = SCSEBlock(64, self.act_func)

        self.concat = VGGBlock(320, 64, 64, self.act_func)

        self.final = nn.Conv2d(64, 1, kernel_size=1)


    def freeze_bn(self):
        for m in self.encoder.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False


    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.up(self.center(self.pool(conv5)))

        dec5 = self.scse5(self.dec5(torch.cat([center, conv5], 1)))
        dec4 = self.scse4(self.dec4(torch.cat([self.up(dec5), conv4], 1)))
        dec3 = self.scse3(self.dec3(torch.cat([self.up(dec4), conv3], 1)))
        dec2 = self.scse2(self.dec2(torch.cat([self.up(dec3), conv2], 1)))
        dec1 = self.scse1(self.dec1(torch.cat([self.up(dec2), conv1], 1)))

        concat = self.concat(torch.cat([
            dec1,
            self.up(dec2),
            self.up4x(dec3),
            self.up8x(dec4),
            self.up16x(dec5)
        ], 1))

        x_out = self.final(concat)

        return x_out


class SASCSECatResUNet34(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        if self.args.act_func == 'ReLU':
            self.act_func = nn.ReLU(inplace=True)
        elif self.args.act_func == 'ELUp1':
            self.act_func = ELUp1(inplace=True)

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4x = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8x = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up16x = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

        self.encoder = resnet.resnet34(pretrained=args.pretrained, first_stride=args.first_stride)

        self.conv1 = nn.Sequential(
            self.encoder.conv1,
            self.encoder.bn1,
            self.encoder.relu)
        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.center = VGGBlock(512, 512, 256, self.act_func)

        self.sa = SABlock(256, 128, 128, 256, self.act_func)

        self.dec5 = VGGBlock(256+512, 512, 64, self.act_func)
        self.dec4 = VGGBlock(64+256, 256, 64, self.act_func)
        self.dec3 = VGGBlock(64+128, 128, 64, self.act_func)
        self.dec2 = VGGBlock(64+64, 64, 64, self.act_func)
        self.dec1 = VGGBlock(64+64, 64, 64, self.act_func)

        self.scse5 = SCSEBlock(64, self.act_func)
        self.scse4 = SCSEBlock(64, self.act_func)
        self.scse3 = SCSEBlock(64, self.act_func)
        self.scse2 = SCSEBlock(64, self.act_func)
        self.scse1 = SCSEBlock(64, self.act_func)

        self.concat = VGGBlock(320, 64, 64, self.act_func)

        self.final = nn.Conv2d(64, 1, kernel_size=1)


    def freeze_bn(self):
        for m in self.encoder.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False


    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.up(self.sa(self.center(self.pool(conv5))))

        dec5 = self.scse5(self.dec5(torch.cat([center, conv5], 1)))
        dec4 = self.scse4(self.dec4(torch.cat([self.up(dec5), conv4], 1)))
        dec3 = self.scse3(self.dec3(torch.cat([self.up(dec4), conv3], 1)))
        dec2 = self.scse2(self.dec2(torch.cat([self.up(dec3), conv2], 1)))
        dec1 = self.scse1(self.dec1(torch.cat([self.up(dec2), conv1], 1)))

        concat = self.concat(torch.cat([
            dec1,
            self.up(dec2),
            self.up4x(dec3),
            self.up8x(dec4),
            self.up16x(dec5)
        ], 1))

        x_out = self.final(concat)

        return x_out


class SCSECatSEResNeXt32x4dUNet101(nn.Module):
    def __init__(self, pretrained=False):
        """
        :param pretrained:
            False - no pre-trained network used
            True - encoder pre-trained with SE-ResNeXt101_32x4d
        """
        super().__init__()

        self.args = args
        if self.args.act_func == 'ReLU':
            self.act_func = nn.ReLU(inplace=True)
        elif self.args.act_func == 'ELUp1':
            self.act_func = ELUp1(inplace=True)
        if self.args.pretrained:
            pretrained = 'imagenet'
        else:
            pretrained = None

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4x = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8x = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up16x = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

        self.encoder = senet.se_resnext101_32x4d(pretrained=pretrained, first_stride=args.first_stride)

        self.conv1 = nn.Sequential(
            self.encoder.layer0.conv1,
            self.encoder.layer0.bn1,
            self.encoder.layer0.relu1)
        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.center = VGGBlock(2048, 1024, 512, self.act_func)

        self.dec5 = VGGBlock(512+2048, 1024, 64, self.act_func)
        self.dec4 = VGGBlock(64+1024, 512, 64, self.act_func)
        self.dec3 = VGGBlock(64+512, 256, 64, self.act_func)
        self.dec2 = VGGBlock(64+256, 128, 64, self.act_func)
        self.dec1 = VGGBlock(64+64, 64, 64, self.act_func)

        self.scse5 = SCSEBlock(64, self.act_func)
        self.scse4 = SCSEBlock(64, self.act_func)
        self.scse3 = SCSEBlock(64, self.act_func)
        self.scse2 = SCSEBlock(64, self.act_func)
        self.scse1 = SCSEBlock(64, self.act_func)

        self.concat = VGGBlock(320, 64, 64, self.act_func)

        self.final = nn.Conv2d(64, 1, kernel_size=1)


    def freeze_bn(self):
        for m in self.encoder.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False


    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.up(self.center(self.pool(conv5)))

        dec5 = self.scse5(self.dec5(torch.cat([center, conv5], 1)))
        dec4 = self.scse4(self.dec4(torch.cat([self.up(dec5), conv4], 1)))
        dec3 = self.scse3(self.dec3(torch.cat([self.up(dec4), conv3], 1)))
        dec2 = self.scse2(self.dec2(torch.cat([self.up(dec3), conv2], 1)))
        dec1 = self.scse1(self.dec1(torch.cat([self.up(dec2), conv1], 1)))

        concat = self.concat(torch.cat([
            dec1,
            self.up(dec2),
            self.up4x(dec3),
            self.up8x(dec4),
            self.up16x(dec5)
        ], 1))

        x_out = self.final(concat)

        return x_out
