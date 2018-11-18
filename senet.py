from collections import OrderedDict

import torch.nn as nn
from pretrainedmodels.models import senet


class SENet(senet.SENet):
    def __init__(self, block, layers, groups, reduction, first_stride=1, dropout_p=0.2,
                 inplanes=128, input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1, num_classes=1000):
        super().__init__(block, layers, groups, reduction, dropout_p,
                     inplanes, input_3x3, downsample_kernel_size,
                     downsample_padding, num_classes)
        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, 64, 3, stride=first_stride, padding=1,
                                    bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=first_stride,
                                    padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
                                            ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))


def senet154(num_classes=1000, pretrained='imagenet', first_stride=1):
    model = SENet(senet.SEBottleneck, [3, 8, 36, 3], groups=64, reduction=16, first_stride=first_stride,
                  dropout_p=0.2, num_classes=num_classes)
    if pretrained is not None:
        settings = senet.pretrained_settings['senet154'][pretrained]
        senet.initialize_pretrained_model(model, num_classes, settings)
    return model


def se_resnet50(num_classes=1000, pretrained='imagenet', first_stride=1):
    model = SENet(senet.SEBottleneck, [3, 4, 6, 3], groups=1, reduction=16, first_stride=first_stride,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = senet.pretrained_settings['se_resnet50'][pretrained]
        senet.initialize_pretrained_model(model, num_classes, settings)
    return model


def se_resnet101(num_classes=1000, pretrained='imagenet', first_stride=1):
    model = SENet(senet.SEBottleneck, [3, 4, 23, 3], groups=1, reduction=16, first_stride=first_stride,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = senet.pretrained_settings['se_resnet101'][pretrained]
        senet.initialize_pretrained_model(model, num_classes, settings)
    return model


def se_resnet152(num_classes=1000, pretrained='imagenet', first_stride=1):
    model = SENet(senet.SEBottleneck, [3, 8, 36, 3], groups=1, reduction=16, first_stride=first_stride,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = senet.pretrained_settings['se_resnet152'][pretrained]
        senet.initialize_pretrained_model(model, num_classes, settings)
    return model


def se_resnext50_32x4d(num_classes=1000, pretrained='imagenet', first_stride=1):
    model = SENet(senet.SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16, first_stride=first_stride,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = senet.pretrained_settings['se_resnext50_32x4d'][pretrained]
        senet.initialize_pretrained_model(model, num_classes, settings)
    return model


def se_resnext101_32x4d(num_classes=1000, pretrained='imagenet', first_stride=1):
    model = SENet(senet.SEResNeXtBottleneck, [3, 4, 23, 3], groups=32, reduction=16, first_stride=first_stride,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = senet.pretrained_settings['se_resnext101_32x4d'][pretrained]
        senet.initialize_pretrained_model(model, num_classes, settings)
    return model
