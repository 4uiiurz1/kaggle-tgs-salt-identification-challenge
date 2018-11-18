import torch.nn as nn

from torchvision.models import resnet
import torch.utils.model_zoo as model_zoo


class ResNet(resnet.ResNet):
    def __init__(self, block, layers, first_stride=1, num_classes=1000):
        super().__init__(block, layers, num_classes)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=first_stride, padding=3,
                               bias=False)


def resnet18(pretrained=False, first_stride=1, **kwargs):
    model = ResNet(resnet.BasicBlock, [2, 2, 2, 2], first_stride, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(resnet.model_urls['resnet18']))
    return model


def resnet34(pretrained=False, first_stride=1, **kwargs):
    model = ResNet(resnet.BasicBlock, [3, 4, 6, 3], first_stride, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(resnet.model_urls['resnet34']))
    return model


def resnet50(pretrained=False, first_stride=1, **kwargs):
    model = ResNet(resnet.Bottleneck, [3, 4, 6, 3], first_stride, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(resnet.model_urls['resnet50']))
    return model


def resnet101(pretrained=False, first_stride=1, **kwargs):
    model = ResNet(resnet.Bottleneck, [3, 4, 23, 3], first_stride, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(resnet.model_urls['resnet101']))
    return model


def resnet152(pretrained=False, first_stride=1, **kwargs):
    model = ResNet(resnet.Bottleneck, [3, 8, 36, 3], first_stride, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(resnet.model_urls['resnet152']))
    return model
