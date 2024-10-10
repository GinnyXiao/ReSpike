import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from torchvision.models.resnet import resnet18, resnet50


class ResNetBackbone(nn.Module):
    def __init__(self, backbone='resnet18'):
        super().__init__()
        if backbone == 'resnet18':
            resnet = resnet18(weights='DEFAULT')
            modules = list(resnet.children())[:-2]      # delete the last pooling and fc layer.
            self.backbone = nn.Sequential(*modules)
            self.fc_inplanes = resnet.fc.in_features
        elif backbone == 'resnet50':
            resnet = resnet50(weights='DEFAULT')
            modules = list(resnet.children())[:-2]      # delete the last pooling and fc layer.
            self.backbone = nn.Sequential(*modules)
            self.fc_inplanes = resnet.fc.in_features
        else:   
            raise NotImplementedError

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B x T x C x H x W) 
            returns :
                y.shape = (B * T, self.fc_inplanes, H, W)  
                y.shape = [B * T, 512, 7, 7] for input x.shape = [B, T, 3, 224, 224]
        """

        B, T, C, H, W = x.shape
        x = x.reshape(B*T, C, H, W)
        y = self.backbone(x)#.reshape(B, T, self.fc_inplanes, 1, 1)
        
        return y


# if __name__ == '__main__':
#     model = ResNetBackbone(backbone='resnet50x3')
#     out = model(torch.randn(4, 16, 3, 224, 224))
#     print(out.shape)