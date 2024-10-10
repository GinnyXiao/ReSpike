import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import functional, layer, surrogate, neuron
from spikingjelly.clock_driven.functional import reset_net

from .fusion.attention import SpatialTransformer
from .fusion.backbone_ann import ResNetBackbone as ANNResNetBackbone
from einops import rearrange, repeat

tau_global = 1./(1. - 0.25)

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.lif1 = neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau= tau_global,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.lif2 = neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=tau_global,
                                   surrogate_function=surrogate.ATan(),
                                   detach_reset=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d((2, 2)),
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.bn1(self.conv1(self.lif1(x)))
        out = self.bn2(self.conv2(self.lif2(out)))
        return out + self.shortcut(x)


class AttentionResNet(nn.Module):
    def __init__(self, 
                 block, 
                 num_blocks, 
                 width_mult=4, 
                 total_timestep=6, 
                 ann_backbone='resnet18',
                 transformer_depth=1,     # custom transformer support
                 num_heads=1,             # custom transformer support
                 dim_head=32,             # custom transformer support
                 dropout=0,               # custom transformer support
                 ):

        super(AttentionResNet, self).__init__()

        ####################### snn backbone ##############################
        self.in_planes = 16 * width_mult
        self.total_timestep = total_timestep

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, self.in_planes, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, self.in_planes * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, self.in_planes * 2, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, self.in_planes * 2, num_blocks[3], stride=2)

        ####################### crossattn layers ########################
        in_channels = np.array([1, 2, 4, 8]) * 16 * width_mult
        n_heads = in_channels // dim_head


        if ann_backbone == 'resnet18':
            context_dims = [64, 128, 256, 512]
        elif ann_backbone == 'resnet50':
            context_dims = [256, 512, 1024, 2048]
        else:
            raise NotImplementedError

        # cross modal fusion layers
        self.fusion1 = SpatialTransformer(in_channels=in_channels[0], n_heads=n_heads[0], d_head=dim_head,
										  depth=transformer_depth, dropout=dropout, context_dim=context_dims[0])
        self.fusion2 = SpatialTransformer(in_channels=in_channels[1], n_heads=n_heads[1], d_head=dim_head,
                                          depth=transformer_depth, dropout=dropout, context_dim=context_dims[1])              
        self.fusion3 = SpatialTransformer(in_channels=in_channels[2], n_heads=n_heads[2], d_head=dim_head,
                                          depth=transformer_depth, dropout=dropout, context_dim=context_dims[2])
        self.fusion4 = SpatialTransformer(in_channels=in_channels[3], n_heads=n_heads[3], d_head=dim_head,
                                          depth=transformer_depth, dropout=dropout, context_dim=context_dims[3])

        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, s1, s2, s3, s4):
        """
        input args:
            snn data x: [bs * num_key, frame num, channel, H, W]
            ann context s1, s2, s3, s4: [bs * num_key, (H*W), channel]
        output:
            attn snn out: [bs * num_key, frame num, channel, H, W]
        """

        # ANN feature rearrange to transformer input shapes
        s1 = rearrange(s1, 'b c h w -> b (h w) c')
        s2 = rearrange(s2, 'b c h w -> b (h w) c')
        s3 = rearrange(s3, 'b c h w -> b (h w) c')
        s4 = rearrange(s4, 'b c h w -> b (h w) c')
        
        reset_net(self)
        output_list = []
        for t in range(self.total_timestep):
            out = self.bn1(self.conv1(x[:, t, ...]))
            out = self.layer1(out)         # [bs, 64, 56, 56]
            # out = self.fusion1(out, s1)    # [bs, 64, 56, 56]
            out = self.layer2(out)         # [bs, 128, 28, 28]
            out = self.fusion2(out, s2)    # [bs, 128, 28, 28]
            out = self.layer3(out)         # [bs, 256, 14, 14]
            out = self.fusion3(out, s3)    # [bs, 256, 14, 14]
            out = self.layer4(out)         # [bs, 512, 7, 7]
            out = self.fusion4(out, s4)    # [bs, 512, 7, 7]

            output_list.append(out)

        return torch.stack(output_list, dim=1)


class AttentionResNet_Downsample(nn.Module):
    def __init__(self,
                 block,
                 num_blocks,
                 width_mult=4,
                 total_timestep=6,
                 ann_backbone='resnet18',
                 transformer_depth=1,  # custom transformer support
                 num_heads=1,  # custom transformer support
                 dim_head=32,  # custom transformer support
                 dropout=0,  # custom transformer support
                 ):

        super(AttentionResNet_Downsample, self).__init__()

        ####################### snn backbone ##############################
        self.in_planes = 16 * width_mult
        self.total_timestep = total_timestep

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, self.in_planes, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, self.in_planes * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, self.in_planes * 2, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, self.in_planes * 2, num_blocks[3], stride=2)

        ####################### crossattn layers ########################
        in_channels = np.array([1, 2, 4, 8]) * 16 * width_mult
        n_heads = in_channels // dim_head

        if ann_backbone == 'resnet18':
            context_dims = [64, 128, 256, 512]
        elif ann_backbone == 'resnet50':
            context_dims = [256, 512, 1024, 2048]
        else:
            raise NotImplementedError

        # cross modal fusion layers
        self.fusion1 = SpatialTransformer(in_channels=in_channels[0], n_heads=n_heads[0], d_head=dim_head,
                                          depth=transformer_depth, dropout=dropout, context_dim=context_dims[0])
        self.fusion2 = SpatialTransformer(in_channels=in_channels[1], n_heads=n_heads[1], d_head=dim_head,
                                          depth=transformer_depth, dropout=dropout, context_dim=context_dims[1])
        self.fusion3 = SpatialTransformer(in_channels=in_channels[2], n_heads=n_heads[2], d_head=dim_head,
                                          depth=transformer_depth, dropout=dropout, context_dim=context_dims[2])
        self.fusion4 = SpatialTransformer(in_channels=in_channels[3], n_heads=n_heads[3], d_head=dim_head,
                                          depth=transformer_depth, dropout=dropout, context_dim=context_dims[3])

        self.pool1 = nn.AvgPool2d((4, 4)) # [56, 56] --> [14, 14]
        self.pool2 = nn.AvgPool2d((2, 2)) # [28, 28] --> [14, 14]
        # self.pool3 = nn.AvgPool2d((2, 2)) # [14, 14] --> [7, 7]

        self.upsample1 = torch.nn.Upsample(scale_factor=4, mode='bilinear') # [b, c, 14, 14] --> [b, c, 56, 56]
        self.upsample2 = torch.nn.Upsample(scale_factor=2, mode='bilinear')  # [b, c, 14, 14] --> [b, c, 28, 28]
        # self.upsample3 = torch.nn.Upsample(scale_factor=2, mode='bilinear')  # [b, c, 14, 14] --> [b, c, 56, 56]

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, s1, s2, s3, s4):
        """
        input args:
            snn data x: [bs * num_key, frame num, channel, H, W]
            ann context s1, s2, s3, s4: [bs * num_key, (H*W), channel]
        output:
            attn snn out: [bs * num_key, frame num, channel, H, W]
        """

        # ANN feature rearrange to transformer input shapes
        s1 = self.pool1(s1)  # [bs, 64, 56, 56] --> [bs, 64, 14, 14]
        s2 = self.pool2(s2)  # [bs, 64, 28, 28] --> [bs, 64, 14, 14]

        s1 = rearrange(s1, 'b c h w -> b (h w) c')
        s2 = rearrange(s2, 'b c h w -> b (h w) c')
        s3 = rearrange(s3, 'b c h w -> b (h w) c')
        s4 = rearrange(s4, 'b c h w -> b (h w) c')

        reset_net(self)
        output_list = []
        for t in range(self.total_timestep):
            out = self.bn1(self.conv1(x[:, t, ...]))
            # layer 1
            out = self.layer1(out)  # [bs, 64, 56, 56]
            out = self.pool1(out)   # [bs, 64, 14, 14]
            out = self.fusion1(out, s1)  # [bs, 64, 14, 14]
            out = self.upsample1(out) # [bs, 64, 56, 56]
            # layer 2
            out = self.layer2(out)  # [bs, 128, 28, 28]
            out = self.pool2(out)  # [bs, 128, 14, 14]
            out = self.fusion2(out, s2)  # [bs, 128, 14, 14]
            out = self.upsample2(out) # [bs, 128, 28, 28]
            # layer 3
            out = self.layer3(out)  # [bs, 256, 14, 14]
            out = self.fusion3(out, s3)  # [bs, 256, 14, 14]
            # layer 4
            out = self.layer4(out)  # [bs, 512, 7, 7]
            out = self.fusion4(out, s4)  # [bs, 512, 7, 7]

            output_list.append(out)

        return torch.stack(output_list, dim=1)


class CrossAttenFusion(nn.Module):
    """ 
        cross attention fusion module
    """
    def __init__(self, backbone='resnet18', 
                 key_frame_stride=4, 
                 width_mult=4, 
                 num_classes=51,
                 ann_backbone='resnet18',
                 dropout=0,               # custom transformer support
                 transformer_depth=1,     # custom transformer support
                 num_heads=1,             # custom transformer support
                 dim_head=32,             # custom transformer support
                 downsample=0,
                 as_pretrained=False
                 ):

        super().__init__()

        self.as_pretrained = as_pretrained
        
        ####################### backbones ##############################
        # backbones for ANN and SNN feature extraction
        ann_backbone = ANNResNetBackbone(backbone=backbone)
        # ann resnet backbone layers
        ann_modules = list(ann_backbone.backbone.children())
        self.ann_layer0 = nn.Sequential(*ann_modules[:4])
        self.ann_layer1 = ann_modules[4]
        self.ann_layer2 = ann_modules[5]
        self.ann_layer3 = ann_modules[6]
        self.ann_layer4 = ann_modules[7]

        # snn resnet backbone; crossattn layers integrated inside
        timestep = key_frame_stride - 1
        if downsample:
            self.snn_backbone = attention_resnet18_downsample(total_timestep=timestep,
                                               width_mult=width_mult,
                                               ann_backbone=backbone,
                                               dropout=dropout,
                                               transformer_depth=transformer_depth,
                                               num_heads=num_heads,
                                               dim_head=dim_head,
                                               )
        else:
            self.snn_backbone = attention_resnet18(total_timestep=timestep,
                                                   width_mult=width_mult,
                                                   ann_backbone=backbone,
                                                   dropout=dropout,
                                                   transformer_depth=transformer_depth,
                                                   num_heads=num_heads,
                                                   dim_head=dim_head,
                                                   )

        ####################### classifier ##############################
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        ann_inplanes = 512 if backbone == 'resnet18' else 2048
        self.in_planes = 8 * 16 * width_mult + ann_inplanes
        self.fc = nn.Linear(self.in_planes, num_classes)

        ################# data pre-processing config ####################
        self.stride = key_frame_stride


    def _forward(self, x):
        """
        input x: [B, T, C, H, W]
        -------------------------------------------------------
         1. devide x into keyframes and residule frames: 
            x_key = [B, ::4, C, H, W]
            x_key = [B * (T/4), C, H, W]
            x_res = [B, (T/4), (4-1), C, H, W]
            x_res = [B*(T/4), (4-1), C, H, W]
         2. forward x_key to ANN layer by layer:
            s1, s2, s3, s4: [B*(T/4), c, h, w]
         3. forward x_res to attention SNN backbone:
            out_res = [B*(T/4), (4-1), C, H, W]
         4. average SNN output along the stride dimension:
            out_res = [B*(T/4), C, H, W]
            SNN out classifications now equal to ANN out num
         5. concat ANN and SNN features after the last resblock
         6. apply pooling and FC layers
        """

        ######################## input pre-processing #######################

        B, T, C, H, W = x.shape
        key_num = T // self.stride            # eg. stride = 4, key frame num = T / 4
        res_num = key_num * (self.stride - 1) # eg. stride = 4, res frame num = T - T / 4

        x = x.view(B, key_num, self.stride, C, H, W)
        x_key = x[:, :, 0, ...]                                     # [B, T/4, C, H, W]
        x_res = (x - x_key[:, :, None, ...])[:, :, 1:, ...]         # [B, (T/4), (4-1), C, H, W]

        # reshape x_res and x_key to network input shapes
        x_key = rearrange(x_key, 'b n c h w -> (b n) c h w')        # [B * (T/4), C, H, W]
        x_res = rearrange(x_res, 'b n s c h w -> (b n) s c h w', 
                          n=key_num, s=self.stride-1)               # [B * (T/4), (4-1), C, H, W]

        ############################## forward #############################
        # layer0 - before residual blocks
        # ANN backbone forward
        ann_fea = self.ann_layer0(x_key)   # [B * (T/4), c, 56, 56]
        s1 = self.ann_layer1(ann_fea)      # [B * (T/4), c, 56, 56]
        s2 = self.ann_layer2(s1)           # [B * (T/4), c, 28, 28]
        s3 = self.ann_layer3(s2)           # [B * (T/4), c, 14, 14]
        s4 = self.ann_layer4(s3)           # [B * (T/4), c, 7, 7]

        # Attention SNN backbone forward
        snn_fea = self.snn_backbone(x_res, s1, s2, s3, s4)          # [B * (T/4), (4-1), c, 7, 7]
        
        # average the SNN output along the stride dimension
        snn_fea = snn_fea.mean(1)                                   # [B * (T/4), c, 7, 7]
        
        # concatenate ANN and SNN features
        # ann_fea = rearrange(s4, 'b (h w) c  ->  b c h w', h=7, w=7) # [B * (T/4), c, 7, 7]
        ann_fea = s4
        fuse_fea = torch.concatenate([ann_fea, snn_fea], dim=1)     # [B * (T/4), c1 + c2, 7, 7]

        # average pooling
        out = self.avgpool(fuse_fea)                                # [B * (T/4), c1 + c2, 1, 1]

        # fc layer for final classification
        out = out.view(B, key_num, -1)                              # [B, T/4, c1 + c2]
        out = self.fc(out)                                          # [B, T/4, num_classes]   
        # return out.mean(1)
        return out

    def forward(self, x):
        if self.as_pretrained:
            return self._forward(x)
        else:
            return self._forward(x).mean(1)

def attention_resnet18(**kwargs):
    return AttentionResNet(BasicBlock, [2,2,2,2], **kwargs)

def attention_resnet18_downsample(**kwargs):
    return AttentionResNet_Downsample(BasicBlock, [2,2,2,2], **kwargs)

