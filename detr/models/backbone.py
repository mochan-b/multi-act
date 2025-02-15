# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        # for name, parameter in backbone.named_parameters(): # only train later layers # TODO do we want this?
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        xs = self.body(tensor)
        return xs
        # out: Dict[str, NestedTensor] = {}
        # for name, x in xs.items():
        #     m = tensor_list.mask
        #     assert m is not None
        #     mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        #     out[name] = NestedTensor(x, mask)
        # return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d) # pretrained # TODO do we want frozen batch_norm??
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class MobileNetBackbone(BackboneBase):
    """MobileNet backbone for lightweight processing."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool):
        if name == 'mobilenet_v2':
            backbone = torchvision.models.mobilenet_v2(pretrained=is_main_process())
            num_channels = 1280
        elif name == 'mobilenet_v3_small':
            backbone = torchvision.models.mobilenet_v3_small(pretrained=is_main_process())
            num_channels = 576
        elif name == 'mobilenet_v3_large':
            backbone = torchvision.models.mobilenet_v3_large(pretrained=is_main_process())
            num_channels = 960
        else:
            raise ValueError(f"Unknown MobileNet variant: {name}")
        
        # MobileNet doesn't have layer1, layer2, etc. structure like ResNet
        # We'll use the features module and extract from specific indices
        if return_interm_layers:
            # For MobileNetV2: features[3], features[7], features[14], features[18]
            # For MobileNetV3: features[3], features[6], features[9], features[12]
            if 'v2' in name:
                return_layers = {"3": "0", "7": "1", "14": "2", "18": "3"}
            else:  # v3
                return_layers = {"3": "0", "6": "1", "9": "2", "12": "3"}
        else:
            # Use the last layer
            if 'v2' in name:
                return_layers = {"18": "0"}  # Last conv layer in MobileNetV2
            else:  # v3 (small has 13 layers: 0-12, large may be different)
                if 'small' in name:
                    return_layers = {"12": "0"}  # Last conv layer in MobileNetV3 Small
                else:
                    return_layers = {"16": "0"}  # Last conv layer in MobileNetV3 Large
        
        # Initialize parent class manually to avoid double IntermediateLayerGetter
        super(BackboneBase, self).__init__()
        self.body = IntermediateLayerGetter(backbone.features, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        xs = self.body(tensor)
        return xs


class EfficientNetBackbone(BackboneBase):
    """EfficientNet backbone for efficient processing."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool):
        if name == 'efficientnet_b0':
            backbone = torchvision.models.efficientnet_b0(pretrained=is_main_process())
            num_channels = 1280
        elif name == 'efficientnet_b1':
            backbone = torchvision.models.efficientnet_b1(pretrained=is_main_process())
            num_channels = 1280
        else:
            raise ValueError(f"Unknown EfficientNet variant: {name}")
        
        # EfficientNet structure: features[0-8] for different stages
        if return_interm_layers:
            return_layers = {"2": "0", "4": "1", "6": "2", "8": "3"}
        else:
            return_layers = {"8": "0"}  # Last conv layer
        
        # Initialize parent class manually to avoid double IntermediateLayerGetter
        super(BackboneBase, self).__init__()
        self.body = IntermediateLayerGetter(backbone.features, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        xs = self.body(tensor)
        return xs


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))

        return out, pos


def build_backbone(args, backbone_name=None):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    
    # Use provided backbone_name or fall back to args.backbone
    backbone_name = backbone_name or args.backbone
    
    # Choose the appropriate backbone class based on the name
    if backbone_name.startswith('mobilenet'):
        backbone = MobileNetBackbone(backbone_name, train_backbone, return_interm_layers)
    elif backbone_name.startswith('efficientnet'):
        backbone = EfficientNetBackbone(backbone_name, train_backbone, return_interm_layers)
    else:
        # Default to ResNet backbone
        backbone = Backbone(backbone_name, train_backbone, return_interm_layers, args.dilation)
    
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
