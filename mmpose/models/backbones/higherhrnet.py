# Copyright (c) OpenMMLab. All rights reserved.
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from mmpose.registry import MODELS
from .base_backbone import BaseBackbone
from .resnet import BasicBlock, Bottleneck, get_expansion

class ScaleAwareModule(BaseModule):
    """Scale-aware feature processing module."""
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg=dict(type='BN')):
        super().__init__()
        
        self.scale_conv = nn.Conv2d(
            in_channels, out_channels, 3, 1, 1)
        self.scale_conv2 = nn.Conv2d(
            out_channels, out_channels, 3, 1, 1)
        self.norm = build_norm_layer(norm_cfg, out_channels)[1]
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.scale_conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.scale_conv2(x)
        return x

@MODELS.register_module()
class HigherHRNet(BaseBackbone):
    """HigherHRNet backbone.
    
    Implements the HigherHRNet architecture from:
    "HigherHRNet: Scale-Aware Representation Learning for Bottom-Up Human Pose Estimation"
    """

    def __init__(self,
                 extra,
                 in_channels=3,
                 num_joints=17,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 norm_eval=False,
                 with_cp=False,
                 frozen_stages=-1,
                 init_cfg=None,
                 multiscale_output=True):
        super().__init__(init_cfg=init_cfg)
        
        self.extra = extra
        self.num_joints = num_joints
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.frozen_stages = frozen_stages
        self.multiscale_output = multiscale_output

        # Build HRNet backbone
        self._make_stem_layer()
        self._make_stages()
        
        # Build scale-aware modules
        self.scale_aware_modules = nn.ModuleList([
            ScaleAwareModule(c, c, norm_cfg)
            for c in self.channels
        ])
        
        # Build multi-resolution heads
        self.heads = nn.ModuleList([
            nn.ModuleDict({
                'heatmap': nn.Conv2d(c, num_joints, 1),
                'tagmap': nn.Conv2d(c, num_joints, 1)
            }) for c in self.channels
        ])
        
        # Initialize weights
        self.init_weights()

    def _make_stages(self):
        # Similar to HRNet stage building but with additional
        # scale-aware feature processing
        ...

    def forward(self, x):
        """Forward function."""
        # Get multi-scale features from HRNet
        hrnet_features = super().forward(x)
        
        # Process features with scale-aware modules
        processed_features = []
        for feat, scale_module in zip(hrnet_features, self.scale_aware_modules):
            processed = scale_module(feat)
            processed_features.append(processed)
            
        # Generate predictions at multiple resolutions
        outputs = []
        for feat, head in zip(processed_features, self.heads):
            heatmap = head['heatmap'](feat)
            tagmap = head['tagmap'](feat)
            
            # Apply sigmoid to heatmap
            heatmap = torch.sigmoid(heatmap)
            
            outputs.append({
                'heatmap': heatmap,
                'tagmap': tagmap,
                'feature': feat
            })
            
        return outputs

    def train_step(self, data_batch, optimizer, **kwargs):
        """Training step with multi-resolution supervision."""
        losses = dict()
        
        # Forward
        outputs = self(data_batch['img'])
        
        # Calculate losses at each resolution
        for idx, output in enumerate(outputs):
            scale_weight = 1.0 / (2 ** idx)  # Higher weight for higher resolution
            
            # Heatmap loss
            losses[f'heatmap_loss_{idx}'] = self.heatmap_loss(
                output['heatmap'],
                data_batch[f'heatmap_{idx}']
            ) * scale_weight
            
            # Tag loss
            losses[f'tag_loss_{idx}'] = self.tag_loss(
                output['tagmap'],
                data_batch[f'tagmap_{idx}']
            ) * scale_weight
            
        return losses