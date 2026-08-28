# -*- coding: utf-8 -*-
"""
SCRFD 网络结构（参考 insightface 官方实现 scrfd/mmdet/models）
scrfd_10g_bnkps 配置：
  - backbone: ResNetV1e (BasicBlock, stage_blocks=(3,4,2,3), stage_planes=[56,88,88,224],
              deep_stem=True, avg_down=True, no_pool33=True)
  - neck:     PAFPN (in_channels=[56,88,88,224], out_channels=56, start_level=1, num_outs=3)
  - head:     SCRFDHead (num_classes=1, in_channels=56, stacked_convs=3, feat_channels=80,
              cls_reg_share=True, strides_share=False, scale_mode=2, use_kps=True, reg_max=8)
纯 PyTorch 实现，不依赖 mmcv/mmdet。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvModule(nn.Module):
    """简化版 mmcv ConvModule：Conv2d + (BN) + (ReLU)"""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=False,
                 norm_cfg=None, act=True, inplace=False):
        super(ConvModule, self).__init__()
        use_bias = norm_cfg is None
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=use_bias if bias is None else bias)
        if norm_cfg is not None:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        self.relu = nn.ReLU(inplace=inplace) if act else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Scale(nn.Module):
    """mmcv Scale：学习标量"""

    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x):
        return x * self.scale


class Integral(nn.Module):
    """固定层：从分布计算积分结果（DFL 使用，本模型 use_dfl=False 不参与推理）"""

    def __init__(self, reg_max=8):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer('project', torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x):
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, self.project.type_as(x)).reshape(-1, 4)
        return x


class BasicBlock(nn.Module):
    """BasicBlock for ResNetV1e (avg_down=True 时 downsample 前先 AvgPool)"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, avg_down=False, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.avg_down = avg_down
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResLayer(nn.Sequential):
    """ResLayer：构建 ResNet stage"""

    def __init__(self, block, inplanes, planes, num_blocks, stride=1, avg_down=False):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            if avg_down:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(kernel_size=stride, stride=stride,
                                 ceil_mode=True, count_include_pad=False))
            downsample.extend([
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1,
                          stride=conv_stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            ])
            downsample = nn.Sequential(*downsample)

        layers = []
        layers.append(block(inplanes=inplanes, planes=planes, stride=stride,
                            avg_down=avg_down, downsample=downsample))
        inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(inplanes=inplanes, planes=planes, stride=1,
                                avg_down=avg_down, downsample=None))
        super(ResLayer, self).__init__(*layers)


class ResNetV1e(nn.Module):
    """SCRFD backbone：ResNetV1e (deep_stem + avg_down + no_pool33)"""

    def __init__(self, in_channels=3, stem_channels=56,
                 block_cfg=dict(block='BasicBlock', stage_blocks=(3, 4, 2, 3),
                                stage_planes=[56, 88, 88, 224]),
                 num_stages=4, strides=(1, 2, 2, 2), out_indices=(0, 1, 2, 3)):
        super(ResNetV1e, self).__init__()
        self.block = BasicBlock if block_cfg['block'] == 'BasicBlock' else None
        self.stage_blocks = list(block_cfg['stage_blocks'])
        stage_planes = block_cfg.get('stage_planes', [56, 88, 88, 224])
        self.num_stages = num_stages
        self.out_indices = out_indices
        self.inplanes = stem_channels

        # deep_stem: 3 个 3x3 conv
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels // 2, kernel_size=3, stride=2,
                      padding=1, bias=False),
            nn.BatchNorm2d(stem_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(stem_channels // 2, stem_channels // 2, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(stem_channels // 2, stem_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )
        # no_pool33=True -> MaxPool2d(2, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.res_layers = []
        for i in range(num_stages):
            stride = strides[i]
            planes = stage_planes[i]
            res_layer = ResLayer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=self.stage_blocks[i],
                stride=stride,
                avg_down=True)  # ResNetV1e avg_down=True
            self.inplanes = planes * self.block.expansion
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

    def forward(self, x):
        x = self.stem(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)


class PAFPN(nn.Module):
    """Path Aggregation Feature Pyramid Network"""

    def __init__(self, in_channels, out_channels, num_outs,
                 start_level=0, end_level=-1, add_extra_convs=False):
        super(PAFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.start_level = start_level

        if end_level == -1:
            self.backbone_end_level = self.num_ins
        else:
            self.backbone_end_level = end_level

        self.add_extra_convs = add_extra_convs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(in_channels[i], out_channels, 1,
                                norm_cfg=None, act=True, bias=None, inplace=False)
            fpn_conv = ConvModule(out_channels, out_channels, 3, padding=1,
                                  norm_cfg=None, act=True, bias=None, inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # PAFPN bottom-up
        self.downsample_convs = nn.ModuleList()
        self.pafpn_convs = nn.ModuleList()
        for i in range(self.start_level + 1, self.backbone_end_level):
            d_conv = ConvModule(out_channels, out_channels, 3, stride=2, padding=1,
                                norm_cfg=None, act=True, bias=None, inplace=False)
            pafpn_conv = ConvModule(out_channels, out_channels, 3, padding=1,
                                    norm_cfg=None, act=True, bias=None, inplace=False)
            self.downsample_convs.append(d_conv)
            self.pafpn_convs.append(pafpn_conv)

        # 额外层（num_outs > 层数时），本配置用不上
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                extra_fpn_conv = ConvModule(out_channels, out_channels, 3,
                                            stride=2, padding=1,
                                            norm_cfg=None, act=True, bias=None,
                                            inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(laterals[i], size=prev_shape, mode='nearest')

        # part 1: from original levels
        inter_outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]

        # part 2: add bottom-up path
        for i in range(0, used_backbone_levels - 1):
            inter_outs[i + 1] += self.downsample_convs[i](inter_outs[i])

        outs = []
        outs.append(inter_outs[0])
        outs.extend([
            self.pafpn_convs[i - 1](inter_outs[i])
            for i in range(1, used_backbone_levels)
        ])

        # part 3: add extra levels
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                if self.add_extra_convs == 'on_output':
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    outs.append(self.fpn_convs[i](F.relu(outs[-1])))
        return tuple(outs)


class SCRFDHead(nn.Module):
    """SCRFDHead：与官方 mmdet 实现一致（推理模式）"""

    def __init__(self, num_classes=1, in_channels=56, stacked_convs=3,
                 feat_channels=80, cls_reg_share=True, strides_share=False,
                 scale_mode=2, use_kps=True, reg_max=8, num_anchors=2,
                 strides=[8, 16, 32], use_dfl=False):
        super(SCRFDHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.stacked_convs = stacked_convs
        self.feat_channels = feat_channels
        self.cls_reg_share = cls_reg_share
        self.strides_share = strides_share
        self.scale_mode = scale_mode
        self.use_kps = use_kps
        self.reg_max = reg_max
        self.num_anchors = num_anchors
        self.strides = [tuple([s, s]) for s in strides]
        self.use_dfl = use_dfl
        self.use_scale = scale_mode > 0 and (strides_share or scale_mode == 2)
        self.NK = 5
        self.cls_out_channels = num_classes

        self.integral = Integral(reg_max)

        conv_strides = [0] if self.strides_share else self.strides
        self.cls_stride_convs = nn.ModuleDict()
        self.stride_cls = nn.ModuleDict()
        self.stride_reg = nn.ModuleDict()
        if self.use_kps:
            self.stride_kps = nn.ModuleDict()

        for stride_idx, conv_stride in enumerate(conv_strides):
            key = str(conv_stride)
            cls_convs = nn.ModuleList()
            for i in range(stacked_convs):
                chn = self.in_channels if i == 0 else feat_channels
                cls_convs.append(
                    ConvModule(chn, feat_channels, 3, padding=1,
                               norm_cfg=dict(type='BN'), act=True, bias=False, inplace=False))
            self.cls_stride_convs[key] = cls_convs
            self.stride_cls[key] = nn.Conv2d(
                feat_channels, self.cls_out_channels * self.num_anchors, 3, padding=1)
            if not self.use_dfl:
                self.stride_reg[key] = nn.Conv2d(
                    feat_channels, 4 * self.num_anchors, 3, padding=1)
            else:
                self.stride_reg[key] = nn.Conv2d(
                    feat_channels, 4 * (self.reg_max + 1) * self.num_anchors, 3, padding=1)
            if self.use_kps:
                self.stride_kps[key] = nn.Conv2d(
                    feat_channels, self.NK * 2 * self.num_anchors, 3, padding=1)

        if self.use_scale:
            self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        else:
            self.scales = [None for _ in self.strides]

    def forward_single(self, x, scale, stride):
        """推理：返回 (cls_score, bbox_pred, kps_pred)"""
        cls_feat = x
        reg_feat = x
        cls_convs = self.cls_stride_convs[str(stride)] if not self.strides_share \
            else self.cls_stride_convs['0']
        for cls_conv in cls_convs:
            cls_feat = cls_conv(cls_feat)
        if not self.cls_reg_share:
            reg_feat = cls_feat  # 本配置 cls_reg_share=True，reg 复用 cls_feat
        else:
            reg_feat = cls_feat

        cls_score = self.stride_cls[str(stride)](cls_feat)
        _bbox_pred = self.stride_reg[str(stride)](reg_feat)
        if self.use_scale:
            bbox_pred = scale(_bbox_pred)
        else:
            bbox_pred = _bbox_pred
        if self.use_kps:
            kps_pred = self.stride_kps[str(stride)](reg_feat)
        else:
            kps_pred = bbox_pred.new_zeros(
                (bbox_pred.shape[0], self.NK * 2, bbox_pred.shape[2], bbox_pred.shape[3]))
        return cls_score, bbox_pred, kps_pred

    def forward(self, feats):
        """推理：返回 (cls_scores, bbox_preds, kps_preds)"""
        cls_scores = []
        bbox_preds = []
        kps_preds = []
        for feat, scale, stride in zip(feats, self.scales, self.strides):
            cls_score, bbox_pred, kps_pred = self.forward_single(feat, scale, stride)
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
            kps_preds.append(kps_pred)
        return cls_scores, bbox_preds, kps_preds


class SCRFD(nn.Module):
    """完整 SCRFD 检测网络"""

    def __init__(self, cfg=None):
        super(SCRFD, self).__init__()
        self.backbone = ResNetV1e()
        self.neck = PAFPN(
            in_channels=[56, 88, 88, 224],
            out_channels=56,
            start_level=1,
            num_outs=3)
        self.bbox_head = SCRFDHead(
            num_classes=1,
            in_channels=56,
            stacked_convs=3,
            feat_channels=80,
            cls_reg_share=True,
            strides_share=False,
            scale_mode=2,
            use_kps=True,
            reg_max=8,
            num_anchors=2,
            strides=[8, 16, 32],
            use_dfl=False)

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.neck(feats)
        return self.bbox_head(feats)
