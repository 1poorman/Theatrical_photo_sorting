import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torch.nn import functional as F

from facenet.nets.inception_resnetv1 import InceptionResnetV1


class inception_resnet(nn.Module):
    def __init__(self, pretrained):
        super(inception_resnet, self).__init__()
        self.model = InceptionResnetV1()
        if pretrained:
            state_dict = load_state_dict_from_url(
                "https://github.com/bubbliiiing/facenet-pytorch/releases/download/v1.0/backbone_weights_of_inception_resnetv1.pth",
                model_dir="model_data",
                progress=True,
            )
            self.model.load_state_dict(state_dict)

    def forward(self, x):
        x = self.model.conv2d_1a(x)
        x = self.model.conv2d_2a(x)
        x = self.model.conv2d_2b(x)
        x = self.model.maxpool_3a(x)
        x = self.model.conv2d_3b(x)
        x = self.model.conv2d_4a(x)
        x = self.model.conv2d_4b(x)
        x = self.model.repeat_1(x)
        x = self.model.mixed_6a(x)
        x = self.model.repeat_2(x)
        x = self.model.mixed_7a(x)
        x = self.model.repeat_3(x)
        x = self.model.block8(x)
        return x


class Facenet(nn.Module):

    def __init__(
        self,
        backbone="inception_resnetv1",
        dropout_keep_prob=0.5,
        embedding_size=128,
        num_classes=None,
        mode="train",
        pretrained=False,
    ):
        super(Facenet, self).__init__()
        if backbone == "inception_resnetv1":
            self.backbone = inception_resnet(pretrained)
            flat_shape = 1792
        else:
            raise ValueError(
                "Unsupported backbone - `{}`, Use mobilenet, inception_resnetv1.".format(
                    backbone
                )
            )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.Dropout = nn.Dropout(1 - dropout_keep_prob)
        self.Bottleneck = nn.Linear(flat_shape, embedding_size, bias=False)
        self.last_bn = nn.BatchNorm1d(
            embedding_size, eps=0.001, momentum=0.1, affine=True
        )
        if mode == "train":
            self.classifier = nn.Linear(embedding_size, num_classes)

    def forward(self, x, mode="predict"):
        if mode == "predict":
            x = self.backbone(x)
            x = self.avg(x)
            x = x.view(x.size(0), -1)
            x = self.Dropout(x)
            x = self.Bottleneck(x)
            x = self.last_bn(x)
            x = F.normalize(x, p=2, dim=1)
            return x
        x = self.backbone(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.Dropout(x)
        x = self.Bottleneck(x)
        before_normalize = self.last_bn(x)

        x = F.normalize(before_normalize, p=2, dim=1)
        cls = self.classifier(before_normalize)
        return x, cls

    def forward_feature(self, x):
        x = self.backbone(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.Dropout(x)
        x = self.Bottleneck(x)
        before_normalize = self.last_bn(x)
        x = F.normalize(before_normalize, p=2, dim=1)
        return before_normalize, x

    def forward_classifier(self, x):
        x = self.classifier(x)
        return x


# 自定义新模型，继承原有的 Facenet
class ModifiedFacenet(nn.Module):
    def __init__(self, original_model, new_embedding_size=256):
        super(ModifiedFacenet, self).__init__()
        self.original_model = original_model
        self.new_embedding_size = new_embedding_size
        self.new_fc = nn.Linear(
            128, new_embedding_size
        )  # 新的全连接层，将 embedding_size 128 改为 512
        self.new_last_bn = nn.BatchNorm1d(
            new_embedding_size, eps=0.001, momentum=0.1, affine=True
        )

    def forward(self, x, mode="predict"):
        x = self.original_model(x, mode=mode)  # 调用原有模型的 forward 方法
        if mode == "predict":
            x = self.new_fc(x)  # 修改输出
            # x = self.new_last_bn(x)
        return x
