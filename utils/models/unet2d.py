import torch.nn as nn
import torch.nn.functional as F
from .utils_unet import init_weights, unetConv2, unetUp


class Encoder(nn.Module):
    def __init__(self, feature_scale=2, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True, cfg=None):
        super(Encoder, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.cfg = cfg

        self.filters = [64, 128, 256, 512, 1024, 1024, 512]
        self.filters = [int(x / self.feature_scale) for x in self.filters]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, self.filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(self.filters[0], self.filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(self.filters[1], self.filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(self.filters[2], self.filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(self.filters[3], self.filters[4], self.is_batchnorm)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        return center, conv4, conv3, conv2, conv1


class Decoder(nn.Module):
    def __init__(self, feature_scale=2, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True, cfg=None):
        super(Decoder, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.cfg = cfg

        self.filters = [64, 128, 256, 512, 1024, 1024, 512]
        self.filters = [int(x / self.feature_scale) for x in self.filters]

        # upsampling
        self.up_concat4 = unetUp(self.filters[4], self.filters[3], self.is_deconv)
        self.up_concat3 = unetUp(self.filters[3], self.filters[2], self.is_deconv)
        self.up_concat2 = unetUp(self.filters[2], self.filters[1], self.is_deconv)
        self.up_concat1 = unetUp(self.filters[1], self.filters[0], self.is_deconv)

        # self.final = nn.Conv2d(self.filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, center, conv4, conv3, conv2, conv1):
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        # up1 = self.final(up1)

        return up1


class BaseUnet2D(nn.Module):

    def __init__(self, feature_scale=2, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True, cfg=None):
        super(BaseUnet2D, self).__init__()
        self.feature_scale = feature_scale
        self.cfg = cfg
        self.is_batchnorm = is_batchnorm
        self.filters = [64, 128, 256, 512, 1024, 1024, 512]
        self.filters = [int(x / self.feature_scale) for x in self.filters]
        self.encoder = Encoder(feature_scale, n_classes, is_deconv, in_channels, is_batchnorm, cfg)
        self.decoder = Decoder(feature_scale, n_classes, is_deconv, in_channels, is_batchnorm, cfg)

        # final conv (without any concat)
        self.final = nn.Conv2d(self.filters[0], n_classes, 1)

    def forward(self, inputs):

        center, conv4, conv3, conv2, conv1 = self.encoder(inputs)
        up1 = self.decoder(center, conv4, conv3, conv2, conv1)

        final = self.final(up1)
        return {"seg_final": final}

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p


class ProjUnet2D(BaseUnet2D):
    def __init__(self, feature_scale=2, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True, cfg=None,):
        super(ProjUnet2D, self).__init__(feature_scale, n_classes, is_deconv, in_channels, is_batchnorm, cfg)

        self.proj_center = nn.Sequential(
            unetConv2(self.filters[4], self.filters[5], self.is_batchnorm, n=3, ks=2, stride=1, padding=0),
            nn.Flatten(),
            nn.Linear(self.filters[5], self.filters[6])
        )

        if cfg.MODEL.PROJECT_NUM == 5:
            self.proj_final = nn.Sequential(
                unetConv2(self.filters[0], self.filters[2], self.is_batchnorm, n=2, ks=4, stride=4, padding=0),
                unetConv2(self.filters[2], self.filters[3], self.is_batchnorm, n=1, ks=4, stride=4, padding=0),
            )
        else:
            raise ValueError('cfg.MODEL.PROJECT_NUM not in [5]')

    def forward(self, inputs, output_final_feat=False):
        center, conv4, conv3, conv2, conv1 = self.encoder(inputs)
        up1 = self.decoder(center, conv4, conv3, conv2, conv1)
        final = self.final(up1)

        rt = {"seg_final": final}
        if output_final_feat:
            rt["proj_final"] = self.proj_final(up1)
        return rt


def unet_2D(method, cfg):
    modal_dict = {"Base": BaseUnet2D, "Proj": ProjUnet2D}
    return modal_dict[method](n_classes=cfg.DATA.SEG_CLASSES, feature_scale=cfg.MODEL.FEATURE_SCALE,
                              in_channels=cfg.DATA.INP_CHANNELS, cfg=cfg)
