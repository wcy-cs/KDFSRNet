### This net.py is used for Pytorch 1.9.0 CUDA 11
from model import common
import torch.nn as nn
import torch


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y




## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
            self, n_feat, kernel_size, reduction=16,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(
                nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=kernel_size, bias=bias, padding=1))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)

        res += x
        return res





class Student(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(Student, self).__init__()

        n_resblocks = 8
        n_feats = 64
        kernel_size = 3
        scale = 8
        act = nn.ReLU(True)
        self.args = args
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module

        self.down1 = common.invUpsampler_module(conv, 2, n_feats, act=False)

        self.down_stage1 = nn.Sequential(*[RCAB(n_feat=n_feats, kernel_size=3), RCAB(n_feat=n_feats, kernel_size=3),
                                           RCAB(n_feat=n_feats, kernel_size=3), RCAB(n_feat=n_feats, kernel_size=3),
                                           RCAB(n_feat=n_feats, kernel_size=3), RCAB(n_feat=n_feats, kernel_size=3)])
        self.down2 = common.invUpsampler_module(conv, 2, n_feats, act=False)

        self.down_stage2 = nn.Sequential(*[RCAB(n_feat=n_feats, kernel_size=3), RCAB(n_feat=n_feats, kernel_size=3),
                                           RCAB(n_feat=n_feats, kernel_size=3), RCAB(n_feat=n_feats, kernel_size=3),
                                           RCAB(n_feat=n_feats, kernel_size=3), RCAB(n_feat=n_feats, kernel_size=3)])
        self.down3 = common.invUpsampler_module(conv, 2, n_feats, act=False)

        self.down_stage3 = nn.Sequential(*[RCAB(n_feat=n_feats, kernel_size=3), RCAB(n_feat=n_feats, kernel_size=3),
                                           RCAB(n_feat=n_feats, kernel_size=3), RCAB(n_feat=n_feats, kernel_size=3),
                                           RCAB(n_feat=n_feats, kernel_size=3), RCAB(n_feat=n_feats, kernel_size=3)])
        self.down4 = common.invUpsampler_module(conv, 2, n_feats, act=False)

        self.down_stage4 = nn.Sequential(*[RCAB(n_feat=n_feats, kernel_size=3), RCAB(n_feat=n_feats, kernel_size=3),
                                           RCAB(n_feat=n_feats, kernel_size=3), RCAB(n_feat=n_feats, kernel_size=3),
                                           RCAB(n_feat=n_feats, kernel_size=3), RCAB(n_feat=n_feats, kernel_size=3)])
        self.bottleneck = nn.Sequential(*[RCAB(n_feat=n_feats, kernel_size=3), RCAB(n_feat=n_feats, kernel_size=3),
                                          RCAB(n_feat=n_feats, kernel_size=3), RCAB(n_feat=n_feats, kernel_size=3),
                                          RCAB(n_feat=n_feats, kernel_size=3), RCAB(n_feat=n_feats, kernel_size=3)])
        self.up1 = common.Upsampler_module(conv, 2, n_feats, act=False)

        self.up_stage1 = nn.Sequential(*[RCAB(n_feat=n_feats, kernel_size=3), RCAB(n_feat=n_feats, kernel_size=3),
                                         RCAB(n_feat=n_feats, kernel_size=3), RCAB(n_feat=n_feats, kernel_size=3),
                                         RCAB(n_feat=n_feats, kernel_size=3), RCAB(n_feat=n_feats, kernel_size=3)])
        self.up2 = common.Upsampler_module(conv, 2, n_feats, act=False)

        self.up_stage2 = nn.Sequential(*[RCAB(n_feat=n_feats, kernel_size=3), RCAB(n_feat=n_feats, kernel_size=3),
                                         RCAB(n_feat=n_feats, kernel_size=3), RCAB(n_feat=n_feats, kernel_size=3),
                                         RCAB(n_feat=n_feats, kernel_size=3), RCAB(n_feat=n_feats, kernel_size=3)])
        self.up3 = common.Upsampler_module(conv, 2, n_feats, act=False)

        self.up_stage3 = nn.Sequential(*[RCAB(n_feat=n_feats, kernel_size=3), RCAB(n_feat=n_feats, kernel_size=3),
                                         RCAB(n_feat=n_feats, kernel_size=3), RCAB(n_feat=n_feats, kernel_size=3),
                                         RCAB(n_feat=n_feats, kernel_size=3), RCAB(n_feat=n_feats, kernel_size=3)])
        self.up4 = common.Upsampler_module(conv, 2, n_feats, act=False)

        self.up_stage4 = nn.Sequential(*[RCAB(n_feat=n_feats, kernel_size=3), RCAB(n_feat=n_feats, kernel_size=3),
                                         RCAB(n_feat=n_feats, kernel_size=3), RCAB(n_feat=n_feats, kernel_size=3),
                                         RCAB(n_feat=n_feats, kernel_size=3), RCAB(n_feat=n_feats, kernel_size=3)])
        # define tail module
        m_tail = [
            conv(n_feats, args.n_colors, kernel_size)
        ]
        self.head = nn.Sequential(*m_head)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x, parsing=None):
        intp = x
        intermediate_results = []
        res = self.head(intp)
        x1 = self.down1(res)

        x = self.down_stage1(x1)
        intermediate_results.append(x)
        x2 = self.down2(x)
        x = self.down_stage2(x2)
        intermediate_results.append(x)
        x3 = self.down3(x)
        x = self.down_stage3(x3)
        intermediate_results.append(x)
        x4 = self.down4(x)
        x = self.down_stage4(x4)
        intermediate_results.append(x)
    
        x = self.bottleneck(x)
        intermediate_results.append(x)
      

        x = self.up1(x)
        x = self.up_stage1(x)
        intermediate_results.append(x)

        x = self.up2(x)
        x = self.up_stage2(x)
        intermediate_results.append(x)
     
        x = self.up3(x)
        x = self.up_stage3(x)
        intermediate_results.append(x)

        x = self.up4(x)
        x = self.up_stage4(x)
        intermediate_results.append(x)

        x = self.tail(x)


        return intermediate_results, x

class Teacherp(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(Teacherp, self).__init__()

        n_resblocks = 8
        n_feats = 64
        kernel_size = 3
        scale = 8
        act = nn.ReLU(True)
        self.args = args


        # define head module

        m_head = [conv(args.n_colors + 3, n_feats, kernel_size)]

        # define body module

        self.down1 = common.invUpsampler_module(conv, 2, n_feats, act=False)

        self.down_stage1 = nn.Sequential(*[RCAB(n_feat=n_feats, kernel_size=3), RCAB(n_feat=n_feats, kernel_size=3)])
        self.down2 = common.invUpsampler_module(conv, 2, n_feats, act=False)

        self.down_stage2 = nn.Sequential(*[RCAB(n_feat=n_feats, kernel_size=3), RCAB(n_feat=n_feats, kernel_size=3)])
        self.down3 = common.invUpsampler_module(conv, 2, n_feats, act=False)

        self.down_stage3 = nn.Sequential(*[RCAB(n_feat=n_feats, kernel_size=3), RCAB(n_feat=n_feats, kernel_size=3)])
        self.down4 = common.invUpsampler_module(conv, 2, n_feats, act=False)

        self.down_stage4 = nn.Sequential(*[RCAB(n_feat=n_feats, kernel_size=3), RCAB(n_feat=n_feats, kernel_size=3)])
        self.bottleneck = nn.Sequential(*[RCAB(n_feat=n_feats, kernel_size=3), RCAB(n_feat=n_feats, kernel_size=3)])
        self.up1 = common.Upsampler_module(conv, 2, n_feats, act=False)

        self.up_stage1 = nn.Sequential(*[RCAB(n_feat=n_feats, kernel_size=3), RCAB(n_feat=n_feats, kernel_size=3)])
        self.up2 = common.Upsampler_module(conv, 2, n_feats, act=False)

        self.up_stage2 = nn.Sequential(*[RCAB(n_feat=n_feats, kernel_size=3), RCAB(n_feat=n_feats, kernel_size=3)])
        self.up3 = common.Upsampler_module(conv, 2, n_feats, act=False)

        self.up_stage3 = nn.Sequential(*[RCAB(n_feat=n_feats, kernel_size=3), RCAB(n_feat=n_feats, kernel_size=3)])
        self.up4 = common.Upsampler_module(conv, 2, n_feats, act=False)

        self.up_stage4 = nn.Sequential(*[RCAB(n_feat=n_feats, kernel_size=3), RCAB(n_feat=n_feats, kernel_size=3)])
        # define tail module
        m_tail = [
            conv(n_feats, args.n_colors, kernel_size)
        ]
        self.head = nn.Sequential(*m_head)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x, parsing=None):
        intp = torch.cat((x, parsing), 1)
     
        intermediate_results = []
        res = self.head(intp)
       
        x1 = self.down1(res)

        x = self.down_stage1(x1)
        intermediate_results.append(x)
        x2 = self.down2(x)
        x = self.down_stage2(x2)
        intermediate_results.append(x)
        x3 = self.down3(x)
        x = self.down_stage3(x3)
        intermediate_results.append(x)
        x4 = self.down4(x)
        x = self.down_stage4(x4)
        intermediate_results.append(x)
 
        x = self.bottleneck(x)
        intermediate_results.append(x)

        x = self.up1(x)
        x = self.up_stage1(x)
        intermediate_results.append(x)
     
        x = self.up2(x)
        x = self.up_stage2(x)
        intermediate_results.append(x)
       
        x = self.up3(x)
        x = self.up_stage3(x)
        intermediate_results.append(x)
 
        x = self.up4(x)
        x = self.up_stage4(x)
        intermediate_results.append(x)

        x = self.tail(x)


        return intermediate_results, x
