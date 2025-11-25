import torch
import torch.nn as nn
from dataset import AttentionGate

from pytorch_wavelets import DWTForward

"""
U-Net模型实现，增加了注意力机制以提高裂缝检测精度。
本模型是标准U-Net的改进版本，通过在解码器部分添加注意力门控模块，
使网络能够更好地关注裂缝相关特征，抑制背景干扰。
"""

class DoubleConv(nn.Module):
    """
    双重卷积模块：U-Net的基本构建块
    包含两个连续的3×3卷积层，每个卷积后接BatchNorm和ReLU激活
    这种组合可以增强特征提取能力并加速训练过程
    
    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class HWDownsampling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(HWDownsampling, self).__init__()
        self.wt = DWTForward(J=1, wave='haar', mode='zero')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_channel * 4, out_channel, kernel_size=1, stride=1),
            nn.InstanceNorm2d(out_channel),
            nn.PReLU(),
        )

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        return x



class ec_conv_block(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(ec_conv_block, self).__init__()

        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
        #     nn.BatchNorm2d(out_ch),
        #     nn.PReLU()
        # )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
        #     nn.BatchNorm2d(out_ch),
        #     nn.PReLU()
        # )

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.PReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.PReLU()
        )
    def forward(self, x):
        # out1 = self.conv1(x)
        # out2 = self.conv2(out1)
        out = self.conv(x)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, stride=1, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        pool_out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(pool_out)
        out = self.sigmoid(out)
        return out * x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.PReLU()
        )

    def forward(self, x):
        x = self.up(x)
        return x


class dec_conv_block(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(dec_conv_block, self).__init__()

        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
        #     nn.BatchNorm2d(out_ch),
        #     nn.PReLU()
        # )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
        #     nn.BatchNorm2d(out_ch),
        #     nn.PReLU()
        # )

        self.spa = SpatialAttention().cuda()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.PReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.PReLU()
        )
    def forward(self, x):
        out1 = self.conv(x)
        out2 = self.spa(out1)
        out = out2
        return out


class UNet(nn.Module):
    """
    改进型U-Net网络，用于裂缝检测
    
    主要改进:
    1. 在每个解码器层添加注意力门控(AttentionGate)
    2. 使用BatchNorm加速训练并提高稳定性
    
    参数:
        in_channels: 输入图像通道数，默认为3(RGB图像)
        out_channels: 输出通道数，默认为1(二元分割掩码)
    """
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        # 编码器路径：连续的下采样过程，每层特征通道数翻倍
        self.enc1 = ec_conv_block(in_channels, 64)  # 第一层编码器
        self.enc2 = ec_conv_block(128, 128)          # 第二层编码器
        self.enc3 = ec_conv_block(256, 256)         # 第三层编码器
        self.enc4 = ec_conv_block(512, 512)         # 第四层编码器
        
        # 网络最深层的瓶颈部分，具有最大的特征通道数
        self.bottleneck = ec_conv_block(1024, 1024)
        
        # 解码器路径：结合转置卷积上采样和注意力机制
        # 每层包括:
        # 1. 转置卷积上采样
        # 2. 注意力门控，增强相关特征
        # 3. 特征拼接
        # 4. 双重卷积处理
        
        # 第一层解码器(最深层)
        self.up4 = up_conv(1024, 512)  # 上采样
        self.att4 = AttentionGate(F_g=512, F_l=512, F_int=256)             # 注意力门控
        self.dec4 = dec_conv_block(1024, 512)                                  # 处理拼接后的特征
        
        # 第二层解码器
        self.up3 = up_conv(512, 256)
        self.att3 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.dec3 = dec_conv_block(512, 256)
        
        # 第三层解码器
        self.up2 = up_conv(256, 128)
        self.att2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.dec2 = dec_conv_block(256, 128)
        
        # 第四层解码器(最浅层)
        self.up1 = up_conv(128, 64)
        self.att1 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.dec1 = dec_conv_block(128, 64)
        
        # 最终输出层：1x1卷积将特征图映射为所需的分割掩码
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # 下采样操作：使用最大池化
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down1 = HWDownsampling(64,128)
        self.down2 = HWDownsampling(128, 256)
        self.down3 = HWDownsampling(256, 512)
        self.down4 = HWDownsampling(512, 1024)

    def forward(self, x):
        """
        前向传播过程:
        1. 编码器路径提取多尺度特征
        2. 瓶颈层捕获全局上下文
        3. 解码器路径结合注意力机制恢复空间细节
        """
        # 编码路径: 提取多尺度特征
        enc1 = self.enc1(x)                    # 最高分辨率特征
        enc2 = self.enc2(self.down1(enc1))      # 第一次下采样
        enc3 = self.enc3(self.down2(enc2))      # 第二次下采样
        enc4 = self.enc4(self.down3(enc3))      # 第三次下采样
        
        # 瓶颈: 最低分辨率，最高通道数
        bottleneck = self.bottleneck(self.down4(enc4))
        
        # 解码路径: 结合注意力机制的特征融合
        # 处理最深层特征
        dec4 = self.up4(bottleneck)            # 上采样瓶颈特征
        enc4_att = self.att4(dec4, enc4)       # 注意力处理编码器特征
        dec4 = torch.cat((dec4, enc4_att), dim=1)  # 拼接特征
        dec4 = self.dec4(dec4)                 # 处理拼接后的特征
        
        # 处理第三层特征
        dec3 = self.up3(dec4)
        enc3_att = self.att3(dec3, enc3)
        dec3 = torch.cat((dec3, enc3_att), dim=1)
        dec3 = self.dec3(dec3)
        
        # 处理第二层特征
        dec2 = self.up2(dec3)
        enc2_att = self.att2(dec2, enc2)
        dec2 = torch.cat((dec2, enc2_att), dim=1)
        dec2 = self.dec2(dec2)
        
        # 处理第一层特征(最浅层)
        dec1 = self.up1(dec2)
        enc1_att = self.att1(dec1, enc1)
        dec1 = torch.cat((dec1, enc1_att), dim=1)
        dec1 = self.dec1(dec1)
        
        # 最终输出: 生成裂缝分割掩码
        # 注意：需要在外部使用sigmoid函数将输出转换为概率
        return self.final_conv(dec1) 
