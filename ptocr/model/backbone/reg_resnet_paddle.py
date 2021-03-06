import torch.nn.functional as F
import torch.nn as nn

class ConvBNLayer(nn.Module):
    def __init__(self,in_channels,
            out_channels,
            kernel_size,
            stride=1,
            groups=1,
            is_relu=False,
            is_vd_mode=False):
        super(ConvBNLayer,self).__init__()

        self.is_vd_mode = is_vd_mode
        self.is_relu = is_relu

        if is_vd_mode:
            self._pool2d_avg = nn.AvgPool2d(kernel_size=stride, stride=stride, padding=0, ceil_mode=True)

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=1 if is_vd_mode else stride,
                              padding=(kernel_size - 1) // 2,
                              groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self,x):
        if self.is_vd_mode:
            x = self._pool2d_avg(x)
        x = self.bn(self.conv(x))
        if self.is_relu:
            x = self.relu(x)
        return x


class BottleneckBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 shortcut=True,
                 if_first=False):
        super(BottleneckBlock, self).__init__()

        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            is_relu=True)

        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            is_relu=True)

        self.conv2 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels * 4,
            kernel_size=1)

        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels * 4,
                kernel_size=1,
                stride=stride,
                is_vd_mode=not if_first and stride[0] != 1)

        self.shortcut = shortcut

    def forward(self,x):
        y = self.conv0(x)
        y = self.conv2(self.conv1(y))
        if self.shortcut:
            short = x
        else:
            short = self.short(x)
        y = y+short
        y = F.relu(y)
        return y


class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 shortcut=True,
                 if_first=False):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            is_relu=True)
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3)

        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                is_vd_mode=not if_first and stride[0] != 1)

        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv0(x)
        y = self.conv1(y)

        if self.shortcut:
            short = x
        else:
            short = self.short(x)
        y = y+short
        y = F.relu(y)
        return y

class ResNet(nn.Module):
    def __init__(self, in_channels=3, layers=50, **kwargs):
        super(ResNet, self).__init__()

        self.layers = layers
        supported_layers = [18, 34, 50, 101, 152, 200]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(
                supported_layers, layers)

        if layers == 18:
            depth = [2, 2, 2, 2]
        elif layers == 34 or layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        elif layers == 200:
            depth = [3, 12, 48, 3]
        num_channels = [64, 256, 512,
                        1024] if layers >= 50 else [64, 64, 128, 256]
        num_filters = [64, 128, 256, 512]

        self.conv1_1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=3,
            stride=1,
            is_relu=True)
        self.conv1_2 = ConvBNLayer(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            is_relu=True)
        self.conv1_3 = ConvBNLayer(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            is_relu=True)
        self.pool2d_max = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        block_list = []
        if layers >= 50:
            for block in range(len(depth)):
                shortcut = False
                for i in range(depth[block]):
                    if i == 0 and block != 0:
                        stride = (2, 1)
                    else:
                        stride = (1, 1)
                    bottleneck_block = BottleneckBlock(
                            in_channels=num_channels[block]
                            if i == 0 else num_filters[block] * 4,
                            out_channels=num_filters[block],
                            stride=stride,
                            shortcut=shortcut,
                            if_first=block == i == 0)
                    shortcut = True
                    block_list.append(bottleneck_block)
                self.out_channels = num_filters[block]
        else:
            for block in range(len(depth)):
                shortcut = False
                for i in range(depth[block]):
                    if i == 0 and block != 0:
                        stride = (2, 1)
                    else:
                        stride = (1, 1)

                    basic_block = BasicBlock(
                            in_channels=num_channels[block]
                            if i == 0 else num_filters[block],
                            out_channels=num_filters[block],
                            stride=stride,
                            shortcut=shortcut,
                            if_first=block == i == 0)
                    shortcut = True
                    block_list.append(basic_block)
                self.out_channels = num_filters[block]
        self.block = nn.Sequential(*block_list)
        self.out_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        y = self.conv1_1(x)
        y = self.conv1_2(y)
        y = self.conv1_3(y)
        y = self.pool2d_max(y)
        for block in self.block:
            y = block(y)
        y = self.out_pool(y)
        return y



    
def resnet18(pretrained=False, is_gray=False,**kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if is_gray:
        in_channels = 1
    else:
        in_channels = 3
    model = ResNet(in_channels=in_channels, layers=18, **kwargs)
    if pretrained:
        pass
    return model

def resnet34(pretrained=False, is_gray=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if is_gray:
        in_channels = 1
    else:
        in_channels = 3
    model = ResNet(in_channels=in_channels, layers=34, **kwargs)
    if pretrained:
        pass
    return model

def resnet50(pretrained=False, is_gray=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if is_gray:
        in_channels = 1
    else:
        in_channels = 3
    model = ResNet(in_channels=in_channels, layers=50, **kwargs)
    if pretrained:
        pass
    return model