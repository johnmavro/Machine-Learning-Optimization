import torch.nn as nn
import torch
import torch.nn.functional as F


class GoogleNet(torch.nn.Module):
    def __init__(self, num_class=100):
        super().__init__()

        class Inception(torch.nn.Module):
            def __init__(self, input_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
                super().__init__()

                # 1x1conv branch
                self.b1 = nn.Sequential(
                    nn.Conv2d(input_channels, n1x1, kernel_size=1),
                    nn.BatchNorm2d(n1x1),
                    nn.ReLU(inplace=True)
                )

                # 1x1conv -> 3x3conv branch
                self.b2 = nn.Sequential(
                    nn.Conv2d(input_channels, n3x3_reduce, kernel_size=1),
                    nn.BatchNorm2d(n3x3_reduce),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(n3x3_reduce, n3x3, kernel_size=3, padding=1),
                    nn.BatchNorm2d(n3x3),
                    nn.ReLU(inplace=True)
                )

                # 1x1conv -> 5x5conv branch
                # we use 2 3x3 conv filters stacked instead
                # of 1 5x5 filters to obtain the same receptive
                # field with fewer parameters
                self.b3 = nn.Sequential(
                    nn.Conv2d(input_channels, n5x5_reduce, kernel_size=1),
                    nn.BatchNorm2d(n5x5_reduce),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(n5x5_reduce, n5x5, kernel_size=3, padding=1),
                    nn.BatchNorm2d(n5x5, n5x5),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
                    nn.BatchNorm2d(n5x5),
                    nn.ReLU(inplace=True)
                )

                # 3x3pooling -> 1x1conv
                # same conv
                self.b4 = nn.Sequential(
                    nn.MaxPool2d(3, stride=1, padding=1),
                    nn.Conv2d(input_channels, pool_proj, kernel_size=1),
                    nn.BatchNorm2d(pool_proj),
                    nn.ReLU(inplace=True)
                )

            def forward(self, x):
                return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)

        self.prelayer = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )

        # although we only use 1 conv layer as prelayer,
        # we still use name a3, b3.......
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        # """In general, an Inception network is a network consisting of
        # modules of the above type stacked upon each other, with occasional
        # max-pooling layers with stride 2 to halve the resolution of the
        # grid"""
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        # input feature size: 8*8*1024
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(p=0.4)
        self.linear = nn.Linear(1024, num_class)

    def forward(self, x):
        output = self.prelayer(x)
        output = self.a3(output)
        output = self.b3(output)

        output = self.maxpool(output)

        output = self.a4(output)
        output = self.b4(output)
        output = self.c4(output)
        output = self.d4(output)
        output = self.e4(output)

        output = self.maxpool(output)

        output = self.a5(output)
        output = self.b5(output)

        output = self.avgpool(output)
        output = self.dropout(output)
        output = output.view(output.size()[0], -1)
        output = self.linear(output)

        return output


class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, dropout_rate=0.3, num_classes=10):
        super().__init__()
        self.in_planes = 16

        assert ((depth-4) % 6 == 0), 'The depth of a wide-resnet should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        class wide_basic(nn.Module):
            def __init__(self, in_planes, planes, dropout_rate, stride=1):
                super(wide_basic, self).__init__()
                self.bn1 = nn.BatchNorm2d(in_planes)
                self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
                self.dropout = nn.Dropout(p=dropout_rate)
                self.bn2 = nn.BatchNorm2d(planes)
                self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

                self.shortcut = nn.Sequential()
                if stride != 1 or in_planes != planes:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
                    )

            def forward(self, x):
                out = self.dropout(self.conv1(F.relu(self.bn1(x))))
                out = self.conv2(F.relu(self.bn2(out)))
                out += self.shortcut(x)

                return out

        self.conv1 = self.conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def conv3x3(self, in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out
