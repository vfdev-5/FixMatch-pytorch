# Wide-ResNet-28-2
# https://github.com/google-research/fixmatch/blob/08d9b83d7cc87e853e6afc5a86b12aacff56cdea/libml/models.py#L62

import torch
import torch.nn as nn


class Residual(nn.Module):
    def __init__(
        self, in_channels, out_channels, stride, activate_before_residual=False
    ):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels, momentum=0.001)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.001)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        self.activate_before_residual = activate_before_residual
        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, padding=0
            )

    def forward(self, x0):
        x = self.leaky_relu(self.bn1(x0))
        if self.activate_before_residual:
            x0 = x
        x = self.leaky_relu(self.bn2(self.conv1(x)))
        x = self.conv2(x)
        x0 = self.skip(x0)
        return x0 + x


class WideResNet(nn.Module):
    def __init__(self, num_classes=10, filters=32, repeat=4):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)

        b0 = nn.Sequential(
            Residual(16, filters, stride=1, activate_before_residual=True),
            *[Residual(filters, filters, stride=1) for _ in range(repeat - 1)]
        )
        b1 = nn.Sequential(
            Residual(filters, filters * 2, stride=2),
            *[Residual(filters * 2, filters * 2, stride=1) for _ in range(repeat - 1)]
        )
        b2 = nn.Sequential(
            Residual(filters * 2, filters * 4, stride=2),
            *[Residual(filters * 4, filters * 4, stride=1) for _ in range(repeat - 1)]
        )
        self.res_blocks = nn.Sequential(b0, b1, b2)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        self.bn = nn.BatchNorm2d(filters * 4, momentum=0.001)
        self.reduce = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(filters * 4, num_classes)

        self.init_weights()

    def forward(self, x):
        x = self.conv(x)
        x = self.res_blocks(x)
        x = self.bn(x)
        x = self.leaky_relu(x)
        x = self.reduce(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(
                    m.weight,
                    std=torch.tensor(
                        0.5 * m.kernel_size[0] * m.kernel_size[0] * m.out_channels
                    ).rsqrt(),
                )
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
