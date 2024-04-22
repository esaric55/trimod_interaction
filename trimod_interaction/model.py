import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1, downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * planes // 32
        self.conv1 = conv1x1x1(inplanes, mid_planes)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = conv1x1x1(mid_planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNeXt(nn.Module):

    def __init__(self, input_dim, block, layers, block_inplanes, cardinality=32, no_max_pool=False):
        super(ResNeXt, self).__init__()
        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool  # Added this line

        self.conv1 = nn.Conv3d(input_dim, self.in_planes, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0], cardinality)
        self.layer2 = self._make_layer(block, block_inplanes[1], layers[1], cardinality, stride=2)
        self.layer3 = self._make_layer(block, block_inplanes[2], layers[2], cardinality, stride=2)
        self.layer4 = self._make_layer(block, block_inplanes[3], layers[3], cardinality, stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.classifiers = nn.ModuleList()

        # Fixed linear classifiers
        self.classifier = nn.Linear(block_inplanes[3] * block.expansion, 3)  # Output size 3

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, cardinality, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1x1(self.in_planes, planes * block.expansion, stride), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.in_planes, planes, cardinality, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x: batch size, window, channels, width, height
        x = x.permute(0, 2, 1, 4, 3)
        # x: batch size, channels, window, height, width

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # Apply the classifiers
        x = F.softmax(self.classifier(x), dim=1)  # Using softmax for mutually exclusive labels

        return x

def generate_model(input_dim):
    layers = [3, 4, 6, 3]
    block_inplanes = [128, 256, 512, 1024]
    model = ResNeXt(input_dim, ResNeXtBottleneck, layers, block_inplanes)
    return model
