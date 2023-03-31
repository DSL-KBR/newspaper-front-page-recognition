import torch
import torch.nn as nn

from networks.ResNeStlib.resnet import Bottleneck, ResNet


class FullyConvolutionalResNeSt(ResNet):

    def __init__(self, num_class=1000, model_name='resnest50_fast_1s4x24d', **kwargs):
        if model_name == 'resnest50':
            super().__init__(Bottleneck, [3, 4, 6, 3],
                             radix=2, groups=1, bottleneck_width=64,
                             deep_stem=True, stem_width=32, avg_down=True,
                             avd=True, avd_first=False, **kwargs)
            state_dict = torch.load('./pretrainedModels/resnest50-fb9de5b3.pth')
            self.load_state_dict(state_dict)
        elif model_name == 'resnest50_fast_1s4x24d':
            super().__init__(Bottleneck, [3, 4, 6, 3],
                             radix=1, groups=4, bottleneck_width=24,
                             deep_stem=True, stem_width=32, avg_down=True,
                             avd=True, avd_first=True, **kwargs)
            print('Loading previously trained resnest50')
            state_dict = torch.load('./pretrainedModels/resnest50_fast_1s4x24d-d4a4f76f.pth')
            self.load_state_dict(state_dict)

        self.avgpool = nn.AvgPool2d(7, 7)
        self.last_conv = torch.nn.Conv2d(in_channels=self.fc.in_features, out_channels=1000, kernel_size=1)

        self.last_conv.weight.data.copy_(self.fc.weight.data.view(*self.fc.weight.data.shape, 1, 1))
        self.last_conv.bias.data.copy_(self.fc.bias.data)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.last_conv(x)

        return x
