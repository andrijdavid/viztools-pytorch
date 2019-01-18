import os
import torch
import torch.nn as nn

from torchvision.datasets.utils import download_url, check_integrity


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)



url = 'https://download.pytorch.org/models/vgg16-397923af.pth'
md5 = '463aeb51ba5e122501bd03f4ad6d5374'
cfg = [64, 64, 'M',
       128, 128, 'M',
       256, 256, 256, 'M',
       512, 512, 512, 'M',
       512, 512, 512, 'M']


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg), **kwargs)
    if pretrained:
        model_name = "vgg16"
        root = os.path.dirname(os.path.abspath(__file__))
        
        file_path = os.path.join(root, f"{model_name}.pth")
        if not os.path.exists(file_path):
            download_url(url, root, f"{model_name}.pth", md5)
        if not check_integrity(file_path, md5):
            print(f"{file_path} corrupted!")
            os.system(f"rm {file_path}")
            download_url(url, root, f"{model_name}.pth", md5)

        model.load_state_dict(torch.load(file_path))

    return model
