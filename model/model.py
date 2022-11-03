from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from timm import create_model

class MyResNet18(nn.Module):
    def __init__(self):
        super(MyResNet18, self).__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        self.mask_classifier = nn.Linear(512, 3)
        self.gender_classifier = nn.Linear(512, 2)
        self.age_classifier = nn.Linear(512, 3)
        self.init_param()

    def forward(self, input):
        image_features = self.backbone(input)
        mask_out = self.mask_classifier(image_features)
        gender_out = self.gender_classifier(image_features)
        age_out = self.age_classifier(image_features)
        return mask_out, gender_out, age_out

    def init_param(self):
        for m in [self.mask_classifier, self.gender_classifier, self.age_classifier]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

class MyVit(nn.Module):
    def __init__(self):
        super(MyVit, self).__init__()
        self.backbone = create_model("vit_small_patch16_384", pretrained=True)
        self.backbone.head = nn.Identity()
        self.mask_classifier = nn.Linear(384, 3)
        self.gender_classifier = nn.Linear(384, 2)
        self.age_classifier = nn.Linear(384, 3)
        self.init_param()
        
    def forward(self, input):
        image_features = self.backbone(input)
        mask_out = self.mask_classifier(image_features)
        gender_out = self.gender_classifier(image_features)
        age_out = self.age_classifier(image_features)
        return mask_out, gender_out, age_out

    def init_param(self):
        for m in [self.mask_classifier, self.gender_classifier, self.age_classifier]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_num),
        nn.LeakyReLU())

# Residual block
class DarkResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(DarkResidualBlock, self).__init__()

        reduced_channels = int(in_channels/2)

        self.layer1 = conv_batch(in_channels, reduced_channels, kernel_size=1, padding=0)
        self.layer2 = conv_batch(reduced_channels, in_channels)

    def forward(self, x):
        residual = x

        out = self.layer1(x)
        out = self.layer2(out)
        out += residual
        return out

class Darknet53(nn.Module):
    def __init__(self, block = DarkResidualBlock):
        super(Darknet53, self).__init__()
        self.features = nn.Sequential(
            conv_batch(3, 32),
            conv_batch(32, 64, stride=2),
            self.make_layer(block, in_channels=64, num_blocks=1),
            conv_batch(64, 128, stride=2),
            self.make_layer(block, in_channels=128, num_blocks=2),
            conv_batch(128, 256, stride=2),
            self.make_layer(block, in_channels=256, num_blocks=8),
            conv_batch(256, 512, stride=2),
            self.make_layer(block, in_channels=512, num_blocks=8),
            conv_batch(512, 1024, stride=2),
            self.make_layer(block, in_channels=1024, num_blocks=4),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mask_classifier = nn.Linear(1024, 3)
        self.gender_classifier = nn.Linear(1024, 2)
        self.age_classifier = nn.Linear(1024, 3)
        self.init_param()

    def forward(self, x):
        out = self.features(x)
        out = self.global_avg_pool(out)
        image_features = out.view(-1, 1024)
        mask_out = self.mask_classifier(image_features)
        gender_out = self.gender_classifier(image_features)
        age_out = self.age_classifier(image_features)
        return mask_out, gender_out, age_out

    def make_layer(self, block, in_channels, num_blocks):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)

    def init_param(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)