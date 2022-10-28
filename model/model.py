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

    def forward(self, input):
        image_features = self.backbone(input)
        mask_out = self.mask_classifier(image_features)
        gender_out = self.gender_classifier(image_features)
        age_out = self.age_classifier(image_features)
        return mask_out, gender_out, age_out

class MyVit(nn.Module):
    def __init__(self):
        super(MyVit, self).__init__()
        self.backbone = create_model("vit_small_patch16_384", pretrained=True)
        self.backbone.head = nn.Identity()
        self.mask_classifier = nn.Linear(384, 3)
        self.gender_classifier = nn.Linear(384, 2)
        self.age_classifier = nn.Linear(384, 3)
        
    def forward(self, input):
        image_features = self.backbone(input)
        mask_out = self.mask_classifier(image_features)
        gender_out = self.gender_classifier(image_features)
        age_out = self.age_classifier(image_features)
        return mask_out, gender_out, age_out