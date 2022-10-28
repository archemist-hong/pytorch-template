import torchvision.transforms as transforms

train_transform = transforms.Compose([
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.ToTensor()
])

vit_train_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor()
])

ColorJitter =  transforms.Compose([
    transforms.ColorJitter(brightness=.5, hue=.3),
    transforms.ToTensor()
])

RandomAffine = transforms.Compose([
    transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
    transforms.ToTensor()
])

RandomHorizontalFlip = transforms.Compose([
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.ToTensor()
])

RandomRotation = transforms.Compose([
    transforms.RandomRotation(degrees=(0, 180)),
    transforms.ToTensor()
])

RandomVerticalFlip = transforms.Compose([
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor()
])

GaussianBlur = transforms.Compose([
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.ToTensor()
])

RandomPerspective = transforms.Compose([
    transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
    transforms.ToTensor()
])

RandomAdjustSharpness = transforms.Compose([
    transforms.RandomAdjustSharpness(sharpness_factor=2),
    transforms.ToTensor()
])

RandomAutocontrast = transforms.Compose([
    transforms.RandomAutocontrast(),
    transforms.ToTensor()
])