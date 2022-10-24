import torchvision.transforms as transforms

# variable name must be "transform"
transform = transforms.Compose([
    transforms.ToTensor()
])