from pathlib import Path
import pandas as pd
import PIL
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch

class TrainDataset(Dataset):
    def __init__(self, root, transform):
        self.mask_classes = {
            "mask1": 0,
            "mask2": 0,
            "mask3": 0,
            "mask4": 0,
            "mask5": 0,
            "incorrect_mask": 1,
            "normal": 2
        } # wear:0, incorrect:1, not wear:2
        self.gender_classes = {
            "male": 0,
            "female": 1
        }
        self.age_classes = {
            "<30": 0,
            ">=30 and <60": 1,
            ">=60": 2
        }
        self.transform = transform
        train_path = Path(root) / "images"
        train_info = pd.read_csv(train_path.parent/'train.csv')
        self.paths = [f for path in train_info['path'] for f in (train_path / path).iterdir() if f.stem[0] != '.'] # get paths of image files except hidden files

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = PIL.Image.open(self.paths[index])
        labels = self.get_labels(self.paths[index])
        if self.transform:
            image = self.transform(image)
        return image, labels

    def get_labels(self, path):
        mask_one_hot = F.one_hot(torch.tensor(self.mask_classes[path.stem]), 3)
        id_, gender, race, age = path.parent.stem.split('_')
        gender_one_hot = F.one_hot(torch.tensor(self.gender_classes[gender]), len(self.gender_classes))
        age = (2 if int(age) >= 60 else (1 if int(age)>= 30 else 0)) # >=60:2, >=30:1, <30:0
        age_one_hot = F.one_hot(torch.tensor(age), len(self.age_classes))
        return mask_one_hot, gender_one_hot, age_one_hot

class TestDataset(Dataset):
    def __init__(self, root, transform):
        self.valid_path = Path(root)
        self.valid_info = pd.read_csv(self.valid_path / 'info.csv')
        self.transform = transform
    def __len__(self):
        return self.valid_info.shape[0]
    def __getitem__(self, index):
        image = PIL.Image.open(self.valid_path / 'images' / self.valid_info['ImageID'][index])
        if self.transform:
            image = self.transform(image)
        return image
