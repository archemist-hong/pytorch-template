from pathlib import Path
import pandas as pd
import PIL
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, root, transform):
        self.transform = transform
        train_path = Path(root)
        train_info = pd.read_csv(train_path.parent/'train.csv')
        self.paths = [f for path in train_info['path'] for f in (train_path / path).iterdir() if f.stem[0] != '.'] # get paths of image files except hidden files

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = PIL.Image.open(self.paths[index])
        labels = self.get_labels(self.paths[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, labels

    def get_labels(self, path):
        id_, gender, race, age = path.parent.stem.split('_')
        return gender, race, age