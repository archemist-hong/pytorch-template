from utils.json_parser import parse_json # json package import
from importlib import import_module # dynamically create instance
from dataloader.augmentation import transform

# config parsing
config = parse_json("config.json")

# create dataset
dataset_module = import_module('dataloader.dataset')
TrainDataset = getattr(dataset_module, config.get('traindataset').get('name'))
train_dataset = TrainDataset(
    root = config.get('traindataset').get('args')['root'],
    transform = transform if config.get('traindataset').get('args')['transform'] else None
)
print(len(train_dataset))
print(train_dataset[1])

