from utils.json_parser import parse_json # json package import
from importlib import import_module # dynamically create instance

# config parsing
config = parse_json("config.json")

# import modules
dataset_module = import_module('dataloader.dataset')
augmentation_module = import_module('dataloader.augmentation')

# create dataset
TrainDataset = getattr(dataset_module, config.get('traindataset').get('name'))
train_transform = getattr(augmentation_module, config.get('traindataset').get('args')['transform'])
train_dataset = TrainDataset(
    root = config.get('traindataset').get('args')['root'],
    transform = train_transform
)

TestDataset = getattr(dataset_module, config.get('testdataset').get('name'))
test_transform = getattr(augmentation_module, config.get('testdataset').get('args')['transform'])
test_dataset = TestDataset(
    root = config.get('testdataset').get('args')['root'],
    transform = test_transform
)

# create dataloader


