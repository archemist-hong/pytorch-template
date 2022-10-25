from utils.json_parser import parse_json # json package import
from importlib import import_module # dynamically create instance
from torch.utils.data import DataLoader

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

dataset = {
    'train': train_dataset,
    'test': test_dataset
}

# create dataloader
train_dataloader_config = config.get('traindataloader').get('args')
train_dataloader = DataLoader(
    dataset = dataset['train'],
    batch_size = train_dataloader_config['batch_size'],
    shuffle = train_dataloader_config['shuffle'],
    sampler = train_dataloader_config['sampler'],
    batch_sampler = train_dataloader_config['batch_sampler'],
    num_workers = train_dataloader_config['num_workers'],
    collate_fn = train_dataloader_config['collate_fn'],
    pin_memory = train_dataloader_config['pin_memory'],
    drop_last = train_dataloader_config['drop_last'],
    timeout = train_dataloader_config['timeout'],
    worker_init_fn = train_dataloader_config['worker_init_fn'],
    prefetch_factor = train_dataloader_config['prefetch_factor'],
    persistent_workers = train_dataloader_config['persistent_workers']
    )

test_dataloader_config = config.get('testdataloader').get('args')
test_dataloader = DataLoader(
    dataset = dataset['test'],
    batch_size = test_dataloader_config['batch_size'],
    shuffle = test_dataloader_config['shuffle'],
    sampler = test_dataloader_config['sampler'],
    batch_sampler = test_dataloader_config['batch_sampler'],
    num_workers = test_dataloader_config['num_workers'],
    collate_fn = test_dataloader_config['collate_fn'],
    pin_memory = test_dataloader_config['pin_memory'],
    drop_last = test_dataloader_config['drop_last'],
    timeout = test_dataloader_config['timeout'],
    worker_init_fn = test_dataloader_config['worker_init_fn'],
    prefetch_factor = test_dataloader_config['prefetch_factor'],
    persistent_workers = test_dataloader_config['persistent_workers']
    )

