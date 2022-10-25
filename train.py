from utils.json_parser import parse_json # json package import
from importlib import import_module # dynamically create instance
from utils.slack import *

import torch
from torch.utils.data import DataLoader
from torchsummary import summary
import numpy as np
import random
import os

# config parsing
config = parse_json("config.json")

# import modules
dataset_module = import_module('dataloader.dataset')
augmentation_module = import_module('dataloader.augmentation')
model_module = import_module('model.model')
loss_module = import_module('model.loss')
optimizer_module = import_module('model.optimizer')

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# set seed
SEED = config.get('training')['seed']
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)  # type: ignore
torch.backends.cudnn.deterministic = True  # type: ignore
torch.backends.cudnn.benchmark = True  # type: ignore

# create dataset
TrainDataset = getattr(dataset_module, config.get('traindataset').get('name'))
train_transform = getattr(augmentation_module, config.get('traindataset').get('args')['transform'])
train_dataset = TrainDataset(
    root = config.get('traindataset').get('args')['root'],
    transform = train_transform
)

valid_dataset, train_dataset = torch.utils.data.random_split(train_dataset,
                                                           [int(len(train_dataset)*config.get('traindataset').get('args')['val_ratio']), 
                                                           len(train_dataset) - int(len(train_dataset)*config.get('traindataset').get('args')['val_ratio'])],
                                                           generator=torch.Generator().manual_seed(SEED))

dataset = {
    'train': train_dataset,
    'valid': valid_dataset
}

# create dataloader
train_dataloader_config = config.get('traindataloader')
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

valid_dataloader = DataLoader(
    dataset = dataset['valid'],
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

# create model
Architecture = getattr(model_module, config.get('architecture').get('name'))
model = Architecture().to(device)
if config.get('architecture').get('args')['show_summary']:
    img, label = train_dataset[0]
    summary(model, img.shape)

# define loss
loss_function = getattr(loss_module, config.get('loss').get('name'))
loss_fn = loss_function()

# define optimizer
optimizer_function = getattr(optimizer_module, config.get('optimizer').get('name'))
optimizer = optimizer_function(model.parameters())

# slack instance (for sending message)
if config.get('slack').get('enable'):
    slack = SlackAPI(config.get('slack').get('token'))
    channel_id = slack.get_channel_id(config.get('slack').get('channel_name'))
    message_ts = slack.get_message_ts(channel_id, config.get('slack').get('query'))

    init_message = get_init_message(
        device,
        train_dataloader_config['batch_size'],
        config.get('traindataset').get('args')['val_ratio'],
        train_dataloader_config['num_workers'],
        model,
        optimizer,
        loss_fn,
        config.get('training')['epochs'])
    slack.post_thread_message(channel_id, message_ts, init_message)

