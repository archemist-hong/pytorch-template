from utils.json_parser import parse_json # json package import
from importlib import import_module # dynamically create instance
from utils.slack import *

import torch
from torch.utils.data import DataLoader
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import os
from tqdm import tqdm

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

# create dataloader
train_dataloader_config = config.get('traindataloader')
train_dataloader = DataLoader(
    dataset = train_dataset,
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
    dataset = valid_dataset,
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

dataloaders = {
    'train': train_dataloader,
    'valid': valid_dataloader
}

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
slack_enable = config.get('slack').get('enable')
if slack_enable:
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

# tensor board instance (for logging)
if config.get('tensorboard')['enable']:
    writer = SummaryWriter(config.get('tensorboard')['run_dir'])

# train

for epoch in range(int(config.get('training')['epochs'])):
    for phase in ["train", "valid"]:
        running_loss = 0.
        running_mask_acc = 0.
        running_gender_acc = 0.
        running_age_acc = 0.
        if phase == "train":
            model.train() 
        elif phase == "valid":
            model.eval()

        with tqdm(dataloaders[phase], unit="batch") as tepoch:
            for ind, (images, labels) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad() # initialize optimizer
                with torch.set_grad_enabled(phase == "train"): # 연산량 최소화
                    mask_logits, gender_logits, age_logits = model(images)
                    _, mask_preds = torch.max(mask_logits, 1)
                    _, gender_preds = torch.max(gender_logits, 1)
                    _, age_preds = torch.max(age_logits, 1) 
                    mask_labels, gender_labels, age_labels = labels[:, :3].float(), labels[:, 3:5].float(), labels[:, 5:].float()
                    loss = loss_fn(mask_logits, mask_labels) + loss_fn(gender_logits, gender_labels) + loss_fn(age_logits, age_labels)

                    if phase == "train":
                        loss.backward() # gradient 계산
                        optimizer.step() # 모델 업데이트

                running_loss += loss.item() * images.size(0) # 한 Batch에서의 loss 값 누적
                running_mask_acc += torch.sum(mask_preds == torch.argmax(mask_labels, 1).data) # 한 Batch에서의 Accuracy 값 누적
                running_gender_acc += torch.sum(gender_preds == torch.argmax(gender_labels, 1).data) # 한 Batch에서의 Accuracy 값 누적
                running_age_acc += torch.sum(age_preds == torch.argmax(age_labels, 1).data) # 한 Batch에서의 Accuracy 값 누적

                tepoch.set_postfix(
                    loss=loss.item(),
                    mask_accuracy=(torch.sum(mask_preds == torch.argmax(mask_labels, 1).data)/train_dataloader_config['batch_size']).item(),
                    gender_accuracy=(torch.sum(gender_preds == torch.argmax(gender_labels, 1).data)/train_dataloader_config['batch_size']).item(),
                    age_accuracy=(torch.sum(age_preds == torch.argmax(age_labels, 1).data)/train_dataloader_config['batch_size']).item()
                    )
        # 한 epoch이 모두 종료되었을 때,
        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_mask_acc = running_mask_acc / len(dataloaders[phase].dataset)
        epoch_gender_acc = running_gender_acc / len(dataloaders[phase].dataset)
        epoch_age_acc = running_age_acc / len(dataloaders[phase].dataset)

        # ...log the running loss
        if phase == 'valid':
            writer.add_scalar('mean loss',
                            epoch_loss,
                            epoch)
            # ...log the accuracy
            writer.add_scalar('mean accuracy(mask)',
                            epoch_mask_acc,
                            epoch)
            writer.add_scalar('mean accuracy(gender)',
                            epoch_gender_acc,
                            epoch)
            writer.add_scalar('mean accuracy(age)',
                            epoch_age_acc,
                            epoch)
        if slack_enable:
            epoch_message = get_epoch_message(phase, epoch, epoch_loss, epoch_mask_acc, epoch_gender_acc, epoch_age_acc)
            slack.post_thread_message(channel_id, message_ts, epoch_message)

#if slack_enable:
    #final_message = get_final_message()
    #slack.post_thread_message(channel_id, message_ts, final_message)

if slack_enable:
    error_message = f"`오류` 오류가 발생했습니다!: {e}"
    slack.post_thread_message(channel_id, message_ts, error_message)