from utils.json_parser import parse_json # json package import
from importlib import import_module # dynamically create instance
from utils.slack import *
from utils.utils import *
from logger.tensorboard import *

import torch
from torch.utils.data import DataLoader
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torchmetrics import ConfusionMatrix
import numpy as np
import pandas as pd
import random
import os
from tqdm import tqdm

# config parsing
print('Parsing from config.json ...')
config = parse_json("config.json")

# import modules
print('Importing modules ...')
dataset_module = import_module('dataloader.dataset')
augmentation_module = import_module('dataloader.augmentation')
model_module = import_module('model.model')
loss_module = import_module('model.loss')
optimizer_module = import_module('model.optimizer')
metric_module = import_module('model.metrics')

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Setting device {device} ...')

# set seed
SEED = config.get('training')['seed']
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)  # type: ignore
torch.backends.cudnn.deterministic = True  # type: ignore
torch.backends.cudnn.benchmark = True  # type: ignore
print(f'Setting seed ({SEED}) ...')

# check directories
check_directories(config.get('training')['experiment_path'])

# create dataset
print(f'Making Dataset ...')
TrainDataset = getattr(dataset_module, config.get('traindataset').get('name'))
train_transform = getattr(augmentation_module, config.get('traindataset').get('args')['transform'])
dataset = TrainDataset(
    root = config.get('traindataset').get('args')['root'],
    transform = train_transform
)
## stratified split
train_indices, valid_indices, _, _ = train_test_split(
    range(len(dataset)),
    dataset.labels,
    stratify = dataset.labels,
    test_size = config.get('traindataset').get('args')['val_ratio'],
    random_state = SEED
)
train_dataset = torch.utils.data.Subset(dataset, train_indices)
valid_dataset = torch.utils.data.Subset(dataset, valid_indices)

# create dataloader
print('Making DataLoader ...')
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
print('Making Model ...')
Architecture = getattr(model_module, config.get('architecture').get('name'))
model = Architecture().to(device)
if config.get('architecture').get('args')['show_summary']:
    img, label = train_dataset[0]
    summary(model, img.shape)

# define loss
print('Defining Loss ...')
loss_function = getattr(loss_module, config.get('loss').get('name'))
loss_fn = loss_function()

# define optimizer
print('Defining Optimizer ...')
optimizer_function = getattr(optimizer_module, config.get('optimizer').get('name'))
optimizer = optimizer_function(model.parameters())

# define metrics
print('Defining Metrics ...')
metric_function = getattr(metric_module, config.get('metrics').get('name'))
mask_metric = metric_function(3).to(device)
gender_metric = metric_function(2).to(device)
age_metric = metric_function(3).to(device)
final_metric = metric_function(18).to(device)

# slack instance (for sending message)
slack_enable = config.get('slack').get('enable')
if slack_enable:
    print('Making Slack instance ...')
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
if config.get('tensorboard'):
    print('Making Tensorboard Writer ...')
    writer = SummaryWriter(config.get('training')['experiment_path'])

# train
scaler = torch.cuda.amp.GradScaler()
print('Start Training ...')
for epoch in range(int(config.get('training')['epochs'])):
    for phase in ["train", "valid"]:
        running_loss = 0.
        mask_preds_list = []
        mask_labels_list = []
        gender_preds_list = []
        gender_labels_list = []
        age_preds_list = []
        age_labels_list = []
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
                    # Casts operations to mixed precision
                    with torch.cuda.amp.autocast():
                        mask_logits, gender_logits, age_logits = model(images)
                        mask_labels, gender_labels, age_labels = split_labels(labels)
                        loss = loss_fn(mask_logits, F.one_hot(mask_labels, 3).float()) + \
                            loss_fn(gender_logits, F.one_hot(gender_labels, 2).float()) + \
                            loss_fn(age_logits, F.one_hot(age_labels, 3).float())
                    _, mask_preds = torch.max(mask_logits, 1)
                    _, gender_preds = torch.max(gender_logits, 1)
                    _, age_preds = torch.max(age_logits, 1) 

                    if phase == "train":
                        scaler.scale(loss).backward() # gradient 계산
                        scaler.step(optimizer) # 모델 업데이트
                        scaler.update()

                running_loss += loss.item() * images.size(0) # 한 Batch에서의 loss 값 누적
                mask_preds_list.append(mask_preds)
                mask_labels_list.append(mask_labels)
                gender_preds_list.append(gender_preds)
                gender_labels_list.append(gender_labels)
                age_preds_list.append(age_preds)
                age_labels_list.append(age_labels)
                tepoch.set_postfix(
                    loss=loss.item(),
                    )
        # 한 epoch이 모두 종료되었을 때,
        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_mask_pred = torch.cat(mask_preds_list)
        epoch_mask_label = torch.cat(mask_labels_list)
        epoch_mask_f1 = mask_metric(epoch_mask_pred, epoch_mask_label)
        epoch_gender_pred = torch.cat(gender_preds_list)
        epoch_gender_label = torch.cat(gender_labels_list)
        epoch_gender_f1 = gender_metric(epoch_gender_pred, epoch_gender_label)
        epoch_age_pred = torch.cat(age_preds_list)
        epoch_age_label = torch.cat(age_labels_list)
        epoch_age_f1 = age_metric(epoch_age_pred, epoch_age_label)
        epoch_final_pred = epoch_mask_pred*6 + epoch_gender_pred*3 + epoch_age_pred
        epoch_final_label = epoch_mask_label*6 + epoch_gender_label*3 + epoch_age_label
        epoch_final_f1 = final_metric(epoch_mask_pred, epoch_mask_label)

        if config.get('tensorboard'):
            # ...log the running loss
            writer.add_scalar('Train/loss' if phase == 'train' else 'Valid/loss',
                            epoch_loss,
                            epoch)
            # ...log the accuracy
            writer.add_scalar('Train/F1(mask)' if phase == 'train' else 'Valid/F1(mask)',
                            epoch_mask_f1,
                            epoch)
            writer.add_scalar('Train/F1(gender)' if phase == 'train' else 'Valid/F1(gender)',
                            epoch_gender_f1,
                            epoch)
            writer.add_scalar('Train/F1(age)' if phase == 'train' else 'Valid/F1(age)',
                            epoch_age_f1,
                            epoch)
            writer.add_scalar('Train/F1(total)' if phase == 'train' else 'Valid/F1(total)',
                            epoch_final_f1,
                            epoch)
        # save checkpoint
        if (phase == "train") and (epoch % config.get('checkpoint')['frequency'] == 0):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss
                },
                os.path.join(config.get('training')['experiment_path'], 'checkpoint', f"checkpoint_model_{epoch}_{epoch_loss:.3f}_{epoch_final_f1:.3f}.pt"))
        if slack_enable:
            epoch_message = get_epoch_message(phase, epoch, epoch_loss, epoch_mask_f1, epoch_gender_f1, epoch_age_f1, epoch_final_f1)
            slack.post_thread_message(channel_id, message_ts, epoch_message)

#if slack_enable:
    #final_message = get_final_message()
    #slack.post_thread_message(channel_id, message_ts, final_message)

#if slack_enable:
#    error_message = f"`오류` 오류가 발생했습니다!: {e}"
#    slack.post_thread_message(channel_id, message_ts, error_message

# test
print('Start Testing ...')
model.eval()
mask_preds_list = []
mask_labels_list = []
gender_preds_list = []
gender_labels_list = []
age_preds_list = []
age_labels_list = []
images_list = []

with tqdm(valid_dataloader, unit="batch") as tepoch:
    for ind, (images, labels) in enumerate(tepoch):
        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False): # 연산량 최소화
            mask_logits, gender_logits, age_logits = model(images)
            mask_labels, gender_labels, age_labels = split_labels(labels)
            _, mask_preds = torch.max(mask_logits, 1)
            _, gender_preds = torch.max(gender_logits, 1)
            _, age_preds = torch.max(age_logits, 1) 

        if config.get('tensorboard'):
            final_preds = mask_preds*6 + gender_preds*3 + age_preds
            wrong_idx = torch.where(final_preds != labels, True, False)
            images_list.append(images[wrong_idx])

        mask_preds_list.append(mask_preds)
        mask_labels_list.append(mask_labels)
        gender_preds_list.append(gender_preds)
        gender_labels_list.append(gender_labels)
        age_preds_list.append(age_preds)
        age_labels_list.append(age_labels)
# 한 epoch이 모두 종료되었을 때,
epoch_mask_pred = torch.cat(mask_preds_list)
epoch_mask_labels = torch.cat(mask_labels_list).to(device)
epoch_gender_pred = torch.cat(gender_preds_list)
epoch_gender_labels = torch.cat(gender_labels_list).to(device)
epoch_age_pred = torch.cat(age_preds_list)
epoch_age_labels = torch.cat(age_labels_list).to(device)

# mask data에만
epoch_final_pred = epoch_mask_pred*6 + epoch_gender_pred*3 + epoch_age_pred
epoch_final_labels = epoch_mask_labels*6 + epoch_gender_labels*3 + epoch_age_labels

wrong_idx = torch.where(epoch_final_pred != epoch_final_labels, True, False)
images_list = [image for image in images_list if image.shape[0] != 0]
if config.get('tensorboard'):
    writer.add_figure('predictions vs. actuals',
                        plot_classes_preds(torch.cat(images_list, dim = 0), epoch_final_pred[wrong_idx], epoch_final_labels[wrong_idx]),
                        epoch)

mask_cm = ConfusionMatrix(3)
gender_cm = ConfusionMatrix(2)
age_cm = ConfusionMatrix(3)
total_cm = ConfusionMatrix(18)

# Confusion matrix의 상단(가로방향: 모델), 좌측(세로방향: GT)
mask_result = mask_cm(epoch_mask_pred.cpu(), epoch_mask_labels.cpu())
gender_result = gender_cm(epoch_gender_pred.cpu(), epoch_gender_labels.cpu())
age_result = age_cm(epoch_age_pred.cpu(), epoch_age_labels.cpu())
final_result = total_cm(epoch_final_pred.cpu(), epoch_final_labels.cpu())

pd.DataFrame(mask_result.numpy()).to_csv(os.path.join(config.get('training')['experiment_path'], 'mask_cm.csv'))
pd.DataFrame(gender_result.numpy()).to_csv(os.path.join(config.get('training')['experiment_path'], 'gender_cm.csv'))
pd.DataFrame(age_result.numpy()).to_csv(os.path.join(config.get('training')['experiment_path'], 'age_cm.csv'))
pd.DataFrame(final_result.numpy()).to_csv(os.path.join(config.get('training')['experiment_path'], 'final_cm.csv'))