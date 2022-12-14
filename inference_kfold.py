from utils.json_parser import parse_json # json package import
from utils.utils import *
from importlib import import_module # dynamically create instance
import torch
from torch.utils.data import DataLoader
from torchsummary import summary
import numpy as np
import pandas as pd
import random
import os
from tqdm import tqdm

model_list = [
    "checkpoint_model_24_0.325_0.768.pt",
    "checkpoint_model_25_0.287_0.742.pt",
    "checkpoint_model_28_0.004_0.725.pt",
    "checkpoint_model_28_0.403_0.733.pt",
    "checkpoint_model_10_0.005_0.966.pt"
]

# config parsing
print('Parsing from config.json ...')
config = parse_json("config.json")

# import modules
print('Importing modules ...')
dataset_module = import_module('dataloader.dataset')
augmentation_module = import_module('dataloader.augmentation')
model_module = import_module('model.model')

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"
print(f'Setting device {device} ...')

# set seed
SEED = config.get('inference')['seed']
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)  # type: ignore
torch.backends.cudnn.deterministic = True  # type: ignore
torch.backends.cudnn.benchmark = True  # type: ignore
print(f'Setting seed ({SEED}) ...')

# create dataset
print(f'Making Dataset ...')
TestDataset = getattr(dataset_module, config.get('testdataset').get('name'))
if config.get('testdataset').get('args')['test_time_augmentation']:
    test_transform = getattr(augmentation_module, config.get('traindataset').get('args')['transform'])
else:
    test_transform = getattr(augmentation_module, config.get('testdataset').get('args')['transform'])
test_dataset = TestDataset(
    root = config.get('testdataset').get('args')['root'],
    transform = test_transform
)

# create dataloader
print('Making DataLoader ...')
test_dataloader_config = config.get('testdataloader')
test_dataloader = DataLoader(
    dataset = test_dataset,
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

softmax = torch.nn.Softmax(dim = 1)
for k in range(len(model_list)):
    print(f"\n fold: {k} \n")
    mask_prob_bag = []
    gender_prob_bag = []
    age_prob_bag = []
    
    # load model
    print('Loading Model ...')
    Architecture = getattr(model_module, config.get('architecture').get('name'))
    model = Architecture().to(device)
    if config.get('architecture').get('args')['show_summary']:
        img, label = test_dataset[0]
        summary(model, img.shape)
    check_point = torch.load(os.path.join(config.get('training')['experiment_path']+f"-{k}", 'checkpoint', model_list[k]))
    model.load_state_dict(check_point['model_state_dict'])

    # test
    print('Start Testing ...')
    model.eval()
    mask_preds_list = []
    gender_preds_list = []
    age_preds_list = []

    with tqdm(test_dataloader, unit="batch") as tepoch:
        for ind, images in enumerate(tepoch):
            images = images.to(device)

            with torch.set_grad_enabled(False): # ????????? ?????????
                mask_logits, gender_logits, age_logits = model(images)

                mask_probs = softmax(mask_logits)
                gender_probs = softmax(gender_logits)
                age_probs = softmax(age_logits)

            mask_preds_list.append(mask_probs)
            gender_preds_list.append(gender_probs)
            age_preds_list.append(age_probs)

    # ??? epoch??? ?????? ??????????????? ???,
    epoch_mask_pred = torch.cat(mask_preds_list)
    epoch_gender_pred = torch.cat(gender_preds_list)
    epoch_age_pred = torch.cat(age_preds_list)
    
    # ??? ????????? ????????? ????????????
    mask_prob_bag.append(epoch_mask_pred)
    gender_prob_bag.append(epoch_gender_pred)
    age_prob_bag.append(epoch_age_pred)

# ?????? ?????? ?????? ???
mask_soft_voting = sum(mask_prob_bag).div(len(model_list))
gender_soft_voting = sum(gender_prob_bag).div(len(model_list))
age_soft_voting = sum(age_prob_bag).div(len(model_list))

epoch_mask_pred = torch.argmax(mask_soft_voting, dim = 1)
epoch_gender_pred = torch.argmax(gender_soft_voting, dim = 1)
epoch_age_pred = torch.argmax(age_soft_voting, dim = 1)
epoch_final_pred = epoch_mask_pred*6 + epoch_gender_pred*3 + epoch_age_pred

submission = test_dataset.valid_info
submission['ans'] = epoch_final_pred.cpu().numpy()
check_directories(config.get('training')['experiment_path'], config_check = False)
submission.to_csv(os.path.join(config.get('training')['experiment_path'], 'submission.csv'), index=False)