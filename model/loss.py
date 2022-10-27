from utils.json_parser import parse_json # json package import
import torch.nn.functional as F
from torch import nn

# config parsing
config = parse_json("config.json")
loss_args = config.get('loss').get('args')

# define loss
def CrossEntropy():
    return nn.CrossEntropyLoss(
        weight = loss_args['weight'],
        size_average = loss_args['size_average'],
        ignore_index = loss_args['ignore_index'],
        reduce = loss_args['reduce'],
        reduction = loss_args['reduction'],
        label_smoothing = loss_args['label_smoothing']
    )

class FocalLoss(nn.Module):
    def __init__(self, alpha = 0.25, gamma = 2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, input, target):
        log_pt = -F.cross_entropy(input, target)
        modulating_factor = (1-log_pt.exp()).pow(self.gamma)
        return -self.alpha * modulating_factor * log_pt      