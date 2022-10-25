from utils.json_parser import parse_json # json package import
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