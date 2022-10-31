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
    def __init__(self, alpha = 0.25, gamma = 2, label_smoothing = 0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    def forward(self, input, target):
        log_pt = -F.cross_entropy(input, target, label_smoothing = self.label_smoothing)
        modulating_factor = (1-log_pt.exp()).pow(self.gamma)
        return -self.alpha * modulating_factor * log_pt     

class F1_Loss(nn.Module):
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    - http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/
    '''
    def __init__(self, dim, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
        self.dim = dim
        
    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, self.dim).float()
        y_pred = F.softmax(y_pred, dim=1)
        
        tp = (y_true * y_pred).sum(dim=0).float()
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).float()
        fp = ((1 - y_true) * y_pred).sum(dim=0).float()
        fn = (y_true * (1 - y_pred)).sum(dim=0).float()

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2* (precision*recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        return 1 - f1.mean() 