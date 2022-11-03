from torchmetrics import F1Score

def get_f1score(num_classes):
    return F1Score(num_classes, average="macro")