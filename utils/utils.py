from pathlib import Path
import shutil

def check_directories(experiment_path):
    path = Path(experiment_path)
    if not path.exists():
        path.mkdir()
    if not (path/'checkpoint').exists():
        (path/'checkpoint').mkdir()
    if (path/'config.json').exists():
        raise("config file exists!")
    else:
        shutil.copy(str(path.parent.parent.parent / 'config.json'), str(path / 'config.json'))

def count_parameters(model, requires_grad):
    return sum(p.numel() for p in model.parameters() if p.requires_grad == requires_grad)

# for mask classification only
def split_labels(labels):
    mask_labels = labels.div(6, rounding_mode='trunc')
    gender_labels = (labels - mask_labels*6).div(3, rounding_mode='trunc')
    age_labels = (labels - mask_labels*6 - gender_labels*3)
    return mask_labels, gender_labels, age_labels