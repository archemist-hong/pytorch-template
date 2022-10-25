from utils.json_parser import parse_json # json package import

# config parsing
config = parse_json("config.json")

# import modules
dataset_module = import_module('dataloader.dataset')
augmentation_module = import_module('dataloader.augmentation')
model_module = import_module('model.model')
loss_module = import_module('model.loss')
optimizer_module = import_module('model.optimizer')

TestDataset = getattr(dataset_module, config.get('testdataset').get('name'))
test_transform = getattr(augmentation_module, config.get('testdataset').get('args')['transform'])
test_dataset = TestDataset(
    root = config.get('testdataset').get('args')['root'],
    transform = test_transform
)

test_dataloader_config = config.get('testdataloader')
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