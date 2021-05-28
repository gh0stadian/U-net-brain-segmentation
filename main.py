from train import Train
from dataloader import *
from model.unet import UNet
import yaml

stream = open("train_config.yaml", 'r')
config = yaml.load(stream)

print("CONFIG" + "-"*14)
print("DATAROOT: " + str(config['dataset_root_directory']))
print("ROTATION: " + str(config['rotation_limit_in_degrees']))
print("GAMMA VALUE: " + str(config['gamma_correction_type']))
print("-"*20)

#   TRAIN DATASET
train_dataset = MRIDataset(root_dir=config['dataset_root_directory'] + 'train/',
                           rotation_transform_degrees=config['rotation_limit_in_degrees'],
                           gamma_value=config['gamma_correction_type'])
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

#   VALID DATASET
valid_dataset = MRIDataset(root_dir=config['dataset_root_directory'] + 'valid/',
                           rotation_transform_degrees=config['rotation_limit_in_degrees'],
                           gamma_value=config['gamma_correction_type'])
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

#   UNET
unet = UNet(1, 1, False)

try:
    unet.cuda()
except:
    print("U-NET TO CUDA -> FATAL ERROR")
    exit(1)
else:
    print("U-NET TO CUDA -> PASS")

# BEGIN TRAINING
Train(n_of_epoch=150, train_loader=train_loader, valid_loader=valid_loader, model=unet, config=config)
