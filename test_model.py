import torch
from dataloader import *
from monai.losses.dice import DiceLoss
import yaml
import SimpleITK as sitk

from model.unet import UNet
from reload_staged_model import load_checkpoint_for_test


def calculate_volume_diff(input_image, output_image):
    return 2 * (np.sum(input_image) - np.sum(output_image)) / (np.sum(output_image) + np.sum(input_image))


def calculate_iou(input_image, output_image):
    intersection = np.logical_and(output_image, input_image)
    union = np.logical_or(output_image, input_image)
    return np.sum(intersection) / np.sum(union)


def calculate_dice(input_image, output_image):
    intersection = np.logical_and(output_image, input_image)
    return 2 * np.sum(intersection) / (np.sum(output_image) + np.sum(input_image))


def clamp_data(data):
    data = np.where(data<0.5, 0, data)
    data = np.where(data>=0.5 , 1, data)
    return data


stream = open("test_config.yaml", 'r')
config = yaml.load(stream)

model = UNet(1, 1, False)
model = load_checkpoint_for_test(model, config['model_path'])
model.eval()

loss_function = DiceLoss()

test_dataset = MRIDataset(config['dataset_root_directory'])

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

valid_iterator = iter(test_loader)

sum_of_dice = 0
sum_of_iou = 0
sum_of_volume_diff = 0

print("TESTING, PLEASE WAIT")

for batch in valid_iterator:
    model_output = model.forward(batch['input'])

    output = model_output.cpu().detach().numpy()
    output = clamp_data(output)
    target = batch['output'].cpu().detach().numpy()

    sum_of_dice += calculate_dice(output, target)
    sum_of_iou += calculate_iou(output, target)
    sum_of_volume_diff += calculate_volume_diff(output, target)

print("DICE: " + str(sum_of_dice / len(test_loader)))
print("IOU: " + str(sum_of_iou / len(test_loader)))
print("VOLUME DIFF: " + str(sum_of_volume_diff / len(test_loader)))
