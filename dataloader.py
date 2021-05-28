import os
from random import randrange, uniform
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from rotate_image import rotate
import numpy as np


class MRIDataset(Dataset):
    def __init__(self, root_dir, rotation_transform_degrees=0, gamma_value=0):
        self.root_dir = root_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.names = os.listdir(root_dir + "input")
        self.rotation_transform_degrees = rotation_transform_degrees
        self.gamma_value = gamma_value

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        i_file, o_file = self.load_sitk(idx)
        sample = {'input': torch.from_numpy(i_file).to(self.device),
                  'output': torch.from_numpy(o_file).to(self.device),
                  'name': str(self.names[idx])}
        return sample

    def load_sitk(self, idx):
        # GET PATH
        input_image_path = os.path.join(self.root_dir + "input/", self.names[idx])
        output_image_path = os.path.join(self.root_dir + "output/", self.names[idx])

        # OPEN IMAGES
        input_image = sitk.ReadImage(input_image_path)
        output_image = sitk.ReadImage(output_image_path)

        # RANDOM ROTATE
        if self.rotation_transform_degrees != 0:
            input_image, output_image = random_rotate(input_image, output_image, self.rotation_transform_degrees)

        # CONVERT TO NUMPY
        input_image = sitk.GetArrayFromImage(input_image)
        output_image = sitk.GetArrayFromImage(output_image)

        # CONVERT TO FLOAT32
        input_image = input_image.astype(np.float32)
        output_image = output_image.astype(np.float32)

        # FILTER HIPPOCAMPUS
        output_image = highlight_hippocampus(output_image)

        # GAMMA CORRECTION
        if self.gamma_value != 0:
            input_image = data_normalizer_min_max(input_image)
            input_image = gamma_correction(input_image, self.gamma_value)

        # NORMALIZE INPUT
        input_image = data_normalizer(input_image)

        # EXPAND DIMENSIONS
        input_image = np.expand_dims(input_image, axis=0)
        output_image = np.expand_dims(output_image, axis=0)

        return input_image, output_image


def data_normalizer(data):
    return (data - np.mean(data)) / np.std(data)


def data_normalizer_min_max(data):
    return (data - np.min(data)) / np.ptp(data)


def highlight_hippocampus(data):
    data = np.logical_or(data == 17, data == 53).astype(np.float32)
    return data


def random_rotate(input_file, output_file, max_rotate_range):
    x_axis_degrees = uniform(-max_rotate_range, max_rotate_range)
    y_axis_degrees = uniform(-max_rotate_range, max_rotate_range)
    z_axis_degrees = uniform(-max_rotate_range, max_rotate_range)

    rotated_input = rotate(input_file, x_axis_degrees, y_axis_degrees, z_axis_degrees, "input")
    rotated_output = rotate(output_file, x_axis_degrees, y_axis_degrees, z_axis_degrees, "output")

    return rotated_input, rotated_output


def gamma_correction(data, gamma_type):
    gamma_value = 1
    if gamma_type == 1:
        gamma_value = uniform(0.9, 1.1)
    elif gamma_type == 2:
        gamma_value = uniform(0.8, 1.2)
    elif gamma_type == 3:
        gamma_value = uniform(0.7, 1.3)
    return np.array(data ** gamma_value, dtype='float32')
