import numpy as np
import SimpleITK as sitk


def rotate(image, x_degrees, y_degrees, z_degrees, image_type="input"):
    x_rad = np.deg2rad(x_degrees)
    y_rad = np.deg2rad(y_degrees)
    z_rad = np.deg2rad(z_degrees)
    image_center = get_center(image)
    euler_transform = sitk.Euler3DTransform(image_center, x_rad, y_rad, z_rad, (0, 0, 0))
    euler_transform.SetCenter(image_center)
    euler_transform.SetRotation(x_rad, y_rad, z_rad)
    if image_type == "input":
        resampled_image = resample(image, euler_transform)
        return resampled_image
    else:
        resampled_image = resample_label(image, euler_transform)
        return resampled_image


def get_center(image):
    width, height, depth = image.GetSize()
    return image.TransformIndexToPhysicalPoint((int(np.ceil(width / 2)),
                                                int(np.ceil(height / 2)),
                                                int(np.ceil(depth / 2))))


def resample(image, transform):
    reference_image = image
    interpolator = sitk.sitkLinear
    default_value = 0
    return sitk.Resample(image, reference_image, transform,
                         interpolator, default_value)


def resample_label(image, transform):
    reference_image = image
    interpolator = sitk.sitkNearestNeighbor
    default_value = 0
    return sitk.Resample(image, reference_image, transform,
                         interpolator, default_value)