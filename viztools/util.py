import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict, Iterable
from torchvision import transforms

__all__ = [
    "defuse_model",
    "normalize_image",
    "convert_image_tensor",
    "combine_images",
    "assert_numpy_image",
    "insert_flatten_layer"
]


def defuse_model(model):
    """
    Flatten Model and returned an OrderedDict of (layer name, module) pairs

    Args:
        model      : (torch.nn.Module) pytorch model
    Returns:
        layer_list : (collections.OrderedDict{str: nn.Module}) tag, module pair
    """
    assert isinstance(model, nn.Module), "'model' is not nn.Module"
    
    layers = OrderedDict()
    counter = dict(set([(type(x), 0) for x in model.modules()]))

    def func(module):
        children = list(module.children())
        if not len(children):
            counter[type(module)] += 1
            layers[f"{module._get_name()}-{counter[type(module)]}"] = module
        for child in children: func(child)
    func(model)

    return layers


def normalize_image(image, reverse=False):
    """
    Normalize and image using predefined mean and standard deviation

    Args:
        image   : (np.ndarray) RGB image in H x W x C shape
        reverse : (bool) switch for normalization and denormalization
    Returns:
        image   : (np.ndarray) normalized image in same shape
    """
    assert_numpy_image(image)
    # normalize image
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if not reverse:
        image = np.float32(image) / 255
        image = (image - mean) / std
    else:
        image = image * std + mean
        image = np.clip(image, 0, 1)
        image = np.uint8(image * 255)

    return image


def convert_image_tensor(image, reverse=False):
    """
    Convert from numpy image array to PyTorch Tensor

    Args:
        image   : (np.ndarray/torch.Tensor) input image
        reverse : (bool) switch for backwrad conversion
    Returns:
        image   : (np.ndarray/torch.Tensor) processed image
    """
    if not reverse:
        assert_numpy_image(image)
        # H x W x C -> C x H x W
        image = image.transpose(2, 0, 1)
        # C x H x W -> 1 x C x H x W
        image = torch.FloatTensor(image).unsqueeze(0).requires_grad_(True)
    else:
        assert type(image) == torch.Tensor, "'image' is not torch.Tensor"
        # 1 x C x H x W -> C x H x W
        image = image.squeeze(0).detach().cpu().numpy()
        # C x H x W -> H x W x C
        image = image.transpose(1, 2, 0)

    return image


def combine_images(images, border_size=5):
    """
    Arrange small images into big square images, pad extra spaces with white
    images and add border to each image

    Args:
        images: (list) list of images to arrange
    Returns:
        image: (np.ndarray) image matrix in RGB or gray
    """
    assert isinstance(images, Iterable), "'images' is non-iterable"
    if type(images) != list: images = list(images)
    n = math.ceil(math.sqrt(len(images)))
    # arrage images into matrix
    img_mat = [images[i * n:(i + 1) * n] for i in range(n + 1)]
    # white image for padding
    white = np.ones(images[0].shape, dtype=images[0].dtype)
    if len(images[0].shape) == 3: white *= 255

    def add_border(img):
        assert_numpy_image(img)
        # for gray image
        if len(img.shape) == 2:
            w, h = img.shape
            new_w = w + 2 * border_size
            new_h = h + 2 * border_size
            r_img = np.zeros([new_w, new_h], dtype=images[0].dtype)
            r_img[border_size:-border_size, border_size:-border_size] = img
        # for RGB image
        else:
            w, h, c = img.shape
            new_w = w + 2 * border_size
            new_h = h + 2 * border_size
            r_img = np.zeros([new_w, new_h, c], dtype=images[0].dtype)
            r_img[border_size:-border_size, border_size:-border_size, :] = img
        return r_img

    # merge images horizontally
    rows = []
    for r_img in img_mat:
        if len(r_img) < n:
            if len(r_img) == 0:
                continue
            r_img += [white for _ in range(n - len(r_img))]
        rows.append(np.hstack(list(map(add_border, r_img))))

    # merge all rows vertically
    image = np.vstack(rows)
    # add border to image
    image = add_border(image)

    return image


def assert_numpy_image(image):
    """
    Check whether image is under following conditions:
        - image is a numpy array type
        - image is in 2-dimension or 3-dimension
        - image has shape H x W x C or H x W
        - image has 3 channels or gray
    """
    assert type(image) == np.ndarray, "'image' is not np.ndarray"
    assert len(image.shape) in [2, 3], "'image' is not in 2D or 3D"
    if len(image.shape) == 3:
        assert image.shape[2] == 3, "'image' is not 3 channels"


def insert_flatten_layer(layers):
    """
    Insert Flatten Layer before first Linear layer to complete the pipeline

    Args:
        layers: (OrderedDict) initial layers dictionary
    Returns:
        layers: (OrderedDict) complete layers dictionary
    """
    assert type(layers) == OrderedDict, "'layers' is not OrderedDict"
    # locate first linear layer
    index = list(layers.keys()).index("Linear-1")
    # split into left and right
    left = list(layers.keys())[:index], list(layers.values())[:index]
    right = list(layers.keys())[index:], list(layers.values())[index:]
    # make key, value pair
    left = list(zip(*left))
    right = list(zip(*right))
    # Flatten layer
    class Flatten(nn.Module):
        def forward(self, x): return x.view(x.size()[0], -1)
    # insert Flatten layer
    layers = OrderedDict(left + [("Flatten-1", Flatten())] + right)

    return layers
