import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from collections import OrderedDict
from viztools.util import (
    defuse_model,
    assert_numpy_image,
    insert_flatten_layer,
    normalize_image,
    convert_image_tensor
)


def viz_act_val(image, model, layer, kernel):
    """
    Record activation values of each layer and return them as OrderedDict

    Args:
        image            : (np.ndarray) input image
        model            : (nn.Module) PyTorch pretrained model
        layer            : (str) target layer
        kernel           : (int) target kernel
    Returns:
        activation_value : (np.ndarray) activation value in array
    """
    assert_numpy_image(image)
    assert isinstance(model, nn.Module), "'module' is not nn.Module"
    layers = defuse_model(model)
    assert layer in layers.keys(), f"'{layer}' not in 'model'"
    # insert Flatten layer if needed
    if "Linear-1" in layers.keys(): layers = insert_flatten_layer(layers)
    # preprocess image
    image = convert_image_tensor(normalize_image(image))

    if torch.cuda.is_available():
        for tag in layers.keys(): layers[tag] = layers[tag].cuda()
        image = image.cuda()
    
    # activation_values = OrderedDict()
    # activation_values["input"] = [
    #     image[0, c].data.numpy() for c in range(image.size()[1])
    # ]
    for tag, module in layers.items():
        image = module(image)
        if tag == layer:
            assert kernel < image.size()[1], "'kernel' out of bound"
            activation = image[0, kernel].cpu().numpy()
            activation -= activation.min()
            activation /= activation.max()
            return activation

        # activation_values[tag] = []
        # for channel in range(image.size()[1]):
        #     img = image[0, channel].data.numpy()
        #     img -= img.min()
        #     img /= img.max()
        #     activation_values[tag].append(img)

    # return activation_values


