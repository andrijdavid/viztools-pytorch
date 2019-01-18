import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from viztools.util import (
    defuse_model,
    insert_flatten_layer,
    normalize_image,
    convert_image_tensor
)


def viz_cnn_filter(model,
                   layer,
                   kernel,
                   img_dim=[224, 224, 3],
                   nb_epoch=32,
                   learning_rate=0.1,
                   weight_decay=1e-6,
                   regularization="none",
                   reg_coeff=0.001):
    """
    Optimize image to visualize particular filter in Convolutional Neural
    Network

    Args:
        model          : (nn.Module) PyTorch pretrained model
        layer          : (str) target layer
        kernel         : (int) target kernel
        img_dim        : (list) input dimension of model in H x W x C
        nb_epoch       : (int) number of epochs
        learning_rate  : (float) control learning speed
        weight_decay   : (float) control weight decay
        regularization : (str) regularizer to use l1, l2, or none
        reg_coeff      : (float) regularization coefficient

    Returns:
        image         : (np.ndarray) optimized image on specific filter
    """
    assert isinstance(model, nn.Module), "'module' is not nn.Module"
    assert regularization in ["none", "l1", "l2"], "unknown 'regularization'"
    layers = defuse_model(model)
    # rebuild model up to target layer
    try:
        index = list(layers.keys()).index(layer)
        model = nn.Sequential(*list(layers.values())[:index + 1])
        model.eval()
    except:
        raise Exception(f"'{layer}' not found in 'model'")
    # insert Flatten layer if needed
    if "Linear-1" in layers.keys(): layers = insert_flatten_layer(layers)

    # generate random image
    image = np.uint8(np.random.uniform(115, 140, img_dim))
    # preprocess image
    image = convert_image_tensor(normalize_image(image))

    if torch.cuda.is_available():
        image = image.cuda()
        model = model.cuda()

    assert kernel < model(image).size()[1], "'kernel' out of bound"

    optimizer = optim.Adam(
        [image],
        lr=learning_rate,
        weight_decay=weight_decay
    )
    R = {
        "none" : lambda x: 0,
        "l1"   : lambda x: x.abs().sum(),
        "l2"   : lambda x: x.pow(2).sum()
    }[regularization]
    
    for _ in range(nb_epoch):
        optimizer.zero_grad()
        loss = - model(image)[0, kernel].mean() + reg_coeff * R(image)
        loss.backward()
        optimizer.step()

    # reverse preprocessing
    image = convert_image_tensor(image, reverse=True)
    image = normalize_image(image, reverse=True)

    return image
