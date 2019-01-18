import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from viztools.util import normalize_image, convert_image_tensor


def gen_class_image(model,
                    target_class=0,
                    img_dim=[224, 224, 3],
                    nb_epoch=100,
                    learning_rate=3,
                    regularization="l2",
                    reg_coeff=0.01,
                    verbose=False):
    """
    Optimize a random image into image of specific class

    Args:
        model          : (nn.Module) PyTorch pretrained model
        target_class   : (int) class to generate
        img_dim        : (list) input dimension of model in H x W x C
        nb_epoch       : (int) number of epochs
        learning_rate  : (float) control learning speed
        regularization : (str) regularizer to use l1, l2, or none
        verbose        : (float) regularization coefficient
    Returns:
        image          : (np.ndarray) optimized image on specific class
    """
    assert isinstance(model, nn.Module), "'module' is not nn.Module"
    assert regularization in ["none", "l1", "l2"], "unknown 'regularization'"
    # generate image
    image = np.uint8(np.zeros(img_dim))
    image = convert_image_tensor(normalize_image(image))

    assert target_class < model(image).size()[1], "'class' out of range"

    if torch.cuda.is_available():
        image = image.cuda()
        model = model.cuda()

    # set up lr scheduler
    optimizer = optim.SGD([image], lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.8,
        patience=3,
        min_lr=0.001
    )
    # set up regularization function
    R = {
        "none": lambda x: 0,
        "l1"  : lambda x: x.abs().sum(),
        "l2"  : lambda x: x.pow(2).sum(),
    }[regularization]

    if verbose: plt.ion()
    for i in range(nb_epoch):
        optimizer.zero_grad()
        # compute loss
        loss = - model(image)[0, target_class] + reg_coeff * R(image)
        scheduler.step(loss)
        loss.backward()
        optimizer.step()
        # plot image
        if verbose:
            plt.figure(0), plt.clf()
            plt.title(" - ".join([
                f"epoch: {i}",
                f"loss: {loss.data.numpy():.4f}",
                f"lr: {optimizer.param_groups[0]['lr']:.3f}"
            ]))
            plt.imshow(
                normalize_image(convert_image_tensor(image, True), True)
            )
            plt.axis("off"), plt.pause(0.01), plt.show()
    if verbose: plt.ioff()

    return normalize_image(convert_image_tensor(image, True), True)