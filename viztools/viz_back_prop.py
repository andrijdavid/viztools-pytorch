import numpy as np
import torch
import torch.nn as nn

from viztools.util import (
    assert_numpy_image, 
    defuse_model,
    insert_flatten_layer,
    convert_image_tensor,
    normalize_image
)


def viz_back_prop(image, model, layer, kernel):
    """
    Visualize the gradient backpropagated from selected layer and neuron

    Args:
        image    : (np.ndarray) input image
        model    : (nn.Module) PyTorch pretrained model
        layer    : (str) target layer
        kernel   : (int) target kernel
    Returns:
        gradient : (np.ndarray) gradient backpropagated
    """
    assert_numpy_image(image)
    assert isinstance(model, nn.Module), "'module' is not nn.Module"
    # preprocess image
    image = convert_image_tensor(normalize_image(image))
    # rebuild model up to target layer
    layers = defuse_model(model)
    assert layer in layers.keys(), f"'{layer}' not in 'model'"
    # insert Flatten layer if needed
    if "Linear-1" in layers.keys(): layers = insert_flatten_layer(layers)
    # rebuild model up to selected layer
    try:
        index = list(layers.keys()).index(layer)
        model = nn.Sequential(*list(layers.values())[:index + 1])
        model.eval()
    except:
        raise Exception(f"'{layer}' not found in 'model'")
    assert kernel < model(image).size()[1], "'kernel' out of bound"

    relu_cache = []
    # only positive gradients are backpropagated, cleaner result
    def relu_fwd(module, data_in, data_out):
        relu_cache.append(data_out)

    def relu_bwd(module, grad_in, grad_out):
        fwd_output = relu_cache.pop()
        # derivative of relu function
        fwd_output = (fwd_output > 0).float()
        # only backprop positive gradients
        grad_out   = fwd_output * grad_in[0].clamp_(min=0)
        return (grad_out, )

    handlers = []
    for pos, module in model._modules.items():
        if isinstance(module, nn.ReLU):
            handlers.append(module.register_forward_hook(relu_fwd))
            handlers.append(module.register_backward_hook(relu_bwd))

    # record gradient of first layer
    gradient = torch.zeros(*image.size())
    def first_layer(module, grad_in, grad_out):
        # global gradient
        gradient[...] = grad_in[0]

    handlers.append(model[0].register_backward_hook(first_layer))
    # backpropagate selected neuron
    model(image)[0, kernel].mean().backward()
    # normalize gradient
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient = gradient.data.numpy()[0].transpose(1, 2, 0)
    gradient = np.uint8(gradient * 255)
    # remove all hooks
    for handler in handlers: handler.remove()

    return gradient