from .transforms import get_classification_transform
from .model import prepare_classification_model
from .data import IGBP_simplified_classes
import torch
from collections import OrderedDict

def prepare_transform_and_model(args):
    if args.s2only:
        transform = get_classification_transform(s2only=True, augment=True)
        inplanes = 13
    elif args.rgbonly:
        transform = get_classification_transform(rgbonly=True, augment=True)
        inplanes = 3
    else: # s1 and s2
        transform = get_classification_transform(augment=True)
        inplanes = 15

    if args.one_vs_all:
        num_classes = 1
    else:
        classes = IGBP_simplified_classes
        num_classes = len(classes)

    model, mask, mask_optimizer = prepare_classification_model(num_classes,
                                         inplanes=inplanes,
                                         resnet=args.resnet, norm=args.norm,
                                                               prototypicalnetwork=args.prototypicalnetwork)


    return transform, model, mask, mask_optimizer

def update_parameters(model, loss, params=None, mask=None, inner_step_size=0.5, first_order=False):
    """Update the parameters of the model, with one step of gradient descent.
    Parameters
    ----------
    model : `MetaModule` instance
        Model.
    loss : `torch.FloatTensor` instance
        Loss function on which the gradient are computed for the descent step.
    inner_step_size : float (default: `0.5`)
        Step-size of the gradient descent step.
    first_order : bool (default: `False`)
        If `True`, use the first-order approximation of MAML.
    Returns
    -------
    params : OrderedDict
        Dictionary containing the parameters after one step of adaptation.
    """

    if params is None:
        meta_params_list = model.module.meta_parameters() if \
            isinstance(model, torch.nn.DataParallel) else model.meta_parameters()
    else:
        meta_params = params
        meta_params_list = list(meta_params.items())

    params = model.module.meta_parameters() if isinstance(model, torch.nn.DataParallel) else model.meta_parameters()

    grads = torch.autograd.grad(loss, params,
                                create_graph=not first_order)

    if mask is not None:
        mask_weights = mask.forward()

    params = OrderedDict()
    for (name, param), grad in zip(meta_params_list, grads):

        # overwrite static step size with dynamically learned step size
        if hasattr(model, "learning_rates"):
            inner_step_size = model.learning_rates[name.replace('.', '-')]

        # gradient mask
        if mask is not None:
            # perform masked gradient update
            params[name] = param - inner_step_size * (mask_weights[name]*grad)
        else:
            # perform manual gradient step
            params[name] = param - inner_step_size * grad

    return params
