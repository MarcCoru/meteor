import matplotlib.pyplot as plt
import numpy as np
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


def tensorboard_batch_figure(batch, summary_writer, classes, targets, predictions, imgsize=4, global_step=None):
    train_img, train_label, train_ids = batch["train"]
    test_img, test_label, test_ids = batch["test"]

    # infer batch_size, shots, ways from the data
    batch_size = train_label.shape[0]
    unique_classes, count_per_class = train_label[0].unique(return_counts=True)
    num_ways = len(unique_classes)
    num_shots = int(count_per_class[0])
    assert num_ways * num_shots == train_label.shape[1]

    for task_id in range(batch_size):
        fig, axs = plt.subplots(2, num_shots * num_ways,
                                figsize=(num_shots * num_ways * imgsize, 2 * imgsize),
                                sharey=True)

        np.array(classes)[targets].reshape(batch_size, -1)

        #### First Row: Support
        axs_row = axs[0]

        rgb_images = np.stack([to_rgb(image) for image in train_img[task_id]])
        id = np.array(train_ids)[:, task_id]
        train_label_row = train_label[task_id]
        train_label_row = np.array(classes)[train_label_row]

        for ax, image, id_, target in zip(axs_row, rgb_images, id, train_label_row):
            ax.imshow(image)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.set_title(f"{target}")
        axs_row[0].set_ylabel("support")

        #### Second Row: Query
        axs_row = axs[1]

        targets_row = targets.reshape(batch_size, -1)[task_id]
        predictions_row = predictions.reshape(batch_size, -1)[task_id]

        targets_row = np.array(classes)[targets_row]
        predictions_row = np.array(classes)[predictions_row]

        rgb_images = np.stack([to_rgb(image) for image in test_img[task_id]])
        id = np.array(test_ids)[:, task_id]
        for ax, image, target, prediction in zip(axs_row, rgb_images, targets_row, predictions_row):
            ax.imshow(image)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.set_title(f"{target} (pred {prediction})")
        axs_row[0].set_ylabel("query")

        name = f"Region {id[0].split('/')[1]} ({id[0].split('/')[0]})"
        summary_writer.add_figure(f"meta-test: {name}", figure=fig, global_step=global_step)

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
