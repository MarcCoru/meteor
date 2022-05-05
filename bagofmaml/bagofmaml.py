from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class BagOfMAML(nn.Module):
    def __init__(self, model, gradient_steps=1, inner_step_size=0.4, first_order=True, verbose=False, device="cpu",
                 batch_size=8, activation="softmax", seed=0):
        super(BagOfMAML, self).__init__()

        self.seed = seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.model = model.to(device)
        self.ways = 1
        self.gradient_steps = gradient_steps
        self.inner_step_size = inner_step_size
        self.first_order = first_order
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1))
        self.verbose = verbose
        self.device = device
        self.activation = activation

        self.labels = None
        self.batch_size = batch_size

    def fit(self, X, Y):
        self.labels = np.unique(Y)
        self.model.train()

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # to be filled by self.fit()
        self.params = []

        for target_class in self.labels:
            self.model.zero_grad()

            y = (Y == target_class).to(float)

            param = OrderedDict(self.model.meta_named_parameters())
            for t in range(self.gradient_steps):
                idxs = np.random.randint(X.shape[0], size=self.batch_size)
                train_logit = self.model(X[idxs].float().to(self.device), params=param)

                inner_loss = self.criterion(train_logit.squeeze(1), y[idxs].to(self.device))
                param = update_parameters(self.model, inner_loss, params=param,
                                          inner_step_size=self.inner_step_size, first_order=self.first_order)

                if self.verbose:
                    train_logit = self.model(X[idxs].float().to(self.device), params=param)
                    loss_after_adaptation = F.binary_cross_entropy_with_logits(train_logit.squeeze(1),
                                                                               y[idxs].to(self.device))
                    print(f"adapting to class {target_class} with {X.shape[0]} samples: step {t}/{self.gradient_steps}: support loss {inner_loss:.2f} -> {loss_after_adaptation:.2f}")

            self.params.append(param)

    @torch.no_grad()
    def predict(self, x, batch_size=16):
        self.model.eval()
        logits = []
        for class_id, param in zip(self.labels, self.params):
            if self.verbose:
                print(f"predicting class {class_id}")

            logit = torch.vstack(
                [self.model(inp.float().to(self.device), params=param) for inp in torch.split(x, batch_size)])
            logits.append(logit.squeeze(1).cpu())

        # N x C
        if self.activation == "softmax":
            probas = torch.softmax(torch.stack(logits), dim=0)
        elif self.activation == "sigmoid":
            probas = torch.sigmoid(torch.stack(logits))
        else:
            raise NotImplementedError()

        predictions = probas.argmax(0)

        return self.labels[predictions], probas

def update_parameters(model, loss, params=None, inner_step_size=0.5, first_order=False):
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
        meta_params = model.meta_parameters()
    else:
        meta_params = params
        meta_params_list = list(meta_params.items())

    grads = torch.autograd.grad(loss, meta_params.values(),
                                create_graph=not first_order)

    params = OrderedDict()
    for (name, param), grad in zip(meta_params_list, grads):

        # overwrite static step size with dynamically learned step size
        if hasattr(model, "learning_rates"):
            inner_step_size = model.learning_rates[name.replace('.', '-')]

        # perform manual gradient step
        params[name] = param - inner_step_size * grad

    return params
