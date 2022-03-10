from collections import OrderedDict
from torch import nn
import torch.nn.functional as F
import torch
import numpy as np

class BagOfMAML(nn.Module):
    def __init__(self, model, gradient_steps=1, inner_step_size=0.4, first_order=True, verbose=False, device="cpu", batch_size=8, activation="softmax"):
        super(BagOfMAML, self).__init__()
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
                    loss_after_adaptation = F.binary_cross_entropy_with_logits(train_logit.squeeze(1), y[idxs].to(self.device))
                    print(f"adapting to class {target_class} with {X.shape[0]} samples: step {t}/{self.gradient_steps}: support loss {inner_loss:.2f} -> {loss_after_adaptation:.2f}")

            self.params.append(param)

    @torch.no_grad()
    def predict(self, x, batch_size=16):
        self.model.eval()
        logits = []
        for class_id, param in zip(self.labels, self.params):
            if self.verbose:
                print(f"predicting class {class_id}")

            logit = torch.vstack([self.model(inp.float().to(self.device), params=param) for inp in torch.split(x, batch_size)])
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


class BagOfMAMLEnsemble(nn.Module):
    def __init__(self, model, gradient_steps=1, inner_step_size=0.4, first_order=False, verbose=False, device="cpu", batch_size=8, num_members=3, holdout_fraction=0.25, activation="softmax"):
        super(BagOfMAMLEnsemble, self).__init__()
        self.model = model.to(device)
        self.ways = 1
        self.gradient_steps = gradient_steps
        self.inner_step_size = inner_step_size
        self.first_order = first_order
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1))
        self.verbose = verbose
        self.device = device
        self.holdout_fraction = holdout_fraction
        self.activation = activation

        # to be filled by self.fit()
        self.params = []
        self.labels = None
        self.batch_size = batch_size
        self.num_members = num_members

    def fit(self, X_all, Y_all):
        self.labels = np.unique(Y_all)
        self.model.train()

        for target_class in self.labels:
            self.model.zero_grad()

            y_all = (Y_all == target_class).to(float)

            member_params = []
            for i in range(self.num_members):
                if self.verbose:
                    print(f"fitting member {i}")

                holdout_size = int(X_all.size(0) * self.holdout_fraction)
                holdout_idxs = np.random.randint(X_all.shape[0], size=holdout_size)
                inverse_holdout_idxs = np.array([i for i in range(X_all.shape[0]) if i not in holdout_idxs])
                X_holdout, y_holdout = X_all[holdout_idxs], y_all[holdout_idxs]
                X, y = X_all[inverse_holdout_idxs], y_all[inverse_holdout_idxs]

                param = OrderedDict(self.model.meta_named_parameters())
                best_holdoutloss = 999
                for t in range(self.gradient_steps):
                    idxs = np.random.randint(X.shape[0], size=self.batch_size)
                    train_logit = self.model(X[idxs].float().to(self.device), params=param)

                    inner_loss = self.criterion(train_logit.squeeze(1), y[idxs].to(self.device))
                    param = update_parameters(self.model, inner_loss, params=param,
                                              inner_step_size=self.inner_step_size, first_order=self.first_order)

                    with torch.no_grad():
                        idxs = np.random.randint(X_holdout.shape[0], size=self.batch_size)
                        train_logit = self.model(X_holdout[idxs].float().to(self.device), params=param)
                        holdout_loss = F.binary_cross_entropy_with_logits(train_logit.squeeze(1), y_holdout[idxs].to(self.device))
                        if holdout_loss < best_holdoutloss:
                            best_holdoutloss = holdout_loss
                            best_params = param
                            if self.verbose:
                                print(
                                    f"optimizing for class {target_class} with {X.shape[0]} samples: step {t}/{self.gradient_steps}: support loss {best_holdoutloss:.2f}")
                member_params.append(best_params)
            self.params.append(member_params)

    @torch.no_grad()
    def predict(self, x, batch_size=16):
        self.model.eval()
        logits = []
        for class_id, member_params in zip(self.labels, self.params):
            logits_ = []
            for param in member_params:
                if self.verbose:
                    print(f"predicting class {class_id}")
                logit = torch.vstack(
                    [self.model(inp.float().to(self.device), params=param) for inp in torch.split(x, batch_size)]).squeeze(-1)

                logits_.append(logit)

            logits.append(torch.stack(logits_).cpu())

        logits = torch.stack(logits)

        if self.activation == "sigmoid":
            probas = torch.sigmoid(logits)
        elif self.activation == "softmax":
            probas = torch.softmax(logits, dim=0)

        scores = probas.mean(1)

        predictions = scores.argmax(0)

        return self.labels[predictions].T, probas


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
