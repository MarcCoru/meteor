from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class METEOR(nn.Module):
    def __init__(self, model, gradient_steps=60, inner_step_size=0.4, first_order=True, verbose=True, device="cpu",
                 batch_size=8, activation="softmax", seed=0, mode="one_vs_all"):
        super(METEOR, self).__init__()

        assert mode in ["one_vs_all", "one_vs_one"]
        self.mode = mode

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
        if self.mode == "one_vs_all":
            self.fit_one_vs_all(X,Y)
        elif self.mode == "one_vs_one":
            self.fit_one_vs_one(X, Y)

    @torch.no_grad()
    def predict(self, x, batch_size=16):
        if self.mode == "one_vs_all":
            return self.predict_one_vs_all(x, batch_size)
        elif self.mode == "one_vs_one":
            return self.predict_one_vs_one(x, batch_size)

    def fit_one_vs_all(self, X, Y):
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

    def fit_one_vs_one(self, X_all, Y_all):
        self.labels = np.unique(Y_all)
        self.model.train()

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.model.to(self.device)

        # to be filled by self.fit()
        self.params = {}
        for source_class in self.labels:
            for target_class in self.labels:

                # skip classifier with itself
                if source_class != target_class:

                    # skip B-A classifier if A-B already trained
                    if not f"{target_class}-{source_class}" in self.params.keys():

                        X_source = X_all[(Y_all == source_class)]
                        X_target = X_all[(Y_all == target_class)]

                        X = torch.vstack([X_source, X_target])
                        Y = torch.hstack([torch.zeros(X_source.shape[0], device=X_source.device, dtype=torch.float),
                                          torch.ones(X_target.shape[0], device=X_target.device, dtype=torch.float)])

                        print("source")
                        param = OrderedDict(self.model.meta_named_parameters())
                        for t in range(self.gradient_steps):
                            idxs = np.random.randint(X.shape[0], size=self.batch_size)
                            train_logit = self.model(X[idxs].float().to(self.device), params=param)
                            inner_loss = self.criterion(train_logit.squeeze(1), Y[idxs].to(self.device))
                            param = update_parameters(self.model, inner_loss, params=param,
                                                      inner_step_size=self.inner_step_size, first_order=self.first_order)

                            if self.verbose:
                                train_logit = self.model(X[idxs].float().to(self.device), params=param)
                                loss_after_adaptation = F.binary_cross_entropy_with_logits(train_logit.squeeze(1),
                                                                                           Y[idxs].to(self.device))
                                print(
                                    f"adapting to class {source_class}-{target_class} with {X.shape[0]} samples: step {t}/{self.gradient_steps}: support loss {inner_loss:.2f} -> {loss_after_adaptation:.2f}")

                        # move to cpu to save memory
                        param = OrderedDict({k: v.cpu() for k, v in param.items()})
                        self.params[f"{source_class}-{target_class}"] = param

    @torch.no_grad()
    def predict_one_vs_all(self, x, batch_size=16):
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

    @torch.no_grad()
    def predict_one_vs_one(self, x, batch_size=16):
        self.model.eval()
        scores = {str(k): [] for k in self.labels}
        votes = {str(k): [] for k in self.labels}
        with torch.no_grad():
            for combination, param in self.params.items():
                print(f"predictioning combination {combination}")
                source, target = combination.split("-")
                param = OrderedDict({k: v.to(self.device) for k, v in param.items()})
                logit = torch.vstack(
                    [self.model(inp.float().to(self.device), params=param) for inp in torch.split(x, batch_size)])
                proba = torch.sigmoid(logit)

                scores[target].append(proba)
                scores[source].append(1 - proba)

                votes[target].append((proba >= 0.5).long())
                votes[source].append((proba < 0.5).long())

        scores_sum = [torch.hstack(scores[v]).sum(1) for v in votes.keys()]
        y_pred_idx = torch.stack(scores_sum).argmax(0)
        y_pred = np.array([int(i) for i in np.array(list(votes.keys()))[y_pred_idx.cpu()]])

        # N x C
        if self.activation == "softmax":
            probas = torch.softmax(torch.stack(scores_sum), dim=0)
        elif self.activation == "sigmoid":
            probas = torch.sigmoid(torch.stack(scores_sum))
        else:
            raise NotImplementedError()

        return y_pred, probas
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
