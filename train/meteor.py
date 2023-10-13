from collections import OrderedDict
from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
from utils import update_parameters

class Meteor(nn.Module):
    def __init__(self, model, gradient_steps=1, inner_step_size=0.4, first_order=True, verbose=False, device="cpu",
                 batch_size=8, activation="softmax", seed=0, mask=None):
        super(Meteor, self).__init__()

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
        self.mask = mask

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
                    print(
                        f"adapting to class {target_class} with {X.shape[0]} samples: step {t}/{self.gradient_steps}: support loss {inner_loss:.2f} -> {loss_after_adaptation:.2f}")

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