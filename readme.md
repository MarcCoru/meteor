# Bag-of-MAML

![](https://github.com/marccoru/bagofmaml/actions/workflows/python-package-conda.yml/badge.svg)

## Install Package and Requirements

```commandline
pip install git+https://github.com/marccoru/bagofmaml.git
```

## Getting Started

```python
from bagofmaml import BagOfMAML
from bagofmaml import models
import torch

# initialize model
basemodel = models.get_model("maml_resnet12_rgb")
bag = BagOfMAML(basemodel)

# fine-tune model to labelled data
X_support, y_support = torch.rand(10, 3, 128, 128), torch.randint(3, (10,))
bag.fit(X_support, y_support)

# predict
X_query = torch.rand(10, 3, 128, 128)
y_pred, y_scores = bag.predict(X_query)
```
