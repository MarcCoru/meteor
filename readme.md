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


## Minimal Working Examples 

require installation of extra packages for plotting via `pip install -e "git+https://github.com/marccoru/bagofmaml.git[examples]"`

### Beirut Example

![](doc/beirut.png)

```python
import torch
from bagofmaml import BagOfMAML
from bagofmaml import models
from bagofmaml.examples.beirut import get_data, plot

# download data
timeseries, dates_dt = get_data()

# select support images from time series (first and last <shot> images)
shot = 3

start = timeseries[:shot]
end = timeseries[-shot:]
X_support = torch.vstack([start, end])
y_support = torch.hstack([torch.zeros(shot), torch.ones(shot)]).long()

# get model
model = models.get_model("maml_resnet12_s2")
bag = BagOfMAML(model, verbose=True, inner_step_size=0.4, gradient_steps=20)

# fit and predict
bag.fit(X_support, y_support)
y_pred, y_score = bag.predict(timeseries)

# plot score
plot(y_score, dates_dt)
```
