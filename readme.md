# Bag-of-MAML

![](https://github.com/marccoru/bagofmaml/actions/workflows/python-package-conda.yml/badge.svg)

```commandline
pip install bagofmaml
```


```commandline
from bagofmaml import BagOfMAMLEnsemble, BagOfMAML
from bagofmaml import models
from bagofmaml.examples.beirut import data

basemodel = models.ResNet()
bag = BagOfMAML(basemodel)
```
