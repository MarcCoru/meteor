# GPU Module

This is an isolated module for code to run on a GPU-enabled backend (separate for the flask server)

## Setup

```bash
docker build -t gpu_module .
```

## Run

```
nvidia-docker run -t gpu_module python main.py
```
to test the prediction for this module on demo data.

the output should look like this (numbers may vary):
```
adapting to class 0 with 3 samples: step 0/3: support loss 0.51 -> 0.02
adapting to class 0 with 3 samples: step 1/3: support loss 0.02 -> 0.01
adapting to class 0 with 3 samples: step 2/3: support loss 0.20 -> 0.06
adapting to class 1 with 3 samples: step 0/3: support loss 0.79 -> 0.00
adapting to class 1 with 3 samples: step 1/3: support loss 0.00 -> 0.00
adapting to class 1 with 3 samples: step 2/3: support loss 0.00 -> 0.00
adapting to class 2 with 3 samples: step 0/3: support loss 0.82 -> 0.17
adapting to class 2 with 3 samples: step 1/3: support loss 0.55 -> 0.02
adapting to class 2 with 3 samples: step 2/3: support loss 0.10 -> 0.00
predicting class 0
predicting class 1
predicting class 2
[{'source': '13', 'target': '10', 'value': 1}, {'source': '13', 'target': '14', 'value': 1}, {'source': '13', 'target': '11', 'value': 1}, {'source': '18', 'target': '19', 'value': 1}, {'source': '18', 'target': '17', 'value': 1}, {'source': '18', 'target': '16', 'value': 1}, {'source': '5', 'target': '1', 'value': 1}, {'source': '5', 'target': '8', 'value': 1}, {'source': '5', 'target': '6', 'value': 1}, {'source': '5', 'target': '4', 'value': 1}, {'source': '5', 'target': '2', 'value': 1}, {'source': '5', 'target': '9', 'value': 1}, {'source': '5', 'target': '7', 'value': 1}, {'source': '5', 'target': '3', 'value': 1}, {'source': '5', 'target': '0', 'value': 1}]```
``
