FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

COPY . .

# install requirements
RUN pip install -r requirements.txt

# download and cache the model once so that is not re-downloaded every time the script runs
RUN python -c 'import torch; torch.hub.load_state_dict_from_url("https://bagofmaml.s3.eu-central-1.amazonaws.com/app/model.pth", map_location="cpu")'

# download data once
RUN python data.py

# provide entrypoint
CMD [ "python", "./main.py"]
