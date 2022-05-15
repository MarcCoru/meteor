run app

```
# install python and requirements
bash setup.sh

# start server
env/bin/python serve.py
```

## Docker

### RUN 

CPU
```bash
sudo docker run -p 8080:8080 marccoru/meteor
```

GPU
```bash
sudo nvidia-docker run -p 8080:8080 marccoru/meteor
```

### Build

```bash
sudo docker build -t marccoru/meteor .
```

