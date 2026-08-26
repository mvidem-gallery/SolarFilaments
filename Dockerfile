FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git python3 python3-pip && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

COPY kaggle.json /root/.kaggle/kaggle.json
RUN chmod 600 /root/.kaggle/kaggle.json

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["/bin/bash"]
