FROM ubuntu:latest


RUN apt -y update -qq && \ 
    apt -y install -qq git wget libgl1 aria2 fonts-dejavu-core rsync git jq moreutils  libgoogle-perftools-dev procps libglib2.0-0 libcairo2-dev pkg-config python3-dev python3-pip

RUN  wget https://github.com/camenduru/gperftools/releases/download/v1.0/libtcmalloc_minimal.so.4 -O /libtcmalloc_minimal.so.4

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    LD_PRELOAD=/libtcmalloc_minimal.so.4 \
    ROOT=/stable-diffusion-webui \
    PYTHONUNBUFFERED=1

# RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install -q torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 torchtext==0.15.2 torchdata==0.6.1 --extra-index-url https://download.pytorch.org/whl/cu118 -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install -q xformers==0.0.20 triton==2.0.0 gradio_client==0.2.7 -i https://pypi.tuna.tsinghua.edu.cn/simple



RUN git clone  https://github.com/AUTOMATIC1111/stable-diffusion-webui.git && \
    cd stable-diffusion-webui && \
    git reset --hard 
# git -C  ${ROOT}/repositories/stable-diffusion-stability-ai reset --hard

#RUN pip install -r ${ROOT}/requirements_versions.txt 


#COPY builder/requirements.txt /requirements.txt
#RUN --mount=type=cache,target=/root/.cache/pip \
#    pip install --upgrade pip && \
#    pip install --upgrade -r /requirements.txt --no-cache-dir && \
#    rm /requirements.txt

RUN git clone https://huggingface.co/embed/negative ${ROOT}/embeddings/negative && \
    git clone https://huggingface.co/embed/lora ${ROOT}/models/Lora/positive && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/embed/upscale/resolve/main/4x-UltraSharp.pth -d ${ROOT}/models/ESRGAN -o 4x-UltraSharp.pth  && \
    wget https://raw.githubusercontent.com/camenduru/stable-diffusion-webui-scripts/main/run_n_times.py -O ${ROOT}/scripts/run_n_times.py && \
    git clone https://github.com/deforum-art/deforum-for-automatic1111-webui ${ROOT}/extensions/deforum-for-automatic1111-webui && \
    git clone https://github.com/camenduru/stable-diffusion-webui-huggingface ${ROOT}/extensions/stable-diffusion-webui-huggingface && \
    git clone https://github.com/kohya-ss/sd-webui-additional-networks ${ROOT}/extensions/sd-webui-additional-networks && \
    git clone https://github.com/Mikubill/sd-webui-controlnet ${ROOT}/extensions/sd-webui-controlnet && \
    git clone https://github.com/aigc-apps/EasyPhoto.git ${ROOT}/extensions/easyphoto

RUN apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*
