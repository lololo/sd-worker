FROM ubuntu:latest


RUN apt -y update -qq && \ 
    apt -y install -qq vim git wget libgl1 aria2 fonts-dejavu-core rsync git jq moreutils  libgoogle-perftools-dev procps libglib2.0-0 libcairo2-dev pkg-config python3-dev python3-pip

RUN  wget https://github.com/camenduru/gperftools/releases/download/v1.0/libtcmalloc_minimal.so.4 -O /libtcmalloc_minimal.so.4

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    LD_PRELOAD=/libtcmalloc_minimal.so.4 \
    ROOT=/stable-diffusion-webui \
    PYTHONUNBUFFERED=1

# RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

RUN pip install -q torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 torchtext==0.15.2 torchdata==0.6.1 --extra-index-url https://download.pytorch.org/whl/cu118 && \
    pip install -q xformers==0.0.20 triton==2.0.0 gradio_client==0.2.7 

COPY builder/requirements.txt /requirements.txt
RUN pip install -r /requirements.txt


RUN git clone -b v2.6 https://github.com/camenduru/stable-diffusion-webui && \
    cd stable-diffusion-webui && \
    git reset --hard 
# git -C  ${ROOT}/repositories/stable-diffusion-stability-ai reset --hard

RUN pip install -r ${ROOT}/requirements_versions.txt 

# COPY --from=download /repositories/ ${ROOT}/repositories/
# # COPY --from=download /model.safetensors /model.safetensors
# RUN mkdir ${ROOT}/interrogate && cp ${ROOT}/repositories/clip-interrogator/data/* ${ROOT}/interrogate


#COPY builder/requirements.txt /requirements.txt
#RUN --mount=type=cache,target=/root/.cache/pip \
#    pip install --upgrade pip && \
#    pip install --upgrade -r /requirements.txt --no-cache-dir && \
#    rm /requirements.txt

RUN git clone https://github.com/Mikubill/sd-webui-controlnet ${ROOT}/extensions/sd-webui-controlnet && \
    git clone https://github.com/lololo/sd-webui-EasyPhoto.git ${ROOT}/extensions/sd-webui-EasyPhoto

# COPY  ../stable-diffusion-webui/extensions/sd-webui-controlnet/models /models
# COPY  /home/lei/Documents/stable-diffusion-webui/extensions/sd-webui-EasyPhoto ${ROOT}/extensions/sd-webui-EasyPhoto
# COPY builder/controlnet_model.txt /controlnet_model.txt
# RUN  aria2c --console-log-level=error -c -x 16 -s 16 -k 1M -i /controlnet_model.txt -d ${ROOT}/extensions/sd-webui-controlnet/models

# RUN  aria2c --console-log-level=error -c -x 16 -s 16 -k 1M http://127.0.0.1:8081/v1-5-pruned-emaonly.ckpt -d ${ROOT}/models/Stable-diffusion  -o v1-5-pruned-emaonly.ckpt

RUN apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

COPY /builder/controlnet_model.txt /controlnet_model.txt

ADD src .

RUN chmod +x /start.sh
CMD /start.sh