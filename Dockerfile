FROM ubuntu:22.04


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

RUN git clone -b v2.6 https://github.com/camenduru/stable-diffusion-webui ${ROOT} && \
    cd ${ROOT} && \
    git reset --hard 

COPY builder/requirements.txt /requirements.txt

RUN git clone https://github.com/lololo/sd-webui-controlnet.git ${ROOT}/extensions/sd-webui-controlnet && \
    git clone https://github.com/lololo/sd-webui-EasyPhoto.git ${ROOT}/extensions/sd-webui-EasyPhoto && \
    git clone https://github.com/CompVis/taming-transformers.git   ${ROOT}/repositories/taming-transformers && \
    git clone https://github.com/Stability-AI/stablediffusion.git ${ROOT}/repositories/stablediffusion && \
    git clone https://github.com/sczhou/CodeFormer.git ${ROOT}/repositories/CodeFormer && \
    git clone https://github.com/salesforce/BLIP.git  ${ROOT}/repositories/BLIP && \
    git clone https://github.com/crowsonkb/k-diffusion.git ${ROOT}/repositories/k-diffusion && \
    git clone https://github.com/pharmapsychotic/clip-interrogator  ${ROOT}/repositories/clip-interrogator && \
    git clone https://github.com/Stability-AI/generative-models ${ROOT}/repositories/generative-models

# RUN mkdir ${ROOT}/interrogate && cp ${ROOT}/repositories/clip-interrogator/data/* ${ROOT}/interrogate
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /requirements.txt \
    pip install -r ${ROOT}/requirements_versions.txt  \
    pip install -r ${ROOT}/repositories/CodeFormer/requirements.txt

RUN apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

COPY /builder/controlnet_model.txt /controlnet_model.txt

ADD src .

RUN chmod +x /start.sh
CMD /start.sh