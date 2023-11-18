FROM alpine/git:2.36.2 as download

COPY builder/clone.sh /clone.sh

# Clone the repos and clean unnecessary files
RUN . /clone.sh taming-transformers https://github.com/CompVis/taming-transformers.git 3ba01b241669f5ade541ce990f7650a3b8f65318 && \
    rm -rf data assets **/*.ipynb

RUN . /clone.sh stable-diffusion-stability-ai https://github.com/Stability-AI/stablediffusion.git cf1d67a6fd5ea1aa600c4df58e5b47da45f6bdbf && \
    rm -rf assets data/**/*.png data/**/*.jpg data/**/*.gif

RUN . /clone.sh CodeFormer https://github.com/sczhou/CodeFormer.git 8392d0334956108ab53d9439c4b9fc9c4af0d66d && \
    rm -rf assets inputs

RUN . /clone.sh BLIP https://github.com/salesforce/BLIP.git 3a29b7410476bf5f2ba0955827390eb6ea1f4f9d && \
    . /clone.sh k-diffusion https://github.com/crowsonkb/k-diffusion.git 045515774882014cc14c1ba2668ab5bad9cbf7c0 && \
    . /clone.sh clip-interrogator https://github.com/pharmapsychotic/clip-interrogator bc07ce62c179d3aab3053a623d96a071101d11cb && \
    . /clone.sh generative-models https://github.com/Stability-AI/generative-models 477d8b9a7730d9b2e92b326a770c0420d00308c9


FROM ubuntu:latest


RUN apt -y update -qq && \ 
    apt -y install -qq git wget aria2 libcairo2-dev pkg-config python3-dev python3-pip

RUN  wget https://github.com/camenduru/gperftools/releases/download/v1.0/libtcmalloc_minimal.so.4 -O /libtcmalloc_minimal.so.4

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    LD_PRELOAD=/libtcmalloc_minimal.so.4 \
    ROOT=/stable-diffusion-webui \
    PYTHONUNBUFFERED=1

RUN --mount=type=cache,target=/cache --mount=type=cache,target=/root/.cache/pip \
    pip install -q torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 torchtext==0.15.2 torchdata==0.6.1 --extra-index-url https://download.pytorch.org/whl/cu118 && \
    pip install -q xformers==0.0.20 triton==2.0.0 gradio_client==0.2.7 -U

RUN --mount=type=cache,target=/root/.cache/pip \
    git clone  https://github.com/AUTOMATIC1111/stable-diffusion-webui.git && \
    cd stable-diffusion-webui 

COPY --from=download /repositories/ ${ROOT}/repositories/

RUN mkdir ${ROOT}/interrogate && cp ${ROOT}/repositories/clip-interrogator/data/* ${ROOT}/interrogate
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r ${ROOT}/repositories/CodeFormer/requirements.txt


COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# RUN git clone https://huggingface.co/embed/negative ${ROOT}/embeddings/negative && \
#        git clone https://huggingface.co/embed/lora ${ROOT}/models/Lora/positive && \
#        aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/embed/upscale/resolve/main/4x-UltraSharp.pth -d ${ROOT}/models/ESRGAN -o 4x-UltraSharp.pth  && \
#        wget https://raw.githubusercontent.com/camenduru/stable-diffusion-webui-scripts/main/run_n_times.py -O ${ROOT}/scripts/run_n_times.py && \
#        git clone https://github.com/deforum-art/deforum-for-automatic1111-webui ${ROOT}/extensions/deforum-for-automatic1111-webui && \
#        git clone https://github.com/camenduru/stable-diffusion-webui-huggingface ${ROOT}/extensions/stable-diffusion-webui-huggingface && \
#        git clone https://github.com/kohya-ss/sd-webui-additional-networks ${ROOT}/extensions/sd-webui-additional-networks && \
#        git clone https://github.com/Mikubill/sd-webui-controlnet ${ROOT}/extensions/sd-webui-controlnet && \
#        git clone https://github.com/aigc-apps/EasyPhoto.git ${ROOT}/extensions/easyphoto

RUN apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*
