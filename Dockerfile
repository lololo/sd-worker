FROM alpine/git:2.36.2 as download

COPY builder/clone.sh /clone.sh

# Clone the repos and clean unnecessary files
RUN . /clone.sh taming-transformers https://github.com/CompVis/taming-transformers.git 24268930bf1dce879235a7fddd0b2355b84d7ea6 && \
    rm -rf data assets **/*.ipynb

RUN . /clone.sh stable-diffusion-stability-ai https://github.com/Stability-AI/stablediffusion.git 47b6b607fdd31875c9279cd2f4f16b92e4ea958e && \
    rm -rf assets data/**/*.png data/**/*.jpg data/**/*.gif

RUN . /clone.sh CodeFormer https://github.com/sczhou/CodeFormer.git c5b4593074ba6214284d6acd5f1719b6c5d739af && \
    rm -rf assets inputs

RUN . /clone.sh BLIP https://github.com/salesforce/BLIP.git 48211a1594f1321b00f14c9f7a5b4813144b2fb9 && \
    . /clone.sh k-diffusion https://github.com/crowsonkb/k-diffusion.git 5b3af030dd83e0297272d861c19477735d0317ec && \
    . /clone.sh clip-interrogator https://github.com/pharmapsychotic/clip-interrogator 2486589f24165c8e3b303f84e9dbbea318df83e8 && \
    . /clone.sh generative-models https://github.com/Stability-AI/generative-models 45c443b316737a4ab6e40413d7794a7f5657c19f


FROM ubuntu:latest

RUN apt -y update -qq && \ 
    apt -y install -qq git wget aria2 libcairo2-dev pkg-config python3-dev python3-pip

RUN  wget https://github.com/camenduru/gperftools/releases/download/v1.0/libtcmalloc_minimal.so.4 -O /libtcmalloc_minimal.so.4

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    LD_PRELOAD=/libtcmalloc_minimal.so.4 \
    ROOT=/stable-diffusion-webui \
    PYTHONUNBUFFERED=1

RUN pip install -q torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 torchtext==0.15.2 torchdata==0.6.1 --extra-index-url https://download.pytorch.org/whl/cu118 && \
    pip install -q xformers==0.0.20 triton==2.0.0 gradio_client==0.2.7 -U

RUN git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git && \
    cd stable-diffusion-webui && \
    pip install -r requirements_versions.txt

COPY --from=download /repositories/ ${ROOT}/repositories/

RUN git clone https://huggingface.co/embed/negative ${ROOT}/embeddings/negative && \
    git clone https://huggingface.co/embed/lora ${ROOT}/models/Lora/positive && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/embed/upscale/resolve/main/4x-UltraSharp.pth -d ${ROOT}/models/ESRGAN -o 4x-UltraSharp.pth  && \
    wget https://raw.githubusercontent.com/camenduru/stable-diffusion-webui-scripts/main/run_n_times.py -O ${ROOT}/scripts/run_n_times.py && \
    git clone https://github.com/deforum-art/deforum-for-automatic1111-webui ${ROOT}/extensions/deforum-for-automatic1111-webui && \
    git clone https://github.com/camenduru/stable-diffusion-webui-huggingface ${ROOT}/extensions/stable-diffusion-webui-huggingface && \
    git clone https://github.com/kohya-ss/sd-webui-additional-networks ${ROOT}/extensions/sd-webui-additional-networks && \
    git clone https://github.com/Mikubill/sd-webui-controlnet ${ROOT}/extensions/sd-webui-controlnet && \
    git clone https://github.com/aigc-apps/EasyPhoto.git ${ROOT}/extensions/easyphoto
