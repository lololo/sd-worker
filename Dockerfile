# ---------------------------------------------------------------------------- #
#                         Stage 1: Download the models                         #
# ---------------------------------------------------------------------------- #
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

# RUN apk add --no-cache wget && \
#     wget -q -O /model.safetensors https://civitai.com/api/download/models/25494



# ---------------------------------------------------------------------------- #
#                        Stage 3: Build the final image                        #
# ---------------------------------------------------------------------------- #
FROM python:3.10.9-slim as build_final_image

ARG SHA=5ef669de080814067961f28357256e8fe27544f4

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    LD_PRELOAD=libtcmalloc.so \
    ROOT=/stable-diffusion-webui \
    PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN export COMMANDLINE_ARGS="--skip-torch-cuda-test --precision full --no-half"
RUN export TORCH_COMMAND='pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/rocm5.6'

RUN apt-get update && \
    apt install -y \
    fonts-dejavu-core rsync git jq moreutils aria2 wget libgoogle-perftools-dev procps libgl1 libglib2.0-0 && \
    apt-get autoremove -y && rm -rf /var/lib/apt/lists/* && apt-get clean -y

RUN --mount=type=cache,target=/cache --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN --mount=type=cache,target=/root/.cache/pip \
    git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git && \
    cd stable-diffusion-webui && \
    git reset --hard ${SHA}&& \ 
    pip install -r requirements_versions.txt

RUN git clone https://huggingface.co/embed/negative ${ROOT}/embeddings/negative && \
       git clone https://huggingface.co/embed/lora ${ROOT}/models/Lora/positive && \
       aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/embed/upscale/resolve/main/4x-UltraSharp.pth -d ${ROOT}/models/ESRGAN -o 4x-UltraSharp.pth  && \
       wget https://raw.githubusercontent.com/camenduru/stable-diffusion-webui-scripts/main/run_n_times.py -O ${ROOT}/scripts/run_n_times.py && \
       git clone https://github.com/deforum-art/deforum-for-automatic1111-webui ${ROOT}/extensions/deforum-for-automatic1111-webui && \
       git clone https://github.com/camenduru/stable-diffusion-webui-images-browser ${ROOT}/extensions/stable-diffusion-webui-images-browser && \
       git clone https://github.com/camenduru/stable-diffusion-webui-huggingface ${ROOT}/extensions/stable-diffusion-webui-huggingface && \
       git clone https://github.com/camenduru/sd-civitai-browser ${ROOT}/extensions/sd-civitai-browser && \
       git clone https://github.com/kohya-ss/sd-webui-additional-networks ${ROOT}/extensions/sd-webui-additional-networks && \
       git clone https://github.com/Mikubill/sd-webui-controlnet ${ROOT}/extensions/sd-webui-controlnet && \
       git clone https://github.com/fkunn1326/openpose-editor ${ROOT}/extensions/openpose-editor && \
       git clone https://github.com/jexom/sd-webui-depth-lib ${ROOT}/extensions/sd-webui-depth-lib && \
       git clone https://github.com/hnmr293/posex ${ROOT}/extensions/posex && \
       git clone https://github.com/nonnonstop/sd-webui-3d-open-pose-editor ${ROOT}/extensions/sd-webui-3d-open-pose-editor && \
       git clone https://github.com/camenduru/sd-webui-tunnels ${ROOT}/extensions/sd-webui-tunnels && \
       git clone https://github.com/etherealxx/batchlinks-webui ${ROOT}/extensions/batchlinks-webui && \
       git clone https://github.com/camenduru/stable-diffusion-webui-catppuccin ${ROOT}/extensions/stable-diffusion-webui-catppuccin && \
       git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui-rembg ${ROOT}/extensions/stable-diffusion-webui-rembg && \
       git clone https://github.com/ashen-sensored/stable-diffusion-webui-two-shot ${ROOT}/extensions/stable-diffusion-webui-two-shot && \
       git clone https://github.com/thomasasfk/sd-webui-aspect-ratio-helper ${ROOT}/extensions/sd-webui-aspect-ratio-helper && \
       git clone https://github.com/tjm35/asymmetric-tiling-sd-webui ${ROOT}/extensions/asymmetric-tiling-sd-webui 

# RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11e_sd15_ip2p_fp16.safetensors -d ${ROOT}/extensions/sd-webui-controlnet/models -o control_v11e_sd15_ip2p_fp16.safetensors  && \
#         aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11e_sd15_shuffle_fp16.safetensors -d ${ROOT}/extensions/sd-webui-controlnet/models -o control_v11e_sd15_shuffle_fp16.safetensors && \
#         aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny_fp16.safetensors -d ${ROOT}/extensions/sd-webui-controlnet/models -o control_v11p_sd15_canny_fp16.safetensors && \
#         aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth_fp16.safetensors -d ${ROOT}/extensions/sd-webui-controlnet/models -o control_v11f1p_sd15_depth_fp16.safetensors && \
#         aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_inpaint_fp16.safetensors -d ${ROOT}/extensions/sd-webui-controlnet/models -o control_v11p_sd15_inpaint_fp16.safetensors && \
#         aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_lineart_fp16.safetensors -d ${ROOT}/extensions/sd-webui-controlnet/models -o control_v11p_sd15_lineart_fp16.safetensors && \
#         aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_mlsd_fp16.safetensors -d ${ROOT}/extensions/sd-webui-controlnet/models -o control_v11p_sd15_mlsd_fp16.safetensors && \
#         aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_normalbae_fp16.safetensors -d ${ROOT}/extensions/sd-webui-controlnet/models -o control_v11p_sd15_normalbae_fp16.safetensors && \
#         aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose_fp16.safetensors -d ${ROOT}/extensions/sd-webui-controlnet/models -o control_v11p_sd15_openpose_fp16.safetensors && \
#         aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_scribble_fp16.safetensors -d ${ROOT}/extensions/sd-webui-controlnet/models -o control_v11p_sd15_scribble_fp16.safetensors && \
#         aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_seg_fp16.safetensors -d ${ROOT}/extensions/sd-webui-controlnet/models -o control_v11p_sd15_seg_fp16.safetensors && \
#         aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_softedge_fp16.safetensors -d ${ROOT}/extensions/sd-webui-controlnet/models -o control_v11p_sd15_softedge_fp16.safetensors && \
#         aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15s2_lineart_anime_fp16.safetensors -d ${ROOT}/extensions/sd-webui-controlnet/models -o control_v11p_sd15s2_lineart_anime_fp16.safetensors && \
#         aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11f1e_sd15_tile_fp16.safetensors -d ${ROOT}/extensions/sd-webui-controlnet/models -o control_v11f1e_sd15_tile_fp16.safetensors && \
#         aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11e_sd15_ip2p_fp16.yaml -d ${ROOT}/extensions/sd-webui-controlnet/models -o control_v11e_sd15_ip2p_fp16.yaml && \
#         aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11e_sd15_shuffle_fp16.yaml -d ${ROOT}/extensions/sd-webui-controlnet/models -o control_v11e_sd15_shuffle_fp16.yaml && \
#         aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_canny_fp16.yaml -d ${ROOT}/extensions/sd-webui-controlnet/models -o control_v11p_sd15_canny_fp16.yaml && \
#         aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11f1p_sd15_depth_fp16.yaml -d ${ROOT}/extensions/sd-webui-controlnet/models -o control_v11f1p_sd15_depth_fp16.yaml && \
#         aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_inpaint_fp16.yaml -d ${ROOT}/extensions/sd-webui-controlnet/models -o control_v11p_sd15_inpaint_fp16.yaml && \
#         aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_lineart_fp16.yaml -d ${ROOT}/extensions/sd-webui-controlnet/models -o control_v11p_sd15_lineart_fp16.yaml && \
#         aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_mlsd_fp16.yaml -d ${ROOT}/extensions/sd-webui-controlnet/models -o control_v11p_sd15_mlsd_fp16.yaml && \
#         aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_normalbae_fp16.yaml -d ${ROOT}/extensions/sd-webui-controlnet/models -o control_v11p_sd15_normalbae_fp16.yaml && \
#         aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_openpose_fp16.yaml -d ${ROOT}/extensions/sd-webui-controlnet/models -o control_v11p_sd15_openpose_fp16.yaml && \
#         aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_scribble_fp16.yaml -d ${ROOT}/extensions/sd-webui-controlnet/models -o control_v11p_sd15_scribble_fp16.yaml && \
#         aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_seg_fp16.yaml -d ${ROOT}/extensions/sd-webui-controlnet/models -o control_v11p_sd15_seg_fp16.yaml && \
#         aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_softedge_fp16.yaml -d ${ROOT}/extensions/sd-webui-controlnet/models -o control_v11p_sd15_softedge_fp16.yaml && \
#         aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15s2_lineart_anime_fp16.yaml -d ${ROOT}/extensions/sd-webui-controlnet/models -o control_v11p_sd15s2_lineart_anime_fp16.yaml && \
#         aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11f1e_sd15_tile_fp16.yaml -d ${ROOT}/extensions/sd-webui-controlnet/models -o control_v11f1e_sd15_tile_fp16.yaml && \
#         aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_style_sd14v1.pth -d ${ROOT}/extensions/sd-webui-controlnet/models -o t2iadapter_style_sd14v1.pth && \
#         aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_sketch_sd14v1.pth -d ${ROOT}/extensions/sd-webui-controlnet/models -o t2iadapter_sketch_sd14v1.pth && \
#         aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_seg_sd14v1.pth -d ${ROOT}/extensions/sd-webui-controlnet/models -o t2iadapter_seg_sd14v1.pth && \
#         aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_openpose_sd14v1.pth -d ${ROOT}/extensions/sd-webui-controlnet/models -o t2iadapter_openpose_sd14v1.pth && \
#         aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_keypose_sd14v1.pth -d ${ROOT}/extensions/sd-webui-controlnet/models -o t2iadapter_keypose_sd14v1.pth && \
#         aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_depth_sd14v1.pth -d ${ROOT}/extensions/sd-webui-controlnet/models -o t2iadapter_depth_sd14v1.pth && \
#         aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_color_sd14v1.pth -d ${ROOT}/extensions/sd-webui-controlnet/models -o t2iadapter_color_sd14v1.pth && \
#         aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_canny_sd14v1.pth -d ${ROOT}/extensions/sd-webui-controlnet/models -o t2iadapter_canny_sd14v1.pth && \
#         aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_canny_sd15v2.pth -d ${ROOT}/extensions/sd-webui-controlnet/models -o t2iadapter_canny_sd15v2.pth && \
#         aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_depth_sd15v2.pth -d ${ROOT}/extensions/sd-webui-controlnet/models -o t2iadapter_depth_sd15v2.pth && \
#         aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_sketch_sd15v2.pth -d ${ROOT}/extensions/sd-webui-controlnet/models -o t2iadapter_sketch_sd15v2.pth && \
#         aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_zoedepth_sd15v1.pth -d ${ROOT}/extensions/sd-webui-controlnet/models -o t2iadapter_zoedepth_sd15v1.pth && \
#         aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/sd15/resolve/main/v1-5-pruned-emaonly.ckpt -d ${ROOT}/models/Stable-diffusion -o v1-5-pruned-emaonly.ckpt  


COPY --from=download /repositories/ ${ROOT}/repositories/
# COPY --from=download /model.safetensors /model.safetensors
RUN mkdir ${ROOT}/interrogate && cp ${ROOT}/repositories/clip-interrogator/data/* ${ROOT}/interrogate
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r ${ROOT}/repositories/CodeFormer/requirements.txt

# Install Python dependencies (Worker Template)
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

ADD src .

# COPY builder/cache.py /stable-diffusion-webui/cache.py
# RUN cd /stable-diffusion-webui && python cache.py --use-cpu=all --ckpt /model.safetensors

# Cleanup section (Worker Template)
RUN apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Set permissions and specify the command to run
RUN chmod +x /start.sh
CMD /start.sh
