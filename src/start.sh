#!/bin/bash

echo "Worker Initiated"
# aria2c --console-log-level=error -c -x 16 -s 16 -k 1M http://172.17.0.1:8081/v1-5-pruned-emaonly.ckpt -d /stable-diffusion-webui/models/Stable-diffusion/ &

# https://civitai.com/api/download/models/6424

path="/runpod-volume/stable-diffusion-webui"

if [ -d $path ]; then
    
else
    cp -r /stable-diffusion-webui $path
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://civitai.com/api/download/models/11745 -d $path/models/Stable-diffusion/ &
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M -i /controlnet_model.txt -d $path/extensions/sd-webui-controlnet/models/
fi

echo "Starting WebUI API"
python3 -u $path/launch.py --skip-python-version-check --skip-torch-cuda-test --lowram --opt-sdp-attention --disable-safe-unpickle --port 3000 --api --skip-version-check  --no-hashing --no-download-sd-model &

python3 -u /rp_handler.py 

# python3 -u /rp_handler.py  --rp_serve_api
