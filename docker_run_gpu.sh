docker run -it --rm \
    --gpus all \
    --shm-size=8g \
    -v ./:/workspace \
    --name unet_chlee \
    unet_chlee:latest \
    /bin/bash