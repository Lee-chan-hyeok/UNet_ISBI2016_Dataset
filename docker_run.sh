docker run -it \
    --name unet_chlee \
    --shm-size 2g \
    -v ./:/workspace \
    unet_chlee \
    /bin/bash