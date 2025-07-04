#!/bin/bash
set -x

bash install_blob.sh

CN=lingjiejiang

MOUNT_DIR=/mnt/${CN}

sudo mkdir -p ${MOUNT_DIR}
sudo chown $USER ${MOUNT_DIR}

blobfuse2 mount ${MOUNT_DIR} --config-file=config.yaml