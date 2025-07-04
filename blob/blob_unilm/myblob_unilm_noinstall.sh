#!/bin/bash
set -x

# bash blob/install_blob.sh

CN=msranlp

MOUNT_DIR=/mnt/${CN}

umount ${MOUNT_DIR} || true
sudo mkdir -p ${MOUNT_DIR}
sudo chown $USER ${MOUNT_DIR}

blobfuse2 mount ${MOUNT_DIR} --config-file=blob/blob_unilm/config.yaml