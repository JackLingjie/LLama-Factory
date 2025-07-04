#!/bin/bash
set -x

umount /mnt/lingjiejiang

CN=lingjiejiang

MOUNT_DIR=/mnt/${CN}

sudo mkdir -p ${MOUNT_DIR}
sudo chown $USER ${MOUNT_DIR}

blobfuse2 mount ${MOUNT_DIR} --config-file=config.yaml