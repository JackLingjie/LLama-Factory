#!/bin/bash
set -x

curl -sSL -O https://packages.microsoft.com/config/ubuntu/22.04/packages-microsoft-prod.deb

dpkg -i packages-microsoft-prod.deb

apt-get update 

apt-get install azure-cli -y

apt-get install blobfuse2 -y

umount /mnt/lingjiejiang

# bash install_blob.sh

CN=lingjiejiang

MOUNT_DIR=/mnt/${CN}

mkdir -p ${MOUNT_DIR}
chown root ${MOUNT_DIR}

blobfuse2 mount ${MOUNT_DIR} --config-file=config.yaml