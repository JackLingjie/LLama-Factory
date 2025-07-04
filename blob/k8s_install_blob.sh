#!/bin/bash
cd /workspace/LLama-Factoryblob

curl -sSL -O https://packages.microsoft.com/config/ubuntu/22.04/packages-microsoft-prod.deb

dpkg -i packages-microsoft-prod.deb

apt-get update 

apt-get install azure-cli -y

apt-get install blobfuse2 -y

umount /mnt/lingjiejiang


set -x

CN=lingjiejiang

MOUNT_DIR=/mnt/${CN}

mkdir -p ${MOUNT_DIR}
chown $USER ${MOUNT_DIR}

blobfuse2 mount ${MOUNT_DIR} --config-file=config.yaml