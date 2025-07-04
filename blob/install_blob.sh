curl -sSL -O https://packages.microsoft.com/config/ubuntu/22.04/packages-microsoft-prod.deb

sudo dpkg -i packages-microsoft-prod.deb

sudo apt-get update 

sudo apt-get install azure-cli -y

sudo apt-get install blobfuse2 -y

umount /mnt/lingjiejiang