

#!/bin/bash

# 定义节点配置文件的路径
NODES_FILE="bash_script/auto_run_8/distribute_run_8/node_config/nodes.txt"

script=$(cat << 'EOF'
cd /tmp
export PATH=$PATH:/home/aiscuser/.local/bin

git clone https://github.com/JackLingjie/LLama-Factory.git
cd LLama-Factory
git checkout -b debug origin/debug

pip install tensorboard


pip uninstall deepspeed -y
pip install deepspeed==0.15.4
pip install deepspeed==0.15.4

pip install -U flash-attn==2.7.2.post1 --no-build-isolation
pip install --user -e ".[torch,metrics]"
pip install deepspeed==0.15.4
pip install wandb
EOF
)

pdsh -R ssh -w ^$NODES_FILE "bash -c '$script'"