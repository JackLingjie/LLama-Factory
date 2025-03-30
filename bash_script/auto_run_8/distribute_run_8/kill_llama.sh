#!/bin/bash

# 定义节点配置文件的路径
NODES_FILE="bash_script/auto_run_8/distribute_run_8/node_config/nodes.txt"

script=$(cat << 'EOF'
kill -9 $(pgrep -f LLaMA)

kill -9 $(pgrep -f Qwen)
kill -9 $(pgrep -f torchrun)
EOF
)

pdsh -R ssh -w ^$NODES_FILE "bash -c '$script'"