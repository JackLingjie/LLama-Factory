#!/bin/bash

# 定义节点配置文件的路径
NODES_FILE="bash_script/auto_run_8/distribute_run_8/node_config/nodes.txt"

script=$(cat << 'EOF'
kill -9 $(pgrep -f run_gpu)
EOF
)

pdsh -R ssh -w ^$NODES_FILE "bash -c '$script'"

pdsh -R ssh -w ^$NODES_FILE "pkill -9 -f Qwen"
