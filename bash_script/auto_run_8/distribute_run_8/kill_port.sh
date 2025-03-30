#!/bin/bash

# 节点列表文件路径
NODES_FILE="bash_script/auto_run_8/distribute_run_8/node_config/nodes.txt"  # 节点列表文件

# 确保节点列表文件存在
if [ ! -f "$NODES_FILE" ]; then
    echo "Error: $NODES_FILE not found!"
    exit 1
fi

# 检查节点文件内容是否包含合法节点名称
if ! grep -q "node-" "$NODES_FILE"; then
    echo "Error: $NODES_FILE does not contain valid node names!"
    exit 1
fi
# sudo kill -9 $(sudo lsof -t -i :12335)
# 使用 pdsh 在所有节点上安装 lsof 并终止指定端口的进程
pdsh -R ssh -w ^$NODES_FILE bash -c "'
set -e  # 如果有任何命令失败，退出脚本

# 安装 lsof
if ! command -v lsof &> /dev/null; then
    echo \"Installing lsof...\"
    sudo apt update && sudo apt install -y lsof
fi

# 终止运行在端口 12335 的进程
if sudo lsof -t -i :12335 &> /dev/null; then
    echo \"Killing processes on port 12335...\"
    sudo kill -9 \$(sudo lsof -t -i :12335)
else
    echo \"No processes running on port 12335.\"
fi
'"

echo "Commands executed successfully on all nodes!"
