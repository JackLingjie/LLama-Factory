#!/bin/bash
set -x

# kill sleep job
kill -9 $(pgrep -f run_gpu)

# kill port 12335 job
# 指定端口号
PORT=12335

# 确保 lsof 已安装
if ! command -v lsof &> /dev/null; then
    echo "Installing lsof..."
    apt update && apt install -y lsof
fi

# 终止运行在指定端口的进程
if lsof -t -i :$PORT &> /dev/null; then
    echo "Killing processes on port $PORT..."
    kill -9 $(lsof -t -i :$PORT)
else
    echo "No processes running on port $PORT."
fi

echo "Script executed successfully on local machine!"

# kill llama running job
kill -9 $(pgrep -f LLaMA)

kill -9 $(pgrep -f Qwen)
kill -9 $(pgrep -f torchrun)

git pull