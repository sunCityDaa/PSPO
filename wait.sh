#!/bin/bash

# 设置进程 ID
PID=3736301

echo "等待进程 PID=$PID 结束..."

# 循环检测进程是否还存在
while kill -0 "$PID" 2>/dev/null; do
    sleep 5
done

echo "进程 $PID 已结束，开始执行后续命令..."
# 执行后续命令
sh eval.sh 5
