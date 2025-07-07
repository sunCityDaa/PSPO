#!/bin/bash

# # 设置进程 ID
# PID=3736301

# echo "等待进程 PID=$PID 结束..."

# # 循环检测进程是否还存在
# while kill -0 "$PID" 2>/dev/null; do
#     sleep 5
# done

# echo "进程 $PID 已结束，开始执行后续命令..."
# # 执行后续命令




ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file src/open_r1/trl/accelerate_configs/zero2.yaml \
--num_processes=2 \
src/open_r1/ergrpo.py \
--config recipes/EMA_alpha10.yaml

ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file src/open_r1/trl/accelerate_configs/zero2.yaml \
--num_processes=2 \
src/open_r1/ergrpo.py \
--config recipes/EMA_alpha30.yaml

ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file src/open_r1/trl/accelerate_configs/zero2.yaml \
--num_processes=2 \
src/open_r1/ergrpo.py \
--config recipes/EMA_alpha50.yaml

ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file src/open_r1/trl/accelerate_configs/zero2.yaml \
--num_processes=2 \
src/open_r1/ergrpo.py \
--config recipes/EMA_alpha70.yaml

ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file src/open_r1/trl/accelerate_configs/zero2.yaml \
--num_processes=2 \
src/open_r1/ergrpo.py \
--config recipes/EMA_alpha90.yaml

ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file src/open_r1/trl/accelerate_configs/zero2.yaml \
--num_processes=2 \
src/open_r1/ergrpo.py \
--config recipes/EMA_alpha99.yaml

ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file src/open_r1/trl/accelerate_configs/zero2.yaml \
--num_processes=2 \
src/open_r1/ergrpo.py \
--config recipes/per_other.yaml

sh eval.sh 5