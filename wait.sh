#!/bin/bash

# 在这里直接指定要等待的进程 PID
# PID=3499799  

# echo "等待进程 PID=$PID 结束..."

# while kill -0 "$PID" 2>/dev/null; do
#     sleep 5
# done


# echo "进程 PID=$PID 结束..."



# 1.5B


ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file src/open_r1/trl/accelerate_configs/zero2.yaml \
--num_processes=2 \
src/open_r1/ergrpo.py \
--config recipes/per_other.yaml

# ACCELERATE_LOG_LEVEL=info accelerate launch \
# --config_file src/open_r1/trl/accelerate_configs/zero2.yaml \
# --num_processes=2 \
# src/open_r1/ergrpo.py \
# --config recipes/dra_er_grpo.yaml

ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file src/open_r1/trl/accelerate_configs/zero2.yaml \
--num_processes=2 \
src/open_r1/ergrpo.py \
--config recipes/EMA_alpha99.yaml

# LERATE_LOG_LEVEL=info accelerate launch \
# --config_file src/open_r1/trl/accelerate_configs/zero2.yaml \
# --num_processes=2 \
# src/open_r1/grpo.py \
# --config recipes/dra_grpo.yaml