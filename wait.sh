#!/bin/bash

# # # 设置进程 ID


# ACCELERATE_LOG_LEVEL=info accelerate launch \
# --config_file src/open_r1/trl/accelerate_configs/zero2.yaml \
# --num_processes=2 \
# src/open_r1/ergrpo.py \
# --config recipes/EMA_alpha99.yaml

ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file src/open_r1/trl/accelerate_configs/zero2.yaml \
--num_processes=2 \
src/open_r1/ergrpo.py \
--config recipes/dra_er_grpo.yaml





ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file src/open_r1/trl/accelerate_configs/zero2.yaml \
--num_processes=2 \
src/open_r1/ergrpo.py \
--config recipes/1.5B/EMA_alpha99.yaml

ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file src/open_r1/trl/accelerate_configs/zero2.yaml \
--num_processes=2 \
src/open_r1/ergrpo.py \
--config recipes/1.5B/dra_er_grpo.yaml

ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file src/open_r1/trl/accelerate_configs/zero2.yaml \
--num_processes=2 \
src/open_r1/ergrpo.py \
--config recipes/1.5B_inst/EMA_alpha99.yaml

ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file src/open_r1/trl/accelerate_configs/zero2.yaml \
--num_processes=2 \
src/open_r1/ergrpo.py \
--config recipes/1.5B_inst/dra_er_grpo.yaml
