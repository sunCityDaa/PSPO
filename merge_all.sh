#!/bin/bash

# 设置环境变量（可选）
export CUDA_VISIBLE_DEVICES=""  # 用CPU合并，或者指定卡，如 "0"

# 路径配置
BASE_MODEL="/data/model/Qwen2.5-Math-7B-Instruct"
LORA_BASE="/data/ER-GRPO/Qwen2.5-Math-7B-Instruct-data"
MERGED_BASE="/data/ER-GRPO/Qwen2.5-Math-7B-Instruct-data/Merged"

# LoRA 子目录数组（只需修改这一行，填你要 merge 的 LoRA 子文件夹名）
LORA_NAMES=("GRPO" "DrGRPO" "ER-GRPO-alpha99" "PER-GRPO" "ERPER-GRPO-alpha99")

# 遍历合并
for LORA_NAME in "${LORA_NAMES[@]}"
do
    echo "Merging $LORA_NAME..."
    LORA_PATH="${LORA_BASE}/${LORA_NAME}/checkpoint-100"
    MERGED_PATH="${MERGED_BASE}/${LORA_NAME}"

    python ./src/open_r1/merge.py \
        --base_model_dir "$BASE_MODEL" \
        --lora_model_dir "$LORA_PATH" \
        --merged_model_dir "$MERGED_PATH" \
        --torch_dtype "bfloat16" \
        --device_map "cpu"

    echo "Finished merging $LORA_NAME → $MERGED_PATH"
    echo "----------------------------------------"
done
