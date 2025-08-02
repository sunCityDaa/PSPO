# PSPO
Official code for the paper: PSPO: Prompt-Level Prioritization and Experience-Weighted Smoothing  for Efficient Policy Optimization 

Paper link (preprint): https://openreview.net/forum?id=k7Ipg88jzw&noteId=k7Ipg88jzw



> **Abstract.** Reinforcement Fine-tuning (RFT) methods such as Proximal Policy Optimization (PPO) and Group Relative Policy Optimization (GRPO) have demonstrated strong capabilities in aligning Large Language Models (LLMs) with human preferences. However, these approaches often suffer from limited data efficiency, necessitating extensive on-policy rollouts to maintain competitive performance. We propose PSPO (Prompt-Level \textbf{P}rioritization and Experience-Weighted \textbf{S}moothing for Efficient \textbf{P}olicy \textbf{O}ptimization), a lightweight yet effective enhancement to GRPO that improves training stability and sample efficiency through two complementary techniques. First, we introduce an experience-weighted reward smoothing mechanism, which uses exponential moving averages to track group-level reward statistics for each prompt. This enables more stable advantage estimation across training steps without storing entire trajectories, allowing the model to capture historical reward trends in a lightweight and memory-efficient manner. Second, we adopt a prompt-level prioritized sampling strategy, which is an online data selection method inspired by prioritized experience replay. It dynamically emphasizes higher-impact prompts based on their relative advantages, thereby improving data efficiency. Experiments on multiple mathematical reasoning benchmarks and models show that PSPO achieves comparable or better accuracy than GRPO, while significantly accelerating convergence, and maintaining low computational and memory overhead.

## Installation


Please follow the instructions of [Open-R1](https://github.com/huggingface/open-r1) to install the environment.
Log in to Hugging Face and Weights & Biases:
```
huggingface-cli login
wandb login

conda create -n openr1 python=3.11 && conda activate openr1 && pip install --upgrade pip

pip install vllm==0.8.5.post1

pip install setuptools && pip install flash-attn --no-build-isolation

pip install -r requirements.txt
```


**You can then remove ```trl``` package from the environment, because we customized it.**



## Training

### ERS
```
ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file src/open_r1/trl/accelerate_configs/zero2.yaml \
  --num_processes=2 \
  src/open_r1/grpo.py \
  --config recipes/dra_grpo.yaml 
                                          
```
### PPS
```
ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file src/open_r1/trl/accelerate_configs/zero2.yaml \
  --num_processes=2 \
  src/open_r1/ergrpo.py \
  --config recipes/per_other.yaml
```
### PPS+ERS(PSPO)
```
ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file src/open_r1/trl/accelerate_configs/zero2.yaml \
  --num_processes=2 \
  src/open_r1/ergrpo.py \
  --config recipes/dra_er_grpo.yaml
```

### GRPO
```
ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file src/open_r1/trl/accelerate_configs/zero2.yaml \
  --num_processes=2 \
  src/open_r1/grpo.py \
  --config recipes/dra_grpo.yaml
```
### DrGRPO
```
ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file src/open_r1/trl/accelerate_configs/zero2.yaml \
  --num_processes=2 \
  src/open_r1/grpo.py \
  --config recipes/dra_drgrpo.yaml
```

All weights will update to Huggingface.


## Merge lora and base model

```
bash merge_all.sh
```
## Inference via lighteval (Test multiple steps)
We have an evaluation template 

```
base eval.sh
```

## Thanks

Our code is built based on [Open-rs](https://github.com/knoveleng/open-rs). Thanks!



## Bug
```
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1       # 避免 RDMA 死锁问题
export NCCL_P2P_DISABLE=1      # 降低
export NCCL_DEBUG_SUBSYS=INIT,P2P
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64  # 如果是deepseek的话用这


dr_grpo-false 以及/data/ER-GRPO/logs/evals-drgrpo-false,这个不是真正的DrGRPO，advantage的计算和GRPO一样，而loss是dr_grpo

drgrpo也不是真正的drgrpo，因为最后的kl散度还在，且之前所有的STD的实验都错了，使用了标准差，按理来说应该是方差


export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=900
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1


export NCCL_SOCKET_TIMEOUT=5
export CUDA_LAUNCH_BLOCKING=1
```