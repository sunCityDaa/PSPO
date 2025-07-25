#!/bin/sh
# Base model configuration
#  如果用两个卡 需要

# export VLLM_WORKER_MULTIPROC_METHOD=spawn 
# export VLLM_ATTENTION_BACKEND=XFORMERS 


# 从huggingface上下模型的配置
# MODEL="lalalaDa/GRPO"
# BASE_MODEL_ARGS="model_name=$MODEL,\
# dtype=bfloat16,\
# max_model_length=32768,\
# gpu_memory_utilization=0.8,\
# tensor_parallel_size=2,\
# max_num_batched_tokens=4096,\
# generation_parameters={max_new_tokens:3584,temperature:0.7,top_p:1.0}"




BASE_MODEL_ARGS="dtype=bfloat16,\
max_model_length=32768,\
gpu_memory_utilization=0.8,\
tensor_parallel_size=2,\
max_num_batched_tokens=32768,\
generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"


# Qwen1.5B

# BASE_MODEL_ARGS="dtype=bfloat16,\
# max_model_length=4096,\
# gpu_memory_utilization=0.8,\
# tensor_parallel_size=2,\
# max_num_batched_tokens=4096,\
# generation_parameters={max_new_tokens:3096,temperature:0.6,top_p:0.95}"


# Define evaluation tasks
TASKS="aime24 math_500 amc23 minerva olympiadbench"
# TASKS="aime24 math_500 amc23"


# Function to get revision for a given experiment and step
get_revision() {
    exp=$1
    step=$2
    
    # Experiment 1 revisions
    # BNPO
    if [ "$exp" = "1" ]; then
        case $step in
            50)  echo "/data/ER-GRPO/data/BNPO/checkpoint-50" ;;
            100) echo "/data/ER-GRPO/data/BNPO/checkpoint-100" ;;
            150) echo "/data/ER-GRPO/data/BNPO/checkpoint-150" ;;
            200) echo "/data/ER-GRPO/data/BNPO/checkpoint-200" ;;
            250) echo "/data/ER-GRPO/data/BNPO/checkpoint-250" ;;
            300) echo "/data/ER-GRPO/data/BNPO/checkpoint-300" ;;
            350) echo "/data/ER-GRPO/data/BNPO/checkpoint-350" ;;
            400) echo "/data/ER-GRPO/data/BNPO/checkpoint-400" ;;
            450) echo "/data/ER-GRPO/data/BNPO/checkpoint-450" ;;
            500) echo "/data/ER-GRPO/data/BNPO/checkpoint-500" ;;
            *) echo "unknown" ;;
        esac
    # Experiment 2 revisions
    # DRGRPO
    elif [ "$exp" = "2" ]; then
        case $step in
            50)  echo "/data/ER-GRPO/data/Dr_GRPO/checkpoint-50" ;;
            100) echo "/data/ER-GRPO/data/Dr_GRPO/checkpoint-100" ;;
            150) echo "/data/ER-GRPO/data/Dr_GRPO/checkpoint-150" ;;
            200) echo "/data/ER-GRPO/data/Dr_GRPO/checkpoint-200" ;;
            250) echo "/data/ER-GRPO/data/Dr_GRPO/checkpoint-250" ;;
            300) echo "/data/ER-GRPO/data/Dr_GRPO/checkpoint-300" ;;
            350) echo "/data/ER-GRPO/data/Dr_GRPO/checkpoint-350" ;;
            400) echo "/data/ER-GRPO/data/Dr_GRPO/checkpoint-400" ;;
            450) echo "/data/ER-GRPO/data/Dr_GRPO/checkpoint-450" ;;
            500) echo "/data/ER-GRPO/data/Dr_GRPO/checkpoint-500" ;;
            *) echo "unknown" ;;
        esac
    # Experiment 3 revisions
    # GRPO
    elif [ "$exp" = "3" ]; then
        case $step in
            50)  echo "/data/ER-GRPO/data/GRPO/checkpoint-50" ;;
            100) echo "/data/ER-GRPO/data/GRPO/checkpoint-100" ;;
            150) echo "/data/ER-GRPO/data/GRPO/checkpoint-150" ;;
            200) echo "/data/ER-GRPO/data/GRPO/checkpoint-200" ;;
            250) echo "/data/ER-GRPO/data/GRPO/checkpoint-250" ;;
            300) echo "/data/ER-GRPO/data/GRPO/checkpoint-300" ;;
            350) echo "/data/ER-GRPO/data/GRPO/checkpoint-350" ;;
            400) echo "/data/ER-GRPO/data/GRPO/checkpoint-400" ;;
            450) echo "/data/ER-GRPO/data/GRPO/checkpoint-450" ;;
            500) echo "/data/ER-GRPO/data/GRPO/checkpoint-500" ;;
            *) echo "unknown" ;;
        esac
    # ERGRPO-STD
    elif [ "$exp" = "4" ]; then
        case $step in
            50)  echo "/data/ER-GRPO/Qwen-Math-1.5B-data/GRPO/checkpoint-50" ;;
            100) echo "/data/ER-GRPO/Qwen-Math-1.5B-data/GRPO/checkpoint-100" ;;
            150) echo "/data/ER-GRPO/Qwen-Math-1.5B-data/GRPO/checkpoint-150" ;;
            200) echo "/data/ER-GRPO/Qwen-Math-1.5B-data/GRPO/checkpoint-200" ;;
            250) echo "/data/ER-GRPO/Qwen-Math-1.5B-data/GRPO/checkpoint-250" ;;
            300) echo "/data/ER-GRPO/Qwen-Math-1.5B-data/GRPO/checkpoint-300" ;;
            350) echo "/data/ER-GRPO/Qwen-Math-1.5B-data/GRPO/checkpoint-350" ;;
            # 400) echo "/data/ER-GRPO/data/ER-GRPO_std/checkpoint-400" ;;
            # 450) echo "/data/ER-GRPO/data/ER-GRPO_std/checkpoint-450" ;;
            # 500) echo "/data/ER-GRPO/data/ER-GRPO_std/checkpoint-500" ;;
            *) echo "unknown" ;;
        esac
    
    elif [ "$exp" = "5" ]; then
        case $step in
            50)  echo "/data/ER-GRPO/data/ER-GRPO-alpha99-newSTD/checkpoint-50" ;;
            # 100)  echo "/data/model/Qwen2.5-Math-7B-Instruct" ;;
            # 50)  echo "/data/ER-GRPO/data/ER-GRPO-alpha10/checkpoint-50" ;;
            # 100) echo "/data/ER-GRPO/data/ER-GRPO-alpha30/checkpoint-50" ;;
            # 150) echo "/data/ER-GRPO/data/ER-GRPO-alpha50/checkpoint-50" ;;
            # 200) echo "/data/ER-GRPO/data/ER-GRPO-alpha70/checkpoint-50" ;;
            # 250) echo "/data/ER-GRPO/data/ER-GRPO-alpha90/checkpoint-50" ;;
            # 300) echo "/data/ER-GRPO/data/ER-GRPO-alpha99/checkpoint-50" ;;
            # 350) echo "/data/ER-GRPO/data/PER-Other-GRPO/checkpoint-50" ;;
            # 400) echo "/data/ER-GRPO/data/new-ER-GRPO-alpha10/checkpoint-400" ;;
            # 450) echo "/data/ER-GRPO/data/new-ER-GRPO-alpha10/checkpoint-450" ;;
            # 500) echo "/data/ER-GRPO/data/new-ER-GRPO-alpha10/checkpoint-500" ;;
            *) echo "unknown" ;;
        esac
    # alpha 消融实验，Num_genrations
    elif [ "$exp" = "6" ]; then
        case $step in
            50)  echo "/data/ER-GRPO/data/ERPER-GRPO-alpha99/checkpoint-50" ;;
            100) echo "/data/ER-GRPO/data/ERPER-GRPO-alpha99/checkpoint-100" ;;
            150) echo "/data/ER-GRPO/data/ERPER-GRPO-alpha99/checkpoint-150" ;;
            200) echo "/data/ER-GRPO/data/ERPER-GRPO-alpha99/checkpoint-200" ;;
            250) echo "/data/ER-GRPO/data/ERPER-GRPO-alpha99/checkpoint-250" ;;
            300) echo "/data/ER-GRPO/data/ERPER-GRPO-alpha99/checkpoint-300" ;;
            350) echo "/data/ER-GRPO/data/ERPER-GRPO-alpha99/checkpoint-350" ;;
            400) echo "/data/ER-GRPO/data/ERPER-GRPO-alpha99/checkpoint-400" ;;
            450) echo "/data/ER-GRPO/data/ERPER-GRPO-alpha99/checkpoint-450" ;;
            500) echo "/data/ER-GRPO/data/ERPER-GRPO-alpha99/checkpoint-500" ;;
            *) echo "unknown" ;;
        esac
     # ERGRPO  reward alpha = 0.1
    elif [ "$exp" = "7" ]; then
        case $step in
            # 50)  echo "/data/ER-GRPO/Qwen2.5-Math-7B-data/Merged/ER-GRPO-alpha99" ;;
            # 100) echo "/data/ER-GRPO/Qwen2.5-Math-7B-data/Merged/DrGRPO" ;;
            # 150) echo "/data/ER-GRPO/Qwen2.5-Math-7B-data/Merged/ERPER-GRPO-alpha99" ;;
            # 200) echo "/data/ER-GRPO/Qwen2.5-Math-7B-data/Merged/PER-GRPO" ;;
            250) echo "/data/ER-GRPO/Qwen2.5-Math-7B-data/Merged/GRPO" ;;  
            300) echo "/data/ER-GRPO/Qwen2.5-Math-7B-Instruct-data/Merged/ER-GRPO-alpha99" ;;
            # 350) echo "/data/ER-GRPO/Qwen2.5-Math-7B-Instruct-data/Merged/DrGRPO" ;;
            # 400) echo "/data/ER-GRPO/Qwen2.5-Math-7B-Instruct-data/Merged/ERPER-GRPO-alpha99" ;;
            450) echo "/data/ER-GRPO/Qwen2.5-Math-7B-Instruct-data/Merged/PER-GRPO" ;;
            500) echo "/data/ER-GRPO/Qwen2.5-Math-7B-Instruct-data/Merged/GRPO" ;;
            *) echo "unknown" ;;
        esac
    # ERGRPO  
    elif [ "$exp" = "8" ]; then
        case $step in
            50)  echo "/data/ER-GRPO/data/new-ER-GRPO-alpha90/checkpoint-50" ;;
            100) echo "/data/ER-GRPO/data/new-ER-GRPO-alpha90/checkpoint-100" ;;
            150) echo "/data/ER-GRPO/data/new-ER-GRPO-alpha90/checkpoint-150" ;;
            200) echo "/data/ER-GRPO/data/new-ER-GRPO-alpha90/checkpoint-200" ;;
            250) echo "/data/ER-GRPO/data/new-ER-GRPO-alpha90/checkpoint-250" ;;
            300) echo "/data/ER-GRPO/data/new-ER-GRPO-alpha90/checkpoint-300" ;;
            350) echo "/data/ER-GRPO/data/new-ER-GRPO-alpha90/checkpoint-350" ;;
            400) echo "/data/ER-GRPO/data/new-ER-GRPO-alpha90/checkpoint-400" ;;
            450) echo "/data/ER-GRPO/data/new-ER-GRPO-alpha90/checkpoint-450" ;;
            500) echo "/data/ER-GRPO/data/new-ER-GRPO-alpha90/checkpoint-500" ;;
            *) echo "unknown" ;;
        esac
    # 
    elif [ "$exp" = "9" ]; then
        case $step in
            50)  echo "/data/ER-GRPO/data/new-ERPER-GRPO-alpha90/checkpoint-50" ;;
            100) echo "/data/ER-GRPO/data/new-ERPER-GRPO-alpha90/checkpoint-100" ;;
            150) echo "/data/ER-GRPO/data/new-ERPER-GRPO-alpha90/checkpoint-150" ;;
            200) echo "/data/ER-GRPO/data/new-ERPER-GRPO-alpha90/checkpoint-200" ;;
            250) echo "/data/ER-GRPO/data/new-ERPER-GRPO-alpha90/checkpoint-250" ;;
            300) echo "/data/ER-GRPO/data/new-ERPER-GRPO-alpha90/checkpoint-300" ;;
            350) echo "/data/ER-GRPO/data/new-ERPER-GRPO-alpha90/checkpoint-350" ;;
            400) echo "/data/ER-GRPO/data/new-ERPER-GRPO-alpha90/checkpoint-400" ;;
            450) echo "/data/ER-GRPO/data/new-ERPER-GRPO-alpha90/checkpoint-450" ;;
            500) echo "/data/ER-GRPO/data/new-ERPER-GRPO-alpha90/checkpoint-500" ;;
            *) echo "unknown" ;;
        esac
    else
        echo "unknown"
    fi
}

# Function to get steps for a given experiment
get_steps() {
    exp=$1
    
    case $exp in
        1) echo "50 100 150 200 250 300 350 400 450 500" ;;
        2) echo "50 100 150 200 250 300 350 400 450 500" ;;
        3) echo "50 100 150 200 250 300 350 400 450 500" ;;
        # 4) echo "50 100 150 200 250 300 350 400 450 500" ;;
        4) echo "50 100 150 200 250 300 350" ;;
        # 5) echo "50 100 150 200 250 300 350" ;;
        5) echo "50" ;;
        6) echo "50 100 150 200 250 300 350 400 450 500" ;;
        # 7) echo "50 100 150 200 250 300 350 400 450 500" ;;
        7) echo "250 300 450 500" ;;
        8) echo "50 100 150 200 250 300 350 400 450 500" ;;
        9) echo "50 100 150 200 250 300 350 400 450 500" ;;
        *) echo "" ;;
    esac
}

# Function to run evaluations for a given step and revision
run_evaluation() {
    experiment=$1
    step=$2
    revision=$(get_revision "$experiment" "$step")
    output_dir="logs/evals-newSTD/Exp${experiment}_${step}"
    
    # Check if revision is valid
    if [ "$revision" = "unknown" ]; then
        echo "Error: Unknown revision for experiment $experiment, step $step"
        return 1
    fi
    
    # Set model args with the specific revision
    # 从huggingface上下模型的配置
    # model_args="$BASE_MODEL_ARGS,revision=$revision"

    # 从本地加载模型
    model_args="model_name=$revision,$BASE_MODEL_ARGS"

    
    echo "----------------------------------------"
    echo "Running evaluations for experiment $experiment, step $step"
    echo "Revision: rev${experiment}_${step} = $revision"
    echo "Output directory: $output_dir"
    
    # Create output directory if it doesn't exist
    mkdir -p "$output_dir"
    
    # Run evaluations for each task
    for task in $TASKS; do
        echo "Evaluating task: $task"
        lighteval vllm "$model_args" "custom|$task|0|0" \
            --custom-tasks src/open_r1/evaluate.py \
            --use-chat-template \
            --output-dir "$output_dir"
    done
    echo "----------------------------------------"
}

# Function to run an experiment
run_experiment() {
    exp_num=$1
    steps=$(get_steps "$exp_num")
    
    # Check if experiment exists
    if [ -z "$steps" ]; then
        echo "Error: Experiment $exp_num not defined"
        return 1
    fi
    
    echo "========================================"
    echo "Running Experiment $exp_num"
    echo "Steps: $steps"
    echo "========================================"
    
    # Run evaluation for each step in the experiment
    for step in $steps; do
        run_evaluation "$exp_num" "$step"
    done
}

# Function to list all available experiments and revisions
list_configurations() {
    echo "Available Experiments:"
    
    for exp_num in 1 2 3 4 5 6 7 8 9; do
        steps=$(get_steps "$exp_num")
        echo "  Experiment $exp_num: Steps = $steps"
        
        # List revisions for this experiment
        echo "  Revisions:"
        for step in $steps; do
            revision=$(get_revision "$exp_num" "$step")
            echo "    Step $step: rev${exp_num}_${step} = $revision"
        done
        echo ""
    done
}

# Main function to run experiments
main() {
    if [ $# -eq 0 ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
        echo "Usage: $0 [options] <experiment_number> [experiment_number2 ...]"
        echo "Options:"
        echo "  --list, -l    List all available experiments and revisions"
        echo "  --help, -h    Show this help message"
        echo ""
        list_configurations
        exit 0
    fi
    
    if [ "$1" = "--list" ] || [ "$1" = "-l" ]; then
        list_configurations
        exit 0
    fi
    
    for exp_num in "$@"; do
        if [ "$exp_num" = "1" ] || [ "$exp_num" = "2" ] || [ "$exp_num" = "3" ] || [ "$exp_num" = "4" ] || [ "$exp_num" = "5" ] || [ "$exp_num" = "6" ] || [ "$exp_num" = "7" ] || [ "$exp_num" = "8" ] || [ "$exp_num" = "9" ]; then
            run_experiment "$exp_num"
        else
            echo "Error: Experiment $exp_num not defined"
        fi
    done
}

# Execute main function with command line arguments
main "$@"
