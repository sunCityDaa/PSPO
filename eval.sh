#!/bin/sh
# Base model configuration
#  如果用两个卡 需要export VLLM_WORKER_MULTIPROC_METHOD=spawn


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
            50)  echo "46d398a66c2fb69144a9f1a5a3403c2c58323359" ;;
            100) echo "c1b6fd9c7671d5e03227a61617ba3b2ade3122fa" ;;
            150) echo "67ef068f9d1283c1ae1e786b6d6b083bb0f2abda" ;;
            200) echo "cb20ade5820092f60e2729235e855f148371933d" ;;
            250) echo "7ff15366c653975ddd767b2a62385432cbc1bb33" ;;
            300) echo "4c49019f2369261b8ef49c5ce09e0a1e8990c7ed" ;;
            350) echo "dcf5c44784695307ec21d20be348400848be151b" ;;
            400) echo "0100f8beed8c73f9582dc766bd850b5251b7ef16" ;;
            450) echo "18d72d2da2e4d10ed7dc4de71a443b71c59ac274" ;;
            500) echo "36e774021fb940b7be7916a79578485351374abb" ;;
            *) echo "unknown" ;;
        esac
    # Experiment 2 revisions
    # DRGRPO
    elif [ "$exp" = "2" ]; then
        case $step in
            # 50)  echo "3fa00db18c41611c1ae70d1e6b5f668bb2d8592e" ;;
            # 100) echo "c609c349f4b2abc1ec14dcb496715559ad9dede6" ;;
            # 150) echo "1b8635e91636c9029b6d55b26829e2f749003392" ;;
            # 200) echo "2d942ed28c4b0a1b3373376590a90248347b5d57" ;;
            # 250) echo "9ae3908996d6495cd8725e370ddb92af24559dc1" ;;
            # 300) echo "dfa856ecd8941aa6bb22712acd33faae2a388459" ;;
            # 350) echo "02a7a65eed7887ba267912c136405ce9923e9e05" ;;
            # 400) echo "e43e96f04acdf7e1df3c1f40e4feba569f86ece8" ;;
            # 450) echo "95fe21a893b3bc9a40332ee9f17badee09a835a1" ;;
            # 500) echo "ed5b5f719716abf240ef0f7e13d31f72c28df32c" ;;
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
            50)  echo "e06b6eca6e61bc88b5541e692fe3ca89c152ad1f" ;;
            100) echo "7a4eff223674072e4d75cf8b63022f98069f8f65" ;;
            150) echo "c7d3a5905f90d6a9185b432e35c74f013b34c657" ;;
            200) echo "f7904a88368952a8a434fcf10ff2495a472135e3" ;;
            250) echo "557d5384b56f270eb3f4f12381fa054df8e6d271" ;;
            300) echo "d2d336317caf3e29ca012cf00ef33820c00f243c" ;;
            350) echo "9eef018b26f9e1c9adfb54f1759da8b4f1dbe2e2" ;;
            400) echo "45bdb351cd2c828af00b0bd69d3f325c9b4870ad" ;;
            450) echo "389f81cd1133887fc128c854c0c3f83abdead699" ;;
            500) echo "14f6576f0b3340a127f87c28028486eb686a5f75" ;;
            *) echo "unknown" ;;
        esac
    # ERGRPO-STD
    elif [ "$exp" = "4" ]; then
        case $step in
            # 50)  echo "/data/ER-GRPO/data/ER-GRPO_std/checkpoint-50" ;;
            # 100) echo "/data/ER-GRPO/data/ER-GRPO_std/checkpoint-100" ;;
            150) echo "/data/ER-GRPO/data/ER-GRPO_std/checkpoint-150" ;;
            200) echo "/data/ER-GRPO/data/ER-GRPO_std/checkpoint-200" ;;
            250) echo "/data/ER-GRPO/data/ER-GRPO_std/checkpoint-250" ;;
            300) echo "/data/ER-GRPO/data/ER-GRPO_std/checkpoint-300" ;;
            350) echo "/data/ER-GRPO/data/ER-GRPO_std/checkpoint-350" ;;
            400) echo "/data/ER-GRPO/data/ER-GRPO_std/checkpoint-400" ;;
            450) echo "/data/ER-GRPO/data/ER-GRPO_std/checkpoint-450" ;;
            500) echo "/data/ER-GRPO/data/ER-GRPO_std/checkpoint-500" ;;
            *) echo "unknown" ;;
        esac
    # ERGRPO 默认reward alpha = 0.8
    elif [ "$exp" = "5" ]; then
        case $step in
            50)  echo "/data/ER-GRPO/data/ER-GRPO/checkpoint-50" ;;
            100) echo "/data/ER-GRPO/data/ER-GRPO/checkpoint-100" ;;
            150) echo "/data/ER-GRPO/data/ER-GRPO/checkpoint-150" ;;
            200) echo "/data/ER-GRPO/data/ER-GRPO/checkpoint-200" ;;
            250) echo "/data/ER-GRPO/data/ER-GRPO/checkpoint-250" ;;
            300) echo "/data/ER-GRPO/data/ER-GRPO/checkpoint-300" ;;
            350) echo "/data/ER-GRPO/data/ER-GRPO/checkpoint-350" ;;
            400) echo "/data/ER-GRPO/data/ER-GRPO/checkpoint-400" ;;
            450) echo "/data/ER-GRPO/data/ER-GRPO/checkpoint-450" ;;
            500) echo "/data/ER-GRPO/data/ER-GRPO/checkpoint-500" ;;
            *) echo "unknown" ;;
        esac
     # ERGRPO  reward alpha = 0.9
    elif [ "$exp" = "6" ]; then
        case $step in
            50)  echo "/data/ER-GRPO/data/ER-GRPO-alpha90/checkpoint-50" ;;
            100) echo "/data/ER-GRPO/data/ER-GRPO-alpha90/checkpoint-100" ;;
            150) echo "/data/ER-GRPO/data/ER-GRPO-alpha90/checkpoint-150" ;;
            200) echo "/data/ER-GRPO/data/ER-GRPO-alpha90/checkpoint-200" ;;
            250) echo "/data/ER-GRPO/data/ER-GRPO-alpha90/checkpoint-250" ;;
            300) echo "/data/ER-GRPO/data/ER-GRPO-alpha90/checkpoint-300" ;;
            350) echo "/data/ER-GRPO/data/ER-GRPO-alpha90/checkpoint-350" ;;
            400) echo "/data/ER-GRPO/data/ER-GRPO-alpha90/checkpoint-400" ;;
            450) echo "/data/ER-GRPO/data/ER-GRPO-alpha90/checkpoint-450" ;;
            500) echo "/data/ER-GRPO/data/ER-GRPO-alpha90/checkpoint-500" ;;
            *) echo "unknown" ;;
        esac
     # ERGRPO  reward alpha = 0.1
    elif [ "$exp" = "7" ]; then
        case $step in
            50)  echo "/data/ER-GRPO/data/ER-GRPO-alpha10/checkpoint-50" ;;
            100) echo "/data/ER-GRPO/data/ER-GRPO-alpha10/checkpoint-100" ;;
            150) echo "/data/ER-GRPO/data/ER-GRPO-alpha10/checkpoint-150" ;;
            200) echo "/data/ER-GRPO/data/ER-GRPO-alpha10/checkpoint-200" ;;
            250) echo "/data/ER-GRPO/data/ER-GRPO-alpha10/checkpoint-250" ;;
            300) echo "/data/ER-GRPO/data/ER-GRPO-alpha10/checkpoint-300" ;;
            350) echo "/data/ER-GRPO/data/ER-GRPO-alpha10/checkpoint-350" ;;
            400) echo "/data/ER-GRPO/data/ER-GRPO-alpha10/checkpoint-400" ;;
            450) echo "/data/ER-GRPO/data/ER-GRPO-alpha10/checkpoint-450" ;;
            500) echo "/data/ER-GRPO/data/ER-GRPO-alpha10/checkpoint-500" ;;
            *) echo "unknown" ;;
        esac
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
        # 2) echo "300 350 400 450 500" ;;
        3) echo "50 100 150 200 250 300 350 400 450 500" ;;
        4) echo "50 100 150 200 250 300 350 400 450 500" ;;
        # 4) echo "150 200 250 300 350 400 450 500" ;;
        5) echo "50 100 150 200 250 300 350 400 450 500" ;;
        6) echo "50 100 150 200 250 300 350 400 450 500" ;;
        7) echo "50 100 150 200 250 300 350 400 450 500" ;;
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
    output_dir="logs/evals/Exp${experiment}_${step}"
    
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
        if [ "$exp_num" = "1" ] || [ "$exp_num" = "2" ] || [ "$exp_num" = "3" ] || [ "$exp_num" = "4" ] || [ "$exp_num" = "5" ] || [ "$exp_num" = "6"] || [ "$exp_num" = "7" ] || [ "$exp_num" = "8" ] || [ "$exp_num" = "9" ]; then
            run_experiment "$exp_num"
        else
            echo "Error: Experiment $exp_num not defined"
        fi
    done
}

# Execute main function with command line arguments
main "$@"
