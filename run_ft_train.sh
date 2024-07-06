#!/bin/bash

set -x

export CUDA_VISIBLE_DEVICES="0"

source /home/hadoop-mtai-llms/.bashrc
source /usr/local/conda/etc/profile.d/conda.sh
conda activate wdnllm
which python

# Define a function to kill processes by name
function killall {
    echo `ps -ef | grep $1 | grep -v grep | awk '{print $2}'`
    ps -ef | grep $1 | grep -v grep | awk '{print $2}' | xargs kill -9
}

# Set working directory
WORK_DIR="/users/wangdongnian/code/qwen2_ft"
cd ${WORK_DIR}
echo `pwd`

# Common parameters
N_NODES=1
N_GPUS=8
MBS=1 # Single GPU batch size
GAS=1 # Gradient accumulation steps
GRAD_CLIP=1     # Gradient clipping
RANK=0
MASTER_ADDR=`hostname -i`
MASTER_PORT=9902

# Learning rate and scheduler
LR=3e-5 # Initial learning rate
LR_SCHEDULER_TYPE="cosine"
WARMUP_RATION=0.03

# Training parameters
TRAIN_EPOCHS=2          # Number of training epochs
LOGGING_STEPS=50       # Logging frequency in steps
CKPT_SAVE_STEPS=5000    # Checkpoint saving frequency in steps

# Random seed and data precision
SEED=12
DS_DTYPE="bf16" # Options: [fp16, bf16]
RESUME="False"

# LoRA parameters
USE_LORA="True"
LORA_R=64
LORA_ALPHA=16
LORA_DROPOUT=0.05

# Data paths
MODEL_NAME="qwen2_7b_sft"
DATASET_DIR_OR_PATH="/users/wangdongnian/xuefei/v3/train_datset.jsonl"
BASE_MODEL_PATH="/users/wangdongnian/models/huggingface.co/Qwen/Qwen2-7B-Instruct"

# Output directories
OUTPUT_DIR="/users/wangdongnian/outputs/ckpt/${MODEL_NAME}_epoch${TRAIN_EPOCHS}"
mkdir -p $OUTPUT_DIR
TRAIN_LOG="${OUTPUT_DIR}/train_$(date "+%Y%m%d%H%M").log"
# TensorBoard output path
TB_DIR="/users/wangdongnian/outputs/tensorboard/${MODEL_NAME}_epoch${TRAIN_EPOCHS}"
mkdir -p $TB_DIR

# DeepSpeed configuration file
DS_CONFIG_JSON=${OUTPUT_DIR}/${MODEL_NAME}_ds_config.json
ZERO_STAGE=2

# Determine data types based on DS_DTYPE
if [ $DS_DTYPE = "fp16" ]; then
    DS_FP16=true
    DS_BF16=false
    GAS_DTYPE=$DS_DTYPE
elif [ $DS_DTYPE = "bf16" ]; then
    DS_FP16=false
    DS_BF16=true
    GAS_DTYPE="fp32"
fi

# Create DeepSpeed configuration JSON file
cat <<EOT > $DS_CONFIG_JSON
{
  "train_micro_batch_size_per_gpu": $MBS,
  "train_batch_size": "auto",
  "gradient_clipping": ${GRAD_CLIP},
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "bf16": {
    "enabled": ${DS_BF16}
  },
  "data_types": {
    "grad_accum_dtype": "${GAS_DTYPE}"
  },
  "fp16": {
    "enabled": ${DS_FP16},
    "loss_scale": 0,
    "loss_scale_window": 200,
    "hysteresis": 5,
    "min_loss_scale": 1,
    "initial_scale_power": 12
  },
  "steps_per_print": 10,
  "wall_clock_breakdown": true,
  "comms_logger": {
      "enabled": true,
      "verbose": false,
      "prof_all": false,
      "debug": false
    },
    "flops_profiler": {
        "enabled": false,
        "profile_step": 30,
        "module_depth": -1,
        "top_modules": 1,
        "detailed": true,
        "output_file": null
    }
}
EOT

# Script arguments
SCRIPT_ARGS=" \
    --mode_path ${BASE_MODEL_PATH} \
    --dataset_dir_or_path ${DATASET_DIR_OR_PATH} \
    --resume ${RESUME} \
    --seed ${SEED} \
    --output_dir ${OUTPUT_DIR} \
    --ds_config_json ${DS_CONFIG_JSON} \
    --ds_type ${DS_DTYPE} \
    --mbs ${MBS} \
    --gas ${GAS} \
    --nums_epochs ${TRAIN_EPOCHS} \
    --tb_dir ${TB_DIR} \
    --logging_steps ${LOGGING_STEPS} \
    --grad_clip ${GRAD_CLIP} \
    --lr_scheduler_type ${LR_SCHEDULER_TYPE} \
    --lr ${LR} \
    --warmup_ratio ${WARMUP_RATION} \
    --save_total_limit 3 \
    --save_steps ${CKPT_SAVE_STEPS} \
    --use_lora ${USE_LORA} \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
"

# Distributed arguments
DISTRIBUTED_ARGS=" \
    --nnodes $N_NODES \
    --nproc_per_node $N_GPUS \
"

# Check if the number of nodes is greater than or equal to 2
if [ "$N_NODES" -ge 2 ]; then
    DISTRIBUTED_ARGS+=" \
        --node_rank $RANK \
        --master_addr $MASTER_ADDR \
        --master_port $MASTER_PORT \
    "
fi

# Combine all arguments
ALL_ARGS=" $SCRIPT_ARGS "

# Launcher command
LAUNCHER="torchrun $DISTRIBUTED_ARGS ft_train.py "
# LAUNCHER="python ft_train.py "

# Export the final command
export CMD="$LAUNCHER $ALL_ARGS"
echo $CMD

# Kill any existing ft_train.py processes
killall ft_train.py

# Execute the training job
nohup $CMD 2>&1 | tee ${TRAIN_LOG} &
echo "train end : ${OUTPUT_DIR}"