#!/bin/bash

# 设置默认值
DEFAULT_MODEL_PATH="/home/clouder/.xinference/cache/Qwen2.5-Coder-7B-Instruct"
# 如果有命令行参数传入，则使用传入的值，否则使用默认值
MODEL_PATH=${1:-$DEFAULT_MODEL_PATH}

# 1. 运行Optuna超参数搜索
echo "Starting hyperparameter search with Optuna..."
python optuna_search.py

# 2. 使用最佳参数进行完整训练
echo "Starting full training with best parameters..."

## PT ##################################################
# FORCE_TORCHRUN=1 llamafactory-cli train code_full_training.yaml --model_name_or_path ${MODEL_PATH}

## SFT ################################################
FORCE_TORCHRUN=1 llamafactory-cli train code_sft_training.yaml --model_name_or_path ${MODEL_PATH}

## Eval ##############################################
# 获取最新的训练输出目录
LATEST_OUTPUT=$(ls -td saves/Qwen2.5-Coder-7B/SFT/*/train_* | head -n1)
mkdir -p ${LATEST_OUTPUT}/eval

# 运行评估
HF_ENDPOINT=https://hf-mirror.com lm_eval --model vllm \
--batch_size auto \
--model_args pretrained=${LATEST_OUTPUT} \
--trust_remote_code \
--tasks ifeval,cmmlu,ceval-valid > ${LATEST_OUTPUT}/eval/lm_results

## code eval ######################################
HF_ENDPOINT=https://hf-mirror.com accelerate launch /home/clouder/bigcode-evaluation-harness/main.py \
  --model "${LATEST_OUTPUT}" \
  --max_length_generation 1024 \
  --prompt codeqwen \
  --eos "<|im_end|>" \
  --tasks humanevalsynthesize-java,humanevalsynthesize-js,humanevalsynthesize-python \
  --do_sample=False \
  --n_samples 1 \
  --batch_size 1 \
  --allow_code_execution \
  --precision bf16 \
  --metric_output_path ${LATEST_OUTPUT}/eval/code_results.json

## ose eval ##############################
python /home/clouder/ose_code_model_data_preprocess/eval_test.py --model_path ${LATEST_OUTPUT} --output ${LATEST_OUTPUT}/eval/ose_results

echo "Training and evaluation completed. Results are saved in ${LATEST_OUTPUT}" 