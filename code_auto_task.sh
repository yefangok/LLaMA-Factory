#!/bin/bash

# 设置默认值
DEFAULT_SFT_PATH="saves/Qwen2.5-Coder-7B/PT/lora/train_2025-01-06-16-45-00/export"
# 如果有命令行参数传入，则使用传入的值，否则使用默认值
# 使用自定义路径
# ./code_auto_task.sh "/path/to/your/model"
SFT_PATH=${1:-$DEFAULT_SFT_PATH}

## PT ##################################################
# 强制使用torchrun进行分布式训练
# FORCE_TORCHRUN=1: 强制使用torchrun而不是普通的训练模式，目的是能够通过命令行传递参数(llamafactory的要求)
# code_full_training.yaml: 训练配置文件
# --model_name_or_path: 指定预训练模型路径
# FORCE_TORCHRUN=1 llamafactory-cli train code_full_training.yaml --model_name_or_path ${SFT_PATH}

## SFT ################################################
# 使用torchrun进行SFT(Supervised Fine-Tuning)分布式训练
# FORCE_TORCHRUN=1: 强制使用torchrun而不是普通的训练模式，目的是能够通过命令行传递参数(llamafactory的要求)
# code_sft_training.yaml: SFT训练配置文件
# --model_name_or_path: 指定预训练模型路径
# FORCE_TORCHRUN=1 llamafactory-cli train code_sft_training.yaml --model_name_or_path ${SFT_PATH}

## Eval ##############################################
mkdir ${SFT_PATH}/eval

#llamafactory-cli eval code_eval.yaml --task ceval_validation --model_name_or_path ${SFT_PATH} --save_dir ${SFT_PATH}/eval/ceval_validation
HF_ENDPOINT=https://hf-mirror.com lm_eval --model vllm \
--batch_size auto \
--model_args pretrained=${SFT_PATH} \
--trust_remote_code \
--tasks ifeval,cmmlu,ceval-valid > ${SFT_PATH}/eval/lm_results

## code eval ######################################
# evalplus.evaluate --model "/data/export_model/Onecloud-Qwen2.5-Coder-AWQ-v1.4" --dataset mbpp  --backend vllm --greedy

HF_ENDPOINT=https://hf-mirror.com accelerate launch /home/clouder/bigcode-evaluation-harness/main.py \
  --model "${SFT_PATH}" \
  ${PEFT_PATH:+--peft_model ${PEFT_PATH}} \
  --max_length_generation 1024 \
  --prompt codeqwen \
  --eos "<|im_end|>" \
  --tasks humanevalsynthesize-java,humanevalsynthesize-js,humanevalsynthesize-python \
  --do_sample=False \
  --n_samples 1 \
  --batch_size 1 \
  --allow_code_execution \
  --precision bf16 \
  --metric_output_path ${SFT_PATH}/eval/code_results.json

## ose eval ##############################
python /home/clouder/ose_code_model_data_preprocess/eval_test.py --model_path ${SFT_PATH} --output ${SFT_PATH}/eval/ose_results