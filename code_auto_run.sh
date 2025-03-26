#!/bin/bash

SFT_PATH="saves/Qwen2.5-Coder-7B/SFT/full" 

lm_eval --model vllm \
--model_args pretrained=/home/clouder/.xinference/cache/Qwen2.5-Coder-7B-Instruct \
--output_path saves/Qwen2.5-Coder-7B/SFT/origin/eval/ifeval/results.json \
--tasks ifeval

lm_eval --model vllm \
--model_args pretrained=${SFT_PATH}/train_2024-12-24-00-00-00 \
--output_path ${SFT_PATH}/train_2024-12-24-00-00-00/eval/ifeval/results.json \
--tasks ifeval


lm_eval --model vllm \
--model_args pretrained=${SFT_PATH}/train_2024-12-25-10-57-00 \
--output_path ${SFT_PATH}/train_2024-12-25-10-57-00/eval/ifeval/results.json \
--tasks ifeval

lm_eval --model vllm \
--model_args pretrained=${SFT_PATH}/train_2024-12-26-21-30-00 \
--output_path ${SFT_PATH}/train_2024-12-26-21-30-00/eval/ifeval/results.json \
--tasks ifeval

## code 测试 ######################################

mkdir ${SFT_PATH}/train_2024-12-26-21-30-00/eval/humanevalpack

HF_ENDPOINT=https://hf-mirror.com accelerate launch /home/clouder/bigcode-evaluation-harness/main.py \
  --model "${SFT_PATH}/train_2024-12-26-21-30-00" \
  --max_length_generation 1024 \
  --prompt codeqwen \
  --eos "<|im_end|>" \
  --tasks humanevalsynthesize-java,humanevalsynthesize-js,humanevalsynthesize-python \
  --do_sample=False \
  --n_samples 1 \
  --batch_size 1 \
  --allow_code_execution \
  --precision bf16 \
  --metric_output_path ${SFT_PATH}/train_2024-12-26-21-30-00/eval/humanevalpack/results.json
  #--temperature 0

  ################################