# 训练
不要忘记修改code_full_training.yaml，code_sft_training.yaml的源模型路径和输出路径

## Pretrain
```bash
llamafactory-cli train code_full_training.yaml
```

## SFT
```bash
llamafactory-cli train code_sft_training.yaml
```

## Eval
```bash
llamafactory-cli eval code_eval.yaml
```
或者

```bash
lm_eval --model vllm --model_args pretrained=/home/clouder/LLaMA-Factory/saves/Qwen2.5-Coder-7B/SFT/full/train_2024-12-26-21-30-00 --tasks gsm8k --output_path output/xxxmodel

# --tasks ifeval
# --tasks leaderboard
```

# 导出
```bash
llamafactory-cli export export_lora.yaml
```

# 量化
```python
# https://qwen.readthedocs.io/en/latest/quantization/awq.html
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import pandas as pd

# Specify paths and hyperparameters for quantization
model_path = "saves/Qwen2.5-Coder-7B/PT/full/train_2024-11-21-10-48-00"
quant_path = "/data/export_model/Onecloud-Qwen2.5-Coder-AWQ-v1.5"
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load your tokenizer and model with AutoAWQ
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoAWQForCausalLM.from_pretrained(model_path, device_map="auto", safetensors=True)


data = pd.read_parquet('/home/clouder/ose_code_model_data_preprocess/train/dataset_mix.parquet')
data = data.sample(512,random_state=14)['content'].to_list()

model.quantize(tokenizer, quant_config=quant_config, calib_data=data)

model.save_quantized(quant_path, safetensors=True, shard_size="4GB")
tokenizer.save_pretrained(quant_path)
```

# 用vllm调用模型
```bash
python -m vllm.entrypoints.openai.api_server --model /data/export_model/Onecloud-Qwen2.5-Coder-AWQ-v1.4 --max-model-len 4096 --enforce-eager --port 19997 --served-model-name codeqwen1.5-2
```