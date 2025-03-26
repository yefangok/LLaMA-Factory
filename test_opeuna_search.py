from optuna_search import *
from unittest.mock import Mock
import yaml
if __name__ == "__main__":
    trial = Mock(spec=optuna.trial.Trial)
        # 首先选择微调类型
    finetuning_type = trial.suggest_categorical(
        "finetuning_type", [
            "lora", 
            "bone",
        ]
    )
    dataset = trial.suggest_categorical(
        "dataset", [
            "ose_pt", 
            "ose_pt_all",
            "ose_pt_all,ose_annealing_all",
            "ose_pt,ose_annealing"
        ]
    )
    # 基础参数(所有微调方法通用)
    params = {
        "finetuning_type": finetuning_type,
        "dataset": dataset,
        "learning_rate": 5e-5,
        "gradient_accumulation_steps": 8,
        "warmup_steps": 0,
        "max_grad_norm": 1.,
        "num_train_epochs": 3,
        "overwrite_output_dir": True,
        "save_total_limit": 1,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
    }
    
    # 根据微调类型添加特定参数
    if finetuning_type == "lora":
        params.update({
            "lora_rank": 128,
            "lora_alpha": 64,
            "lora_dropout": 0.,
        })
    elif finetuning_type == "bone":
        params.update({
            "lora_rank": 128,
        })
    
    trial.number = 2

    # 生成临时配置文件
    config = generate_config(params)
    config["finetuning_type"] = "bone"
    config["lora_rank"] = 128
    config["dataset"] = "ose_pt,ose_annealing"
    config["output_dir"] = f"/tmp/trial_{trial.number}"
    config_path = f"/tmp/temp_config_{trial.number}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    output_dir = config["output_dir"]
    run_training_process(config_path)
    print(f"Training completed. Output directory: {output_dir}")