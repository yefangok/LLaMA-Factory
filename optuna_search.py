import optuna
import subprocess
import json
import os
import glob
import time
import yaml
from typing import Dict, Any
import requests
import lm_eval
from sglang.utils import (
    execute_shell_command,
    wait_for_server,
    terminate_process,
    print_highlight,
)
from loguru import logger

# 配置日志记录
logger.add("error.log", level="ERROR", format="{time} - {level} - {message}")  # 设置日志文件和格式

def objective(trial: optuna.Trial) -> float:
    """定义Optuna优化目标函数"""
    
    # 首先选择微调类型
    finetuning_type = trial.suggest_categorical(
        "finetuning_type", [
            "lora", 
            "bone",
            "badam"
        ]
    )

    dataset = trial.suggest_categorical(
        "dataset", [
            "ose_pt_all,ose_annealing_all",
            "ose_pt,ose_annealing"
        ]
    )
    
    # 基础参数(所有微调方法通用)
    params = {
        "finetuning_type": finetuning_type,
        "dataset": dataset,
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True),
        "gradient_accumulation_steps": trial.suggest_categorical(
            "gradient_accumulation_steps", [2, 4, 8, 16]
        ),
        "warmup_steps": trial.suggest_int("warmup_steps", 0, 10),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 2.0),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 10),
        "max_steps": -1,
        "output_dir": f"saves/1.5B/trial_{trial.number}",
    }
    
    # 根据微调类型添加特定参数
    if finetuning_type == "lora":
        params.update({
            "lora_rank": trial.suggest_categorical(
                "lora_rank", [16, 32, 64, 128, 256]
            ),
            "lora_alpha": trial.suggest_int("lora_alpha", 16, 256),
            "lora_dropout": trial.suggest_float("lora_dropout", 0.0, 0.5),
        })
    elif finetuning_type == "badam":
        # bone特有的参数(如果有的话)
        params.update({
            "use_badam": True,
            "badam_mode": "layer",
            "badam_switch_mode": "ascending",
            # "badam_switch_mode": trial.suggest_categorical(
            #     "badam_switch_mode", ["ascending", "descending", "random"]
            # ),
            "badam_switch_interval": trial.suggest_int("badam_switch_interval", 5, 20),
            #"badam_update_ratio": trial.suggest_float("badam_update_ratio", 0.01, 0.1),
        })
    elif finetuning_type == "bone":
        params.update({
            "lora_rank": trial.suggest_categorical(
                "lora_rank", [16, 32, 64, 128, 256]
            ),
        })
    # 生成临时配置文件
    config = generate_config(params)
    config_path = f"/tmp/temp_config_{trial.number}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
        
    # 执行训练
    try:
        output_dir = config["output_dir"]
        # 1. 执行训练
        run_training_process(config_path)
        os.remove(config_path)  # 清理临时文件
        result = run_evaluation(output_dir)
        trial.set_user_attr("eval_loss", result["eval_loss"])
        trial.set_user_attr("humaneval_ose", result["humaneval_ose"])
        trial.set_user_attr("mmlu", result["mmlu"])
        trial.set_user_attr("ceval", result["ceval"])
        trial.set_user_attr("ifeval", result["ifeval"])
        trial.set_user_attr("humaneval_js", result["humaneval_js"])
        trial.set_user_attr("humaneval_python", result["humaneval_python"])
        # 返回主要优化指标 (humaneval-ose)
        return result["humaneval_ose"]  # 或其他指标
        
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {str(e)}")  # 记录错误信息
        return float('nan')

def objective_data(trial: optuna.Trial) -> float:
    # 首先选择微调类型
    finetuning_type = trial.suggest_categorical(
        "finetuning_type", [
            #"lora", 
            #"bone",
            "badam",
        ]
    )
    dataset = trial.suggest_categorical(
        "dataset", [
            #"ose_pt", 
            #"ose_pt_all",
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
        "max_steps": 204,         
        "num_train_epochs": -1,
        "output_dir": f"saves/1.5B/trial_{trial.number}",
    }
    
    # 根据微调类型添加特定参数
    if finetuning_type == "lora":
        params.update({
            "lora_rank": 128,
            "lora_alpha": 64,
            "lora_dropout": 0.,
        })
    elif finetuning_type == "badam":
        # bone特有的参数(如果有的话)
        params.update({
            "use_badam": True,
            "badam_mode": "layer",
            "badam_switch_mode": "ascending",
            "badam_switch_interval": 10,
        })
    elif finetuning_type == "bone":
        params.update({
            "lora_rank": 128,
        })
    # 生成临时配置文件
    config = generate_config(params)
    config_path = f"/tmp/temp_config_{trial.number}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)  # 使用yaml格式保存配置文件
        
    # 执行训练
    try:
        output_dir = config["output_dir"]
        run_training_process(config_path)
        os.remove(config_path)  # 清理临时文件
        result = run_evaluation(output_dir)
        trial.set_user_attr("eval_loss", result["eval_loss"])
        trial.set_user_attr("humaneval_ose", result["humaneval_ose"])
        trial.set_user_attr("mmlu", result["mmlu"])
        trial.set_user_attr("ceval", result["ceval"])
        trial.set_user_attr("ifeval", result["ifeval"])
        trial.set_user_attr("humaneval_js", result["humaneval_js"])
        trial.set_user_attr("humaneval_python", result["humaneval_python"])
        # 返回主要优化指标 (humaneval-ose)
        return result["humaneval_ose"]  # 或其他指标
        
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {str(e)}")  # 记录错误信息
        return float('nan')

def generate_config(params: Dict[str, Any]) -> Dict[str, Any]:
    """根据参数生成配置"""
    config = {
        "model_name_or_path": "/home/clouder/.xinference/cache/Qwen2.5-Coder-1.5B-Instruct",
        "output_dir": params["output_dir"],
        "logging_steps": 1,
        "save_steps": 20,
        
        # 训练方法
        "stage": "pt",
        "finetuning_type": params["finetuning_type"],
        
        # 数据集配置
        "dataset_dir": "/home/clouder/ose_code_model_data_preprocess/train",
        "dataset": params["dataset"],
        "template": "empty",
        "cutoff_len": 4096,
        #"eval_dataset": "ose_pt_eval",
        #"disable_shuffling": True,
        "preprocessing_num_workers": 16,
        
        # 基础训练参数
        "do_train": True,
        "learning_rate": params["learning_rate"],
        "num_train_epochs": params["num_train_epochs"],
        "max_steps": params["max_steps"],
        "per_device_train_batch_size": 16,
        "gradient_accumulation_steps": params["gradient_accumulation_steps"],
        "lr_scheduler_type": "cosine",
        "warmup_steps": params["warmup_steps"],
        "optim": "adamw_torch",
        "max_grad_norm": params["max_grad_norm"],
        "bf16": True,
        "packing": False,

        # 评估参数
        "do_eval": True,
        "val_size": 0.1,
        "eval_strategy": "steps",
        "eval_steps": 10,

        "overwrite_output_dir": True,
        "save_total_limit": 1,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
    }
    
    # 根据微调类型添加特定配置
    if params["finetuning_type"] == "lora":
        config.update({
            "lora_rank": params["lora_rank"],
            "lora_alpha": params["lora_alpha"],
            "lora_dropout": params["lora_dropout"],
            "lora_target": "all",
            "pissa_init": True,
            "pissa_iter": 16,
            "pissa_convert": True,
            "use_unsloth": True,
            "per_device_eval_batch_size": 16,
        })
    elif params["finetuning_type"] == "badam":
        config.update({
            "finetuning_type": "full",
            "use_badam": params["use_badam"],
            "badam_mode": params["badam_mode"],
            "badam_switch_mode": params["badam_switch_mode"],
            "badam_switch_interval": params["badam_switch_interval"],
            "enable_liger_kernel": True,
            "per_device_eval_batch_size": 4,
            "eval_accumulation_steps": 1,
        })
    elif params["finetuning_type"] == "bone":
        config.update({
            "lora_rank": params["lora_rank"],
            "use_unsloth": True,
            "per_device_eval_batch_size": 16,
        })
    return config

def run_evaluation(output_dir: str) -> dict:
    """返回评估结果"""

    # 1.5 如果是LoRA模型，执行合并
    export_dir = _merge_lora_if_needed(output_dir)

    _run_bigcode_eval(export_dir,output_dir)

    # 2-4. 运行各项评估
    _run_ose_eval(export_dir,output_dir)
    _run_lm_eval(export_dir,output_dir)
    eval_loss = get_eval_loss(output_dir)
    
    # 5. 收集评估结果
    results = _collect_eval_results(output_dir)
    results["eval_loss"] = eval_loss  # 调用新函数
    
    # 打印所有指标
    print("\nTrial Results:")
    print(f"  HumanEval-OSE: {results['humaneval_ose']:.4f}")
    print(f"  MMLU: {results['mmlu']:.4f}")
    print(f"  CEval: {results['ceval']:.4f}")
    print(f"  IFeval: {results['ifeval']:.4f}")
    print(f"  HumanEval-JS: {results['humaneval_js']:.4f}")
    print(f"  HumanEval-Python: {results['humaneval_python']:.4f}")
    print(f"  Eval Loss: {results['eval_loss']:.4f}")  # 打印eval_loss
    
    return results

def run_training_process(config_path: str):
    """执行训练过程"""
    cmd = [
        "llamafactory-cli",
        "train",
        config_path
    ]
    
    process = subprocess.run(cmd, capture_output=False, text=True)
    if process.returncode != 0:
        raise RuntimeError(f"Training failed: {process.stderr}")

def _merge_lora_if_needed(output_dir: str) -> str:
    """如果是LoRA模型则执行合并,返回最终的输出目录"""
    adapter_config_path = os.path.join(output_dir, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        return output_dir
        
    try:
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
            base_model_path = adapter_config.get("base_model_name_or_path")
            finetuning_type = adapter_config.get("peft_type").lower()
            
        export_config = {
            "model_name_or_path": base_model_path,
            "adapter_name_or_path": output_dir,
            "template": "qwen",
            "finetuning_type": finetuning_type,
            "trust_remote_code": True,
            "export_dir": os.path.join(output_dir, "export"),
            "export_size": 5,
            "export_device": "cpu",
            "export_legacy_format": False
        }
        
        export_config_path = os.path.join(output_dir, "export_config.yaml")
        with open(export_config_path, "w") as f:
            yaml.dump(export_config, f, indent=2)
        
        export_cmd = [
            "llamafactory-cli",
            "export",
            export_config_path
        ]
        
        export_process = subprocess.run(export_cmd, capture_output=False, text=True)
        if export_process.returncode != 0:
            raise RuntimeError(f"Model export failed: {export_process.stderr}")
            
        return os.path.join(output_dir, "export")
        
    except Exception as e:
        logger.error(f"Failed to export model: {str(e)}")  # 记录错误信息

def _run_ose_eval(model_dir: str,output_dir: str):
    """运行OSE评估"""
    eval_cmd = [
        "python",
        "/home/clouder/ose_code_model_data_preprocess/eval_test.py",
        f"--model_path={model_dir}",
        f"--output={output_dir}/ose_results"
    ]
    
    eval_process = subprocess.run(eval_cmd, capture_output=False, text=True)
    if eval_process.returncode != 0:
        raise RuntimeError(f"Evaluation failed: {eval_process.stderr}")

def _run_lm_eval(model_dir: str,output_dir: str):
    """运行LM评估 (mmlu, ceval, ifeval)"""
    lm_eval_cmd = [
        "lm_eval",
        "--model", "hf",
        "--batch_size", "8",
        "--cache_requests", "true",
        "--model_args", f"pretrained={model_dir},dtype=bfloat16",
        "--tasks", "mmlu,ceval-valid,ifeval",
        "--output_path", f"{output_dir}/lm_results"
    ]
    
    subprocess.run(lm_eval_cmd, capture_output=False, text=True, 
                  env={
                      **os.environ, 
                      "HF_ENDPOINT": "https://hf-mirror.com",
                      "HF_DATASETS_OFFLINE": "1"
                  })

def _run_bigcode_eval(model_dir: str,output_dir: str):
    """运行bigcode评估"""
    bigcode_cmd = [
        "accelerate", "launch",
        "/home/clouder/bigcode-evaluation-harness/main.py",
        "--model", model_dir,
        "--max_length_generation", "1024",
        "--prompt", "codeqwen",
        "--eos", "<|im_end|>",
        "--tasks", "humanevalsynthesize-js,humanevalsynthesize-python",
        "--do_sample=False",
        "--n_samples", "1",
        "--batch_size", "1",
        "--allow_code_execution",
        "--precision", "bf16",
        "--metric_output_path", f"{output_dir}/code_results.json"
    ]
    
    subprocess.run(bigcode_cmd, capture_output=False, text=True,
                  env={
                      **os.environ, 
                      "HF_ENDPOINT": "https://hf-mirror.com",
                      "HF_DATASETS_OFFLINE": "1"
                  })

def _collect_eval_results(output_dir: str) -> dict:
    """收集所有评估结果"""
    results = {}
    
    # 读取OSE结果
    eval_output_path = os.path.join(output_dir, "ose_results")
    try:
        with open(eval_output_path) as f:
            content = f.read().strip()
            results["humaneval_ose"] = float(content.split(": ")[1])
    except Exception as e:
        logger.error(f"Failed to read OSE results: {str(e)}")  # 记录错误信息
        print(f"Failed to read OSE results: {str(e)}")
        results["humaneval_ose"] = float('nan')
    
    # 读取LM评估结果
    try:
        lm_results_files = glob.glob(f'{output_dir}/lm_results/**/*.json', recursive=True)
        lm_results_files = lm_eval.utils.get_results_filenames(lm_results_files)
        result_file = lm_eval.utils.get_latest_filename(lm_results_files)
        with open(result_file,'r') as f:
            lm_results = json.load(f)
            results["mmlu"] = lm_results.get("results",{}).get("mmlu", {}).get("acc,none", 0)
            results["ceval"] = lm_results.get("results",{}).get("ceval-valid", {}).get("acc,none", 0)
            results["ifeval"] = lm_results.get("results",{}).get("ifeval", {}).get("inst_level_loose_acc,none", 0)
    except Exception as e:
        logger.error(f"Failed to read LM evaluation results: {str(e)}")  # 记录错误信息
        results["mmlu"] = float('nan')
        results["ceval"] = float('nan')
        results["ifeval"] = float('nan')
    # 读取bigcode评估结果
    try:
        with open(f"{output_dir}/code_results.json") as f:
            code_results = json.load(f)
            results["humaneval_js"] = code_results.get("humanevalsynthesize-js", {}).get("pass@1", 0)
            results["humaneval_python"] = code_results.get("humanevalsynthesize-python", {}).get("pass@1", 0)
    except Exception as e:
        logger.error(f"Failed to read bigcode evaluation results: {str(e)}")  # 记录错误信息
        results["humaneval_js"] = float('nan')
        results["humaneval_python"] = float('nan')
    
    # 读取eval_loss
    try:
        with open(f"{output_dir}/all_results.json") as f:
            all_results = json.load(f)
            results["eval_loss"] = all_results.get("eval_loss", 0)
    except Exception as e:
        logger.error(f"Failed to read eval_loss: {str(e)}")  # 记录错误信息
        results["eval_loss"] = float('nan')
    # 保存所有结果到trial_metrics.json
    with open(f"{output_dir}/trial_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
        
    return results

def start_sglang_api_service(output_dir: str):
    """启动SGLang API服务"""
    cmd = [
        "python",
        "-m", "sglang.launch_server",
        "--model-path", output_dir,
        #"--served-model-name", "onecloud-coder",
        "--port", "19997",
        "--host", "0.0.0.0",
        "--max-total-tokens", "4096",
    ]
    
    try:
        # 不捕获输出，让输出直接显示在控制台
        process = subprocess.Popen(
            cmd,
            # stdout=subprocess.PIPE, 
            # stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # 行缓冲
        )
        
        # # 打印VLLM服务的输出信息
        # stdout, stderr = process.communicate()  # Capture the output and error
        # if stdout:
        #     print(stdout.decode())  # Print standard output
        # if stderr:
        #     print(stderr.decode())  # Print standard error
        
        print("VLLM API service started successfully.")
        return process  # 返回进程对象
    except Exception as e:
        print(f"Error starting VLLM API service: {str(e)}")
        return None  # 返回None以便在finally中检查


# def wait_for_server(base_url: str, timeout: int = None) -> None:
#     """Wait for the server to be ready by polling the /v1/models endpoint.

#     Args:
#         base_url: The base URL of the server
#         timeout: Maximum time to wait in seconds. None means wait forever.
#     """
#     start_time = time.time()
#     while True:
#         try:
#             response = requests.get(
#                 f"{base_url}/v1/models",
#                 headers={"Authorization": "Bearer None"},
#             )
#             if response.status_code == 200:
#                 time.sleep(5)
#                 print(
#                     """\n
#                     NOTE: Typically, the server runs in a separate terminal.
#                     In this notebook, we run the server and notebook code together, so their outputs are combined.
#                     To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.
#                     """
#                 )
#                 break

#             if timeout and time.time() - start_time > timeout:
#                 raise TimeoutError("Server did not become ready within timeout period")
#         except requests.exceptions.RequestException:
#             time.sleep(1)

def get_eval_loss(output_dir: str) -> float:
    """从模型目录获取eval_loss"""
    all_results_path = os.path.join(output_dir, "all_results.json")  # 假设eval_loss保存在此文件中
    try:
        with open(all_results_path, 'r') as f:
            results = json.load(f)  # 解析JSON文件
            return results.get("eval_loss", float("nan"))  # 获取eval_loss，如果不存在则返回nan
    except Exception as e:
        logger.error(f"Failed to read eval_loss from JSON: {str(e)}")  # 记录错误信息
        return float("nan")  # 如果读取失败，返回nan

def main():
    # 指定数据库存储路径
    storage_name = "sqlite:///saves/1.5B/optuna_studies.db"
    study_name = "llama_factory_hpo"  # 为study指定一个固定名称
    
    try:
        # 尝试加载已有的study
        study = optuna.load_study(
            study_name=study_name, 
            storage=storage_name
        )
        print(f"Resuming study from {len(study.trials)} existing trials...")
        
        # 获取所有失败的trials
        failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

        # failed_trials.append({
        #     'finetuning_type': 'bone',
        #     'dataset': 'ose_pt,ose_annealing',
        #     'learning_rate': 2.e-05,
        #     'gradient_accumulation_steps': 4,
        #     'warmup_steps': 5,
        #     'max_grad_norm': 1.5,
        #     'num_train_epochs': 7,
        #     'lora_rank': 128})

        # 获取所有成功的trials的参数
        completed_params = [
            t.params for t in study.trials 
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        
        for failed_trial in failed_trials:
            # 检查失败trial的参数是否在成功trials中出现过
            if failed_trial.params not in completed_params:
                study.enqueue_trial(failed_trial.params)
                print(f"Retrying trial {failed_trial.number} with params: {failed_trial.params}")

    except KeyError as ex:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            direction="maximize",  # 改为maximize因为我们要最大化准确率
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(),
            load_if_exists=True
        )
        print("Creating new study...")
    
    # 设置要运行的总trials数
    n_trials = 50
    # 计算还需要运行多少trials
    remaining_trials = n_trials - len(study.trials)
    
    if remaining_trials > 0:
        print(f"Running {remaining_trials} more trials...")
        study.optimize(objective, n_trials=remaining_trials)
    else:
        print("All trials have been completed!")
    
    # 打印最佳结果
    print("\nBest trial:")
    print(f"  Value: {study.best_trial.value}")
    print("  Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    
    # 保存最佳参数到文件
    best_params = {
        "best_value": study.best_trial.value,
        "best_params": study.best_trial.params
    }
    with open("best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
        
    # 保存可视化结果
    try:
        optuna.visualization.plot_optimization_history(study).write_html("optimization_history.html")
        optuna.visualization.plot_param_importances(study).write_html("param_importances.html")
    except Exception as e:
        print(f"Failed to generate visualizations: {str(e)}")

if __name__ == "__main__":
    main() 