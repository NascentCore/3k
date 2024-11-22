# API配置
API_HOST = "https://llm.sxwl.ai"
API_TOKEN = "Bearer ..."
SX_USER_ID = "1234567890"

# 微调配置
FINETUNE_CONFIG = {
    "gpu_model": "NVIDIA-GeForce-RTX-3090",
    "gpu_count": 1,
    "finetune_type": "lora",
    "hyperparameters": {
        "n_epochs": "3.0",
        "batch_size": "4",
        "learning_rate_multiplier": "5e-5"
    },
    "model_saved_type": "lora",
    "model_name": "google/gemma-2b-it",
    "dataset_name": "user-abfd8ec1-e78b-4437-b300-25beb5561bc2/chinese_medi_train",
    "min_instances": 1,
    "max_instances": 1,
} 

# 评测配置
EVALUATION_CONFIG = {
    "evaluation_file": "./evaluation.json",
    "output_file": "./results.json",
    "model": "/mnt/models",
    "temperature": 1,
    "top_k": 10,
    "stream": False
}