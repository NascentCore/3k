import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import deepspeed
from datasets import Dataset, DatasetDict
from torch.utils.tensorboard import SummaryWriter

# 禁用 Tokenizers 并行警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 优化内存分配
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# 配置路径和参数
local_model_path = "/data2/cairong/models/gemma-2b-it"
output_dir = "./output"
train_file_path = "/data2/cairong/datasets/chinese-poetry-collection/train.csv"
test_file_path = "/data2/cairong/datasets/chinese-poetry-collection/test.csv"

# 加载 tokenizer 和 model
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(local_model_path)

# 配置 DeepSpeed
ds_config = {
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": 1.0,
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "overlap_comm": True,
        "contiguous_gradients": True
    },
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "initial_scale_power": 32,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "activation_checkpointing": {
        "partition_activations": True,
        "contiguous_memory_optimization": True
    },
    "tensor_parallel": {
        "enabled": True,
        "tp_size": 8
    },
    "pipeline": {
        "enabled": True,
        "stage_size": 2
    }
}

# 设置训练参数
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,  # 对应 gradient_accumulation_steps = 2 时的 train_micro_batch_size_per_gpu = 2
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',  # 指定 TensorBoard 日志目录
    logging_steps=500,
    deepspeed=ds_config,
    local_rank=int(os.getenv('LOCAL_RANK', -1)),
    ddp_find_unused_parameters=False,
    fp16=True  # 启用 fp16 以匹配 DeepSpeed 配置
)

# 加载本地CSV数据集
def load_csv_dataset(train_path, test_path):
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)
        return DatasetDict({"train": train_dataset, "test": test_dataset})
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise e

dataset = load_csv_dataset(train_file_path, test_file_path)

# 数据处理
def tokenize_function(examples):
    try:
        tokenized = tokenizer(examples["text1"], padding="max_length", truncation=True, max_length=128)
        tokenized["labels"] = tokenized["input_ids"].copy()  # 为模型生成损失添加标签
        return tokenized
    except KeyError as e:
        print(f"Error in tokenizing function: {e}")
        raise e

# 标记化数据集
try:
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
except Exception as e:
    print(f"Error during tokenization: {e}")
    raise e

# 初始化 TensorBoard 记录器
writer = SummaryWriter()

# 自定义训练器以记录性能数据
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_step = 0

    def log(self, logs):
        super().log(logs)
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(key, value, self.global_step)
        self.global_step += 1

    def training_step(self, model, inputs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # 清理缓存
        return super().training_step(model, inputs)

# 初始化 Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"]
)

# 训练模型
try:
    trainer.train()
except Exception as e:
    print(f"Error during training: {e}")
    raise e

# 关闭 TensorBoard 记录器
writer.close()