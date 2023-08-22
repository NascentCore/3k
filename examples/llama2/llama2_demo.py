import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
import argparse
import torch
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import LineByLineTextDataset
import os

#parser=argparse.ArgumentParser()

#parser.add_argument("--model_path",type=str, default="./Llama2-Chinese-7b-Chat/", help="model path")
#parser.add_argument("--config_path",type=str, default="./Llama2-Chinese-7b-Chat/config.json", help="model config path")
#parser.add_argument("--ds_config",type=str, default="./config/dp_zero3_config.json", help="deepspeed config json")
#parser.add_argument("--local-rank", type=int, default="0")
#args = parser.parse_args()

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

config = LlamaConfig.from_pretrained("./Llama2-Chinese-7b-Chat/config.json")
tokenizer = LlamaTokenizer.from_pretrained("./Llama2-Chinese-7b-Chat/")
model = LlamaForCausalLM(config)
#print(model)

#zero3显存估算
#estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=8, num_nodes=1)


max_seq_length = 16
out_model_path = "llama2_output"
train_epoches = 10
batch_size = 1

tokenizer.pad_token = tokenizer.eos_token


#train_file = "wikitext-103-raw/wiki.train.raw"
#eval_file = "wikitext-103-raw/wiki.valid.raw"
#train_dataset = LineByLineTextDataset(tokenizer=tokenizer,file_path=train_file,block_size=max_seq_length)
#eval_dataset = LineByLineTextDataset(tokenizer=tokenizer,file_path=eval_file,block_size=max_seq_length)

#torch.save(train_dataset,"data/train_dataset_ml8.pt")
#torch.save(eval_dataset,"data/eval_dataset_ml8.pt")

train_dataset = torch.load('data/train_dataset_ml16.pt')
eval_dataset = torch.load('data/eval_dataset_ml16.pt')

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False) #这里bert不一样

training_args = TrainingArguments(
        output_dir=out_model_path,
        overwrite_output_dir=True,
        num_train_epochs=train_epoches,
        per_device_train_batch_size=batch_size,
        save_steps=2000,
        save_total_limit=2,
        prediction_loss_only=True,
        report_to="none",
        deepspeed = "./config/dp_zero3_config.json"
    )


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

trainer.train()
