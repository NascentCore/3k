"""
On LY worker4:

export MASTER_ADDR="214.2.5.4" MASTER_PORT=29501 NCCL_DEBUG=INFO NCCL_NET=IB NCCL_IB_CUDA_SUPPORT=1 NCCL_NET_GDR_LEVEL=SYS NCCL_IB_GDR_LEVEL=SYS NCCL_DEBUG_SUBSYS=ALL NCCL_SOCKET_IFNAME=mlx5_ NCCL_IB_HCA=ibs1

python3 -m torch.distributed.launch --nproc-per-node=8 --nnodes=1 --master-addr="214.2.5.4" --master-port=29501 llama2_demo.py


Env vars:
- NCCL_DEBUG=INFO
- NCCL_NET=IB
- NCCL_SOCKET_IFNAME=
- NCCL_IB_HCA=
- NCCL_IB_CUDA_SUPPORT=1
- NCCL_NET_GDR_LEVEL=SYS
- NCCL_IB_GDR_LEVEL=SYS
- NCCL_DEBUG_SUBSYS=ALL

Note:
- "NCCL_P2P_DISABLE=1" should not be set.
- ibdev2netdev to check NCCL_SOCKET_IFNAME and NCCL_IB_HCA
- "NCCL_COLLNET_ENABLE=1" may need to be set.
- "NCCL_COLLNET_NODE_THRESHOLD=17" may need to be set.
- NCCL version

torch.distributed.launch OR torchrun
"""


# Standard libs.
import argparse
import sys
import os

# 3rd-party libs.
import torch
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
from transformers import (
    LlamaConfig,
    LlamaTokenizer,
    LlamaForCausalLM,
    DataCollatorForLanguageModeling,
    LineByLineTextDataset,
    Trainer,
    TrainingArguments,
)


if not torch.cuda.is_available():
    sys.exit("CUDA/GPU not available on this node. Exiting...")
# This is the number of total available GPUs on this node.
# Note: We have an env var - WORLD_SIZE, which is manully specified.
# For testing distibuted training, we may need to launch
# this script using MPI related commands (e.g., mpirun),
# torch.distributed.launch, or torchrun with appropriate arguments and options.
num_gpus = torch.cuda.device_count()

# Linux is supposed to be always okay.
if not torch.distributed.is_available():
    sys.exit("`torch.distributed` package is not available. Exiting...")

# In MPI, group -- handle -- comminucator -- processes.
# The default (working) group is the world.
torch.distributed.init_process_group(backend="nccl", init_method="env://")

curr_rank = torch.distributed.get_rank()
print(f"world_size: {torch.distributed.get_world_size()}")
print(f"Current global rank: f{curr_rank}")

gpu_name = f"cuda:{curr_rank}"

# python3 llama2_demo.py
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--model_path", type=str,
                        default="./Llama2-Chinese-7b-Chat", help="model path")
arg_parser.add_argument("--config_path", type=str,
                        default="./Llama2-Chinese-7b-Chat/config.json", help="model JSON config path")
arg_parser.add_argument("--ds_config", type=str,
                        default="./config/dp_zero3_config.json", help="DeepSpeed JSON config path")
arg_parser.add_argument("--local-rank", type=int, default=f"{curr_rank}")
args = arg_parser.parse_args()

config = LlamaConfig.from_pretrained(args.config_path)
model = LlamaForCausalLM(config)

gpu_dev = torch.device(gpu_name)
model.to(gpu_dev)

# Pretrained Models are set in evaluation mode by default during initialization.
# Setting it back in training mode before doing any estimation on this model.
# Not reqruied.
model.train()
#print(model)

# A single node with multiple GPUs.
estimate_zero3_model_states_mem_needs_all_live(model,
                                               num_gpus_per_node=num_gpus,
                                               num_nodes=1)

tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
# XXX
tokenizer.pad_token = tokenizer.eos_token

max_seq_length = 16
out_model_path = "llama2_output"
num_train_epochs = 10
batch_size = 1

train_dataset = LineByLineTextDataset(tokenizer=tokenizer,
                                      file_path="wikitext-103-raw/wiki.train.raw",
                                      block_size=max_seq_length)
eval_dataset = LineByLineTextDataset(tokenizer=tokenizer,
                                     file_path="wikitext-103-raw/wiki.valid.raw",
                                     block_size=max_seq_length)
test_dataset = LineByLineTextDataset(tokenizer=tokenizer,
                                     file_path="wikitext-103-raw/wiki.test.raw",
                                     block_size=max_seq_length)
torch.save(train_dataset, "data/train_dataset_ml8.pt")
torch.save(eval_dataset, "data/eval_dataset_ml8.pt")
torch.save(test_dataset, "data/test_dataset_ml8.pt")

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
        log_level=debug,
        output_dir=out_model_path,
        overwrite_output_dir=True,
        do_train=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        save_strategy="steps",  # TBD
        save_steps=2000,  # TBD
        save_total_limit=2,  # TBD
        prediction_loss_only=True,
        report_to="none",
        local_rank=curr_rank,  # XXX
        deepspeed=args.ds_config,
)

trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset.cuda(),
        eval_dataset=eval_dataset.cuda(),
        test_dataset=test_dataset.cuda(),
        data_collator=data_collator,
)

#trainer.train(resume_from_checkpoint=True)
trainer.train()

# XXX: Needs to check if `transformers.Trainer.train()`
# is sync or async by default.
#torch.distributed.destroy_process_group()
