"""
On LY worker4/worker5:

#export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
export MASTER_ADDR="214.2.5.4" MASTER_PORT=29501 NCCL_DEBUG=INFO NCCL_NET=IB NCCL_P2P_LEVEL=SYS NCCL_NET_GDR_READ=1 NCCL_IB_CUDA_SUPPORT=1 NCCL_NET_GDR_LEVEL=SYS NCCL_IB_GDR_LEVEL=SYS NCCL_DEBUG_SUBSYS=ALL NCCL_SOCKET_IFNAME=ibs1 NCCL_IB_HCA=mlx5_

# LY worker4 (master host)
python3 -m torch.distributed.launch --nproc-per-node=8 --nnodes=2 --node_rank=0 --master-addr="214.2.5.4" --master-port=29501 llama2_demo.py > ~/llama2_worker4.log 2>&1 &

# LY worker5
python3 -m torch.distributed.launch --nproc-per-node=8 --nnodes=2 --node_rank=1 --master-addr="214.2.5.4" --master-port=29501 llama2_demo.py > ~/llama2_worker5.log 2>&1 &


Env vars:
- NCCL_DEBUG=INFO
- NCCL_NET=IB
- NCCL_P2P_LEVEL=SYS  # Try to enforce P2P anywhere.
- NCCL_SOCKET_IFNAME=
- NCCL_IB_HCA=
- NCCL_IB_CUDA_SUPPORT=1  # Try to enforce GDR.
- NCCL_NET_GDR_LEVEL=SYS
- NCCL_IB_GDR_LEVEL=SYS
- NCCL_NET_GDR_READ=1
- NCCL_DEBUG_SUBSYS=ALL

Note:
- "NCCL_P2P_DISABLE=1" should not be set.
- "NCCL_P2P_DIRECT_DISABLE=1" should not be set.
- ibdev2netdev to check NCCL_SOCKET_IFNAME (rhs) and NCCL_IB_HCA (lhs)
- "NCCL_COLLNET_ENABLE=1" may need to be set.
- "NCCL_COLLNET_NODE_THRESHOLD=17" may need to be set.
- NCCL version


torch.distributed.launch OR torchrun
"""


# Standard libs.
import argparse
import sys
import os
from datetime import datetime

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
# For testing multi-node-multi-gpu distibuted training,
# we need to launch this script using MPI related commands (e.g., mpirun),
# torch.distributed.launch, or torchrun with appropriate arguments and options.
num_gpus = torch.cuda.device_count()

# Linux is supposed to be always okay.
if not torch.distributed.is_available():
    sys.exit("`torch.distributed` package is not available. Exiting...")

# In MPI, group -- handle -- comminucator -- processes.
# The default (working) group is the world.
torch.distributed.init_process_group(backend="nccl", init_method="env://")

print(f"WORLD_SIZE: {torch.distributed.get_world_size()}")
print(f"Current global rank: {torch.distributed.get_rank()}")

# python3 llama2_demo.py
arg_parser = argparse.ArgumentParser()
# Fine-tuning or transfer learning.
arg_parser.add_argument("--model-path", type=str,
                        default="./Llama2-Chinese-7b-Chat",
                        help="Pretrained model path")
arg_parser.add_argument("--model-config-file", type=str,
                        default="./Llama2-Chinese-7b-Chat/config.json",
                        help="Model JSON config")
arg_parser.add_argument("--ds-config-file", type=str,
                        default="./config/dp_zero3_config.json",
                        help="DeepSpeed JSON config path")
arg_parser.add_argument("--raw-data-dir", type=str,
                        default="./wikitext-103-raw",
                        help="Path for raw data to be tokenized")
arg_parser.add_argument("--tokenized-data-dir", type=str, default="./data/",
                        help="Path for raw data to be tokenized")
arg_parser.add_argument("--saved-model-dir", type=str,
                        default="./llama2_output",
                        help="Path for saving learned models")
# --local-rank=LOCAL_PROCESS_RANK, which will be provided by `torch.distributed` module.
arg_parser.add_argument("--local-rank", type=int)
args = arg_parser.parse_args()

curr_local_rank = args.local_rank
print(f"Current local rank: {curr_local_rank}")

#torch.cuda.set_device(curr_local_rank)

gpu_name = f"cuda:{curr_local_rank}"

config = LlamaConfig.from_pretrained(args.model_config_file)
model = LlamaForCausalLM(config)

gpu_dev = torch.device(gpu_name)
model.to(gpu_dev)

# Pretrained Models are set in evaluation mode by default during initialization.
# Setting it back in training mode before doing any estimation on this model.
# Not reqruied.
model.train()
#print(model)

# A single node with multiple GPUs.
#estimate_zero3_model_states_mem_needs_all_live(model,
#                                               num_gpus_per_node=num_gpus,
#                                               num_nodes=1)

tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
# XXX
tokenizer.pad_token = tokenizer.eos_token

max_seq_length = 16
# FIXME
#out_model_path = f"llama2_output_{datetime.now().strftime('%Y%m%d%H%M%S')}"
out_model_path = args.saved_model_dir
num_train_epochs = 4
batch_size = 8

# Train dataset
if os.path.exists(f"./{args.tokenized_data_dir}/train_dataset_ml8.pt"):
    train_dataset = torch.load(f"{args.tokenized_data_dir}/train_dataset_ml8.pt")
else:
    train_dataset = LineByLineTextDataset(tokenizer=tokenizer,
                                          file_path=f"{args.raw_data_dir}/wiki.train.raw",
                                          block_size=max_seq_length)
    torch.save(train_dataset, f"{args.tokenized_data_dir}/train_dataset_ml8.pt")
# Evaluation dataset
if os.path.exists(f"./{args.tokenized_data_dir}/eval_dataset_ml8.pt"):
    eval_dataset = torch.load(f"{args.tokenized_data_dir}/eval_dataset_ml8.pt")
else:
    eval_dataset = LineByLineTextDataset(tokenizer=tokenizer,
                                         file_path=f"{args.raw_data_dir}/wiki.eval.raw",
                                         block_size=max_seq_length)
    torch.save(eval_dataset, f"{args.tokenized_data_dir}/eval_dataset_ml8.pt")

"""
# Test dataset
if os.path.exists(f"./{args.tokenized_data_dir}/test_dataset_ml8.pt"):
    test_dataset = torch.load(f"{args.tokenized_data_dir}/test_dataset_ml8.pt")
else:
    test_dataset = LineByLineTextDataset(tokenizer=tokenizer,
                                         file_path=f"{args.raw_data_dir}/wiki.test.raw",
                                         block_size=max_seq_length)
    torch.save(test_dataset, f"{args.tokenized_data_dir}/test_dataset_ml8.pt")
"""

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    #log_level="debug",
    log_level="info",
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
    push_to_hub=False,
    local_rank=curr_local_rank,  # XXX
    deepspeed=args.ds_config_file,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

print("Begin training...")

#trainer.train(resume_from_checkpoint=True)
trainer.train()

print("Saving model...")
# FIXME
trainer.save_model()

print("Done training...")

# XXX: Needs to check if `transformers.Trainer.train()`
# is sync or async by default.
torch.distributed.destroy_process_group()

#os.exit(0)
