# Standard libs.
import argparse
import sys
import os
from datetime import datetime
import gc

# 3rd-party libs.
from loguru import logger
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
    TrainerCallback,
)
from accelerate.utils import DistributedType


class DevicePlacementCheckCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        curr_device = torch.cuda.current_device()
        local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        if curr_device != local_rank:
            logger.info(f"Process {os.getpid()} is running on a wrong GPU. "
                        f"Expected: {local_rank}, current: {curr_device}.")
            # FIXME: This way is literally wrong. What we really need to do
            # is to get rid of the process from the wrong GPU placement,
            # rather than terminating it.
            sys.exit(1)

    def on_train_end(self, args, state, control, **kwargs):
        curr_device = torch.cuda.current_device()
        local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        if curr_device != local_rank:
            logger.info(f"Process {os.getpid()} is running on a wrong GPU. "
                        f"Expected: {local_rank}, current: {curr_device}.")
            # FIXME: This way is literally wrong. What we really need to do
            # is to get rid of the process from the wrong GPU placement,
            # rather than terminating it.
            sys.exit(1)

    def on_step_begin(self, args, state, control, **kwargs):
        curr_device = torch.cuda.current_device()
        local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        if curr_device != local_rank:
            logger.info(f"Process {os.getpid()} is running on a wrong GPU. "
                        f"Expected: {local_rank}, current: {curr_device}.")
            # FIXME: This way is literally wrong. What we really need to do
            # is to get rid of the process from the wrong GPU placement,
            # rather than terminating it.
            sys.exit(1)

    def on_step_end(self, args, state, control, **kwargs):
        curr_device = torch.cuda.current_device()
        local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        if curr_device != local_rank:
            logger.info(f"Process {os.getpid()} is running on a wrong GPU. "
                        f"Expected: {local_rank}, current: {curr_device}.")
            sys.exit(1);


if not torch.cuda.is_available():
    sys.exit("CUDA/GPU not available on this node. Exiting...")

# Env vars supposed to be automatically created by `mpirun`.
mpi_local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
mpi_world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
mpi_world_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])

if "CUDA_DEVICE_ORDER" not in os.environ:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    # XXX: Not scalable enough.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(mpi_local_rank)
logger.info(f"CUDA_VISIBLE_DEVICES on process {os.getpid()}: {os.environ['CUDA_VISIBLE_DEVICES']}")

#curr_local_rank = args.local_rank
curr_local_rank = int(os.environ["CUDA_VISIBLE_DEVICES"])
# XXX: The first statement is not encouraged.
torch.cuda.set_device(curr_local_rank)
torch.cuda.device(curr_local_rank)
logger.info(f"Current local rank for process {os.getpid()}: {curr_local_rank}")

gc.collect()
torch.cuda.empty_cache()

# Linux is supposed to be always okay.
if not torch.distributed.is_available():
    sys.exit("`torch.distributed` package is not available. Exiting...")

# In MPI, group -- handle -- comminucator -- processes.
# The default (working) group is the world.
#torch.distributed.init_process_group(backend="nccl", init_method="env://")

# How to work compatibaly with `mpirun`.
torch.distributed.init_process_group(backend="nccl", init_method="env://",
                                     world_size=mpi_world_size, rank=mpi_world_rank)

logger.info(f"WORLD_SIZE: {torch.distributed.get_world_size()}")
logger.info(f"Current global rank: {torch.distributed.get_rank()}")

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
arg_parser.add_argument("--ckpt-dir", type=str,
                        default="./llama2_ckpt",
                        help="Path for saving checkpoints during training")
# --local-rank=LOCAL_PROCESS_RANK, which will be provided by `torch.distributed` module.
arg_parser.add_argument("--local-rank", type=int)
args = arg_parser.parse_args()

logger.info("Creating llama config from pretrained model ...")
config = LlamaConfig.from_pretrained(args.model_config_file)

logger.info("Creating llama model with llama config ...")
model = LlamaForCausalLM(config)

# XXX: When to check in a smart way?
#curr_device = torch.cuda.current_device()
#if curr_device != curr_local_rank:
#    sys.exit(f"Process {os.getpid()} is running on a wrong GPU. "
#             f"Expected: {curr_local_rank}, current: {curr_device}.")

gpu_name = f"cuda:{curr_local_rank}"
gpu_dev = torch.device(gpu_name)

logger.info(f"Copying llama model to {gpu_name} ...")
model.to(gpu_dev)

# Pretrained Models are set in evaluation mode by default during initialization.
# Setting it back in training mode before doing any estimation on this model.
# Not reqruied.
model.train()
#print(model)

# This is the number of total available GPUs on this node.
# For testing multi-node-multi-gpu distibuted training,
# we need to launch this script using MPI related commands (e.g., mpirun),
# torch.distributed.launch, or torchrun with appropriate arguments and options.
#num_gpus = torch.cuda.device_count()

# A single node with multiple GPUs.
# estimate_zero3_model_states_mem_needs_all_live(model,
#                                                num_gpus_per_node=num_gpus,
#                                                num_nodes=1)

# XXX: When to check in a smart way?
#curr_device = torch.cuda.current_device()
#if curr_device != curr_local_rank:
#    sys.exit(f"Process {os.getpid()} is running on a wrong GPU. "
#             f"Expected: {curr_local_rank}, current: {curr_device}.")

logger.info("Creating llama tokenizer...")
tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
# XXX
tokenizer.pad_token = tokenizer.eos_token

max_seq_length = 16
# FIXME
#ckpt_output_path = f"llama2_output_{datetime.now().strftime('%Y%m%d%H%M%S')}"
ckpt_output_path = args.ckpt_dir
num_train_epochs = 4
batch_size = 8

# Train dataset
logger.info("Loading or tokenizing training dataset...")
if os.path.exists(f"./{args.tokenized_data_dir}/train_dataset_ml8.pt"):
    train_dataset = torch.load(f"{args.tokenized_data_dir}/train_dataset_ml8.pt")
else:
    train_dataset = LineByLineTextDataset(tokenizer=tokenizer,
                                          file_path=f"{args.raw_data_dir}/wiki.train.raw",
                                          block_size=max_seq_length)
    torch.save(train_dataset, f"{args.tokenized_data_dir}/train_dataset_ml8.pt")

# Evaluation dataset
logger.info("Loading or tokenizing evaluation dataset...")
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

logger.info("Creating data collector ...")
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

logger.info("Creating training arguments ...")
training_args = TrainingArguments(
    # log_level="debug",
    log_level="info",
    log_on_each_node=False,
    output_dir=ckpt_output_path,
    overwrite_output_dir=True,
    do_train=True,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    save_strategy="epoch",  # TBD
    # save_steps=2000,  # TBD
    save_total_limit=4,  # TBD
    # save_on_each_node=False,
    prediction_loss_only=True,
    report_to="none",
    push_to_hub=False,
    local_rank=curr_local_rank,
    # ddp_backend="nccl",
    deepspeed=args.ds_config_file,
)
# https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py
# For some reason, when enabling DeepSpeed, we need to explicitly do this.
training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

# XXX: When to check in a smart way?
curr_device = torch.cuda.current_device()
if curr_device != curr_local_rank:
    sys.exit(f"Process {os.getpid()} is running on a wrong GPU. "
             f"Expected: {curr_local_rank}, current: {curr_device}.")

logger.info("Creating trainer ...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    callbacks=[DevicePlacementCheckCallback],
)

# XXX: When to check in a smart way?
curr_device = torch.cuda.current_device()
if curr_device != curr_local_rank:
    sys.exit(f"Process {os.getpid()} is running on a wrong GPU. "
             f"Expected: {curr_local_rank}, current: {curr_device}.")

logger.info("Start training ...")
#trainer.train(resume_from_checkpoint=True)
trainer.train()

logger.info("Saving model ...")
# FIXME
trainer.save_model(args.saved_model_dir)

logger.info("Destroying torch distributed training group ...")

# TODO(glen): Needs to check if `transformers.Trainer.train()`
# is sync or async by default.
torch.distributed.destroy_process_group()
