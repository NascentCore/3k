import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['NCCL_DEBUG'] = 'WARN'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def train(rank, world_size, base_model, new_model, guanaco_dataset, training_args):
    setup(rank, world_size)

    dataset = load_dataset(guanaco_dataset, split="train")
    train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)

    compute_dtype = getattr(torch, "float16")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    torch.cuda.set_device(rank)
    model = DDP(model, device_ids=[rank], output_device=rank)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=training_args.per_device_train_batch_size,
                                                   sampler=train_sampler)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_params,
        dataset_text_field="text",
        max_seq_length=128,
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )

    trainer.train()
    if rank == 0:
        trainer.model.module.save_pretrained(new_model)


def main():
    base_model = "/data/models/hf_models/llama-2-7b-chat"
    guanaco_dataset = "/data/dataset/guanaco-llama2-1k"
    new_model = "llama-2-7b-chat-guanaco"

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=25,
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard"
    )

    world_size = torch.cuda.device_count()
    print(f"world_size: {world_size}")
    mp.spawn(train, args=(world_size, base_model, new_model, guanaco_dataset, training_args), nprocs=world_size,
             join=True)


if __name__ == "__main__":
    main()
