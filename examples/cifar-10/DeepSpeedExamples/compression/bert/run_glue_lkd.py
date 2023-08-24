# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import logging
import math
import time
import os
import random
import json
import numpy as np
from pathlib import Path

import datasets
from datasets import load_dataset, load_metric
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

import transformers
from huggingface_hub import Repository
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from transformers.file_utils import get_full_repo_name
from transformers.utils.versions import require_version
from huggingface_transformer.modeling_bert import BertForSequenceClassification
import deepspeed
from deepspeed.compression.compress import init_compression, redundancy_clean
from deepspeed.compression.helper import recursive_getattr
from util import *
logger = logging.getLogger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=64,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_epochs", type=float, default=0, help="Number of epochs for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
 

    #############deepspeed, compression, and knowledage distillation#########
    parser.add_argument("--deepspeed", action="store_true", help="use deepspeed or not")
    parser.add_argument("--deepspeed_config", type=str, default=None, help="deepspeed config")   
    parser.add_argument("--save_best_model", action="store_true",  help="save best checkpoint model")
    parser.add_argument("--clean_best_model", action="store_true", help="clean the  model")
    parser.add_argument("--lkd_enabled", action="store_true", help="using lkd or not")
    parser.add_argument("--distill_method", type=str, default=None, help="knowledage distillation")   
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus")
    parser.add_argument(
        "--model_name_or_path_teacher",
        default=None,
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
    )
    
    parser.add_argument(
        "--pretrained_dir_student",
        type=str,
        default=None,
        help="Where to load the student pretrained model.")
    parser.add_argument(
        "--pretrained_dir_teacher",
        type=str,
        default=None,
        help="Where to load the teacher pretrained model.")
    parser.add_argument(
        "--eval_step",
        type=int,
        default=1000,
        help="when to eval the model.")

    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    return args


def main():
    args = parse_args()
    print_rank_0 = print_rank(args)
    ds_config = None
    if args.deepspeed:
        with open(args.deepspeed_config) as f:
            ds_config = json.load(f)
        layer_reduction_enabled, prune_enabled, quantization_enabled = check_and_identify_compresssion(args, ds_config)
        args.layer_reduction_enabled = layer_reduction_enabled
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if args.lkd_enabled:
        assert args.distill_method != "zero_stage", "zero_stage is not supported for lkd since we need the teacher model"

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.ERROR)
    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        #torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()
    if args.seed is not None:
        set_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.distributed.barrier()
    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("glue", args.task_name)
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (args.train_file if args.train_file is not None else args.valid_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            label_list=None
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            label_list=None
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)
    output_mode = output_modes[args.task_name]
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    if layer_reduction_enabled:
        config.num_hidden_layers = ds_config["compression_training"][ "layer_reduction"]["keep_number_layer"]  #<==========================================here we assume there is an "num_hidden_layers" argument

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    model = BertForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )
    model.to(device)
    teacher_model  = None
    #### load teacher models
    if args.distill_method != 'zero_stage':
        if not args.model_name_or_path_teacher:
            args.model_name_or_path_teacher = args.model_name_or_path
        teacher_config = AutoConfig.from_pretrained(
            args.model_name_or_path_teacher,
            num_labels=num_labels,
            finetuning_task=args.task_name)
        teacher_model = BertForSequenceClassification.from_pretrained(
            args.model_name_or_path_teacher,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=teacher_config,
        )
        teacher_model.to(device)
        if args.pretrained_dir_teacher is not None:
            teacher_model.load_state_dict(
                torch.load(args.pretrained_dir_teacher))
    # model inititalization, config,
    if args.deepspeed:
        if quantization_enabled or prune_enabled or layer_reduction_enabled:
            model = init_compression(model, args.deepspeed_config, teacher_model=teacher_model)  #<==========================================compression argument

    if args.pretrained_dir_student is not None:
            model.load_state_dict(torch.load(args.pretrained_dir_student))  #<==========================================add weight to students if users provides difference models

    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            print_rank_0(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}
   
    label_to_id = None
    replace_config(args, config, model, label_list, num_labels=num_labels, label_to_id=None, is_regression=is_regression)

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        print_rank_0(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=default_data_collator,
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=default_data_collator,
                                 sampler=eval_sampler,
                                 batch_size=args.per_device_eval_batch_size)
    mm_eval_dataloader = None
    if args.task_name == "mnli":
        # Final evaluation on mismatched validation set
        mm_eval_dataset = processed_datasets["validation_mismatched"]
        mm_eval_sampler = SequentialSampler(mm_eval_dataset)
        mm_eval_dataloader = DataLoader(
            mm_eval_dataset,
            collate_fn=default_data_collator,
            sampler=mm_eval_sampler,
            batch_size=args.per_device_eval_batch_size)
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps =  math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    num_warmup_steps = int(args.num_warmup_epochs * num_update_steps_per_epoch)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    # Prepare the model first eo enable compression feature
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)


    out = do_eval(args, model, eval_dataloader, mm_eval_dataloader, device, is_regression=is_regression)
    current_result, _, _, _ = arrange_output(args.task_name, out, None, 0.0)
    print_rank_0(f"at step 0 (without LKD) the (student) model's performance for {args.task_name}: {current_result}")
    
    model.eval()
    teacher_model.eval()
    start_time = time.time()
    
    for l in range(model.module.config.num_hidden_layers): # iterate across BERT layers
        print_rank_0(f"layer {l}")
        student_layer = recursive_getattr(model.module, f'bert.encoder.layer.{l}')  # extract the lth layer of student

        optimizer_param = [
        {
            "params": [p for n, p in student_layer.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in student_layer.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        ]  

        optimizer = AdamW(optimizer_param, lr=args.learning_rate) 

        updated_steps = 0

        for _ in range(args.num_train_epochs):
            for _, batch in enumerate(train_dataloader):  # load each batch
                batch = to_device(batch, device)
                with torch.no_grad():
                    # for simplicity, we always run the full inference of the teacher model.
                    # To get the best performance, you can run the teacher model only for the first l layers,
                    # which requires some modifications to the modeling code.
                    teacher_out = teacher_model(**batch, output_hidden_states=True) # get the output of the teacher model
                layer_input = teacher_out.hidden_states[l] # extract the lth-layer's input of teacher
                teacher_o = teacher_out.hidden_states[l+1] # extract the lth-layer's output of teacher

                real_mask = teacher_model.bert.get_extended_attention_mask(batch['attention_mask'], \
                    batch['input_ids'].shape, batch['input_ids'].device) # get the mask
                student_o = student_layer(layer_input, real_mask)[0] # run inference for the student

                loss = torch.nn.functional.mse_loss(student_o, teacher_o) 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                updated_steps += 1 
                if updated_steps >= args.max_train_steps :  # break when the number of steps is reached, typically in hundreds
                    break
            if updated_steps >= args.max_train_steps:
                break
    
    out = do_eval(args, model, eval_dataloader, mm_eval_dataloader, device, is_regression=is_regression)
    current_result, _, _, _ = arrange_output(args.task_name, out, None, 0.0)
    print_rank_0(f"After {time.time() - start_time}s, (with LKD) the (student) model's performance for {args.task_name}: {current_result}")
    

if __name__ == "__main__":
    main()
