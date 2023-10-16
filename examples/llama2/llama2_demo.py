# Standard libs.
import argparse

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


# python3 llama2_demo.py
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--model_path", type=str,
                        default="./Llama2-Chinese-7b-Chat", help="model path")
arg_parser.add_argument("--config_path", type=str,
                        default="./Llama2-Chinese-7b-Chat/config.json", help="model JSON config path")
arg_parser.add_argument("--ds_config", type=str,
                        default="./config/dp_zero3_config.json", help="DeepSpeed JSON config path")
arg_parser.add_argument("--local-rank", type=int, default="0")
args = arg_parser.parse_args()

config = LlamaConfig.from_pretrained(args.config_path)
model = LlamaForCausalLM(config)
# Pretrained Models are set in evaluation mode by default during initialization.
# Setting it back in training mode before doing any estimation on this model.
# Not reqruied.
model.train()
#print(model)

estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=8,
                                               num_nodes=1)

tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
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
        output_dir=out_model_path,
        overwrite_output_dir=True,
        do_train=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        # XXX: TBD.
        save_strategy="steps",
        # XXX: TBD.
        save_steps=2000,
        # XXX: TBD.
        save_total_limit=2,
        prediction_loss_only=True,
        report_to="none",
        deepspeed=args.ds_config,
)

trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        test_dataset=test_dataset,
        data_collator=data_collator,
)

#trainer.train(resume_from_checkpoint=True)
trainer.train()
