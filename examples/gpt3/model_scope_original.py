# The code below are copied from the "续写训练" section of 
# https://modelscope.cn/models/damo/nlp_gpt3_text-generation_1.3B/summary

# finetune_poetry.py
from torch.utils.tensorboard import SummaryWriter
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.metainfo import Trainers


dataset_dict = MsDataset.load('chinese-poetry-collection')
train_dataset = dataset_dict['train'].remap_columns(
    {'text1': 'src_txt'})
eval_dataset = dataset_dict['test'].remap_columns({'text1': 'src_txt'})
max_epochs = 10
tmp_dir = './gpt3_poetry'

num_warmup_steps = 100

def noam_lambda(current_step: int):
    current_step += 1
    return min(current_step**(-0.5),
               current_step * num_warmup_steps**(-1.5))

def cfg_modify_fn(cfg):
    cfg.train.lr_scheduler = {
        'type': 'LambdaLR',
        'lr_lambda': noam_lambda,
        'options': {
            'by_epoch': False
        }
    }
    cfg.train.optimizer = {'type': 'AdamW', 'lr': 3e-4}
    cfg.train.dataloader = {
        'batch_size_per_gpu': 16,
        'workers_per_gpu': 1
    }
    cfg.train.hooks.append({
        'type': 'MegatronHook'
    })
    cfg.evaluation.dataloader = {
        'batch_size_per_gpu': 8,
        'workers_per_gpu': 1
    }
    cfg.evaluation.metrics = 'ppl'
    return cfg

kwargs = dict(
    model='damo/nlp_gpt3_text-generation_1.3B',
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    max_epochs=max_epochs,
    work_dir=tmp_dir,
    cfg_modify_fn=cfg_modify_fn)

# Construct trainer and train
trainer = build_trainer(
    name=Trainers.gpt3_trainer, default_args=kwargs)
trainer.train()
