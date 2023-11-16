from torch.utils.tensorboard import SummaryWriter
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.metainfo import Trainers


# Remote datasets in Hub
#dataset_dict = MsDataset.load('DuReader_robust-QG')
#train_dataset = dataset_dict['train'].remap_columns({'text1': 'src_txt', 'text2': 'tgt_txt'}) \
#    .map(lambda example: {'src_txt': example['src_txt'] + '\n'})
#eval_dataset = dataset_dict['validation'].remap_columns({'text1': 'src_txt', 'text2': 'tgt_txt'}) \
#    .map(lambda example: {'src_txt': example['src_txt'] + '\n'})

# Local datasets
train_dataset = MsDataset.load('./datasets/DuReader_robust-QG/train.csv')
eval_dataset = MsDataset.load('./datasets/DuReader_robust-QG/test.csv')

# Examples for self-defined datasets.
#train_dataset_dict = {"src_txt": ["input1", "input2"], "tgt_txt": ["output1", "output2"]}
#eval_dataset_dict = {"src_txt": ["input1", "input2"], "tgt_txt": ["output1", "output2"]}
#train_dataset = MsDataset(Dataset.from_dict(train_dataset_dict))
#eval_dataset = MsDataset(Dataset.from_dict(eval_dataset_dict))

max_epochs = 4

tmp_dir = './tmp'

num_warmup_steps = 50

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
    cfg.train.optimizer = {'type': 'AdamW', 'lr': 1e-4}
    cfg.train.dataloader = {
        'batch_size_per_gpu': 4,
        'workers_per_gpu': 1
    }
    cfg.train.hooks.append({
        'type': 'MegatronHook'
    })
    cfg.preprocessor.sequence_length = 512
    cfg.model.checkpoint_model_parallel_size = 1
    return cfg

kwargs = dict(
    model='nlp_gpt3_text-generation_1.3B',
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    max_epochs=max_epochs,
    work_dir=tmp_dir,
    cfg_modify_fn=cfg_modify_fn)

trainer = build_trainer(
    name=Trainers.gpt3_trainer, default_args=kwargs)
trainer.train()
