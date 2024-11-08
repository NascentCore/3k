# encoding: UTF-8
import argparse
import os

from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--local_rank', type=int, default=0)
args = parser.parse_args()
if 'LOCAL_RANK' not in os.environ:
    os.environ['LOCAL_RANK'] = str(args.local_rank)

# 指定工作目录
tmp_dir = "/home/cy/workspace/first_modelscope/work_temp"
# 载入训练数据
train_dataset = MsDataset.load('afqmc_small', split='train')
# 载入评估数据
eval_dataset = MsDataset.load('afqmc_small', split='validation')

model_id = 'damo/nlp_structbert_sentence-similarity_chinese-tiny'



# 配置参数
kwargs = dict(
        model=model_id,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        work_dir=tmp_dir,
        launcher='pytorch'  # 分布式启动方式
)

# 实例化trainer对象
trainer = build_trainer(default_args=kwargs)

# 调用train接口进行训练
trainer.train()

metrics = trainer.evaluate(checkpoint_path=None)
print(metrics)
