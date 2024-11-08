# encoding: UTF-8
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
# 载入训练数据
train_dataset = MsDataset.load('afqmc_small', split='train')
# 载入评估数据
eval_dataset = MsDataset.load('afqmc_small', split='validation')
# 指定文本分类模型
model_id = 'damo/nlp_structbert_sentence-similarity_chinese-tiny'

# 指定工作目录
tmp_dir = "/home/cy/workspace/first_modelscope/work_temp"

# 配置参数
kwargs = dict(
        model=model_id,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        work_dir=tmp_dir)

trainer = build_trainer(default_args=kwargs)
trainer.train()

# 模型的评估,直接调用trainer.evaluate，可以传入train阶段生成的ckpt
# 也可以不传入参数，直接验证model
metrics = trainer.evaluate(checkpoint_path=None)
print(metrics)
