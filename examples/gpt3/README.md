# GPT3

This sample is adapted from [ModeScope GPT3](
https://modelscope.cn/models/damo/nlp_gpt3_text-generation_1.3B/summary).

```
sudo apt install git-lfs
git lfs install

git clone https://modelscope.cn/damo/nlp_gpt3_text-generation_1.3B.git
cd nlp_gpt3_text-generation_1.3B
# This removes git metadata, and almost halves the size
rm -rf .git

# Make sure the directory layout is as follows
[0:29:25] yzhao:gpt3 git:(main) $ ls
Dockerfile  finetune_dureader.py  finetune_poetry.py  nlp_gpt3_text-generation_1.3B  README.md

# Build docker image
docker build . -t modelscope_gpt3:v002
docker tag modelscope_gpt3:v002 registry.cn-hangzhou.aliyuncs.com/sxwl-ai/modelscope_gpt3:v002
docker push registry.cn-hangzhou.aliyuncs.com/sxwl-ai/modelscope_gpt3:v002

docker run -it --gpus=all registry.cn-hangzhou.aliyuncs.com/sxwl-ai/modelscope_gpt3:v002 bash
root@a2bd9b4ad983:/workspace# ls
finetune_dureader.py  finetune_poetry.py  nlp_gpt3_text-generation_1.3B
root@a2bd9b4ad983:/workspace# torchrun finetune_poetry.py
root@a2bd9b4ad983:/workspace# torchrun --nproc_per_node=1 finetune_poetry.py
```
