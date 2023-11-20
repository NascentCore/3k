# CONDA

We use miniconda to manage python environment

```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

# If you are using bash
~/miniconda3/bin/conda init bash

# If you are using zsh
~/miniconda3/bin/conda init zsh

# Restart shell session to activate conda environment
```

## 常用命令
```
创建虚拟环境：
conda create -n your_env_name python=X.X（2.7、3.6等)

下面是创建python=3.6版本的环境，取名叫py36
conda create -n py36 python=3.6 

删除虚拟环境：
conda remove -n your_env_name(虚拟环境名称) --all

查看虚拟环境
conda env list
或 conda info -e

进入虚拟环境
conda activate conda_37

在当前环境中安装包：
conda install package
例如
conda install pytorch torchvision -c pytorch

在当前环境中删除包：
conda remove package

对指定虚拟环境中安装额外的包（不需要进入虚拟环境）
conda install -n your_env_name [package]即可安装package到your_env_name中
对指定虚拟环境中删除额外的包（不需要进入虚拟环境）
conda remove --name your_env_name  package_name

查看当前环境安装了哪些包
conda list 

检查更新当前conda
conda update conda
```
