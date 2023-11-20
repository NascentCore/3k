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
