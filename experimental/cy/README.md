git clone git@github.com:NascentCore/3k.git
cd 3k

# 如果已经有本地 repo，则可以先与远程主干同步
git pull

# 为自己的改动创建新的 branch
git checkout -b first_pr

# 假设 username 是 xueyou
mkdir -p home/xueyou
# 用你最习惯的编辑器，创建 REAMD.md，写一段自我介绍
vi home/xueyou/README.md

# Xueyou // 标题

我是张学友；。。。

# 完成编辑后 commit 
git add .
git commit -m "<username>'s first pr"

# Push 到远程目录
git push -u origin HEAD
