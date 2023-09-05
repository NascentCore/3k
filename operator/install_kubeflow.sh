#!/bin/bash -xe

# ${HOME}/kubernetes has files for installing kubernetes
workpath="${HOME}/kubernetes/kubeflow"
echo "Creating ${workpath} ..."
mkdir -p ${workpath}

cp local-path-config.yaml "${workpath}local-path-config.yaml"
cp local-path-storage.yaml "${workpath}local-path-storage.yaml"
cp local-sc.yaml "${workpath}local-sc.yaml"
cp fix_pvc.py "${workpath}fix_pvc.py"

cd ${workpath}

echo "请输入sudo密码："
read pwd
echo $pwd | sudo -S snap install kustomize
echo $pwd | sudo -S kubectl apply -f local-path-config.yaml
echo $pwd | sudo -S kubectl apply -f local-path-storage.yaml
echo $pwd | sudo -S kubectl apply -f local-sc.yaml
echo "local-path storageclass 成功部署:"
echo $pwd | sudo -S kubectl get po -n local-path-storage

while [ ! -d "${workpath}manifests" ]; do
    if git clone https://github.com/kubeflow/manifests.git; then
        break
    else
        echo "clone kubeflow 失败，等待5s后重试..."
        sleep 5
    fi
done

cd manifests

# 批量替换所有 gcr.io 镜像到 gcr.m.daocloud.io
find . -type f -exec sed -i 's/gcr.io/gcr.m.daocloud.io/g' {} +

python ../fix_pvc.py $PWD
echo '修改文件——添加sc 完成! '

cd ${HOME}/kubernetes

echo $pwd | sudo -S cp -r ./kubeflow /root
echo "运行以下命令将资源部署在k8s："
echo "cd ${workpath}/manifests"
echo "while ! kustomize build example | awk '!/well-defined/' | kubectl apply -f -; do echo 'Retrying to apply resources'; sleep 10; done"
