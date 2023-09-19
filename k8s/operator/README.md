- 将该目录下的所有文件放在你的任意一个文件夹中
- chomd a+x install_kubeflow
- ./install_kubeflow
- 根据提示完成kubeflow的安装即可

## 只安装 Kubeflow Training Operator

```
git clone git@github.com:kubeflow/training-operator.git
cd training-operator
kubectl apply -k manifests/overlays/standalone
# 验证安装
kubectl get crd | grep kubeflow.org
mpijobs.kubeflow.org                                  2023-08-29T09:26:09Z
mxjobs.kubeflow.org                                   2023-08-29T09:26:10Z
paddlejobs.kubeflow.org                               2023-08-29T09:26:10Z
pytorchjobs.kubeflow.org                              2023-08-29T09:26:10Z
tfjobs.kubeflow.org                                   2023-08-29T09:26:10Z
xgboostjobs.kubeflow.org                              2023-08-29T09:26:10Z

kubectl get pods -n kubeflow
NAME                                 READY   STATUS    RESTARTS   AGE
training-operator-7f768bbbdb-5hnmf   1/1     Running   0          2d23h
```
