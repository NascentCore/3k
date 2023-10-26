# modeluploader

此目录是为了构建上传模型文件的可执行文件。

CPod Manager在创建一个MPIJob的同时，会创建一个K8S Job， 此Job会执行这个可执行文件。

## modeluploadjob的工作流程
1. 监控MPIJob的工作状态。
2. 如果状态转为完成（代表模型已经训练完成）， 从Ceph中读取训练结果，上传至S3存储。
3. 上传完成后Job结束。

# 部署
## 首先要创建Secret
kubectl create secret generic akas4oss -n cpod --from-literal=AK=[AccessKey] --from-literal=AS=[AccessSecret]
