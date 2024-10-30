# 在线安装

##  前提条件
1. 算想三千基于 Kubernetes 运行，请确保 Kubernetes 集群版本为 1.24 及以上；
2. 已安装 Helm 3.7.1 及以上版本；

## 部署步骤
1. 克隆项目仓库
```bash
git clone https://github.com/NascentCore/3k.git
cd 3k
```

2. 3k 平台使用 juicefs-csi + 对象存储作为后端存储，此处部署 minio 和 redis 作为示例（可选）：
```bash
# 安装 minio
sudo mkdir -p /data/minio/config
sudo mkdir -p /data/minio/data
sudo docker run -p 9000:9000 -p 9090:9090 --name minio \
    -d --restart=always \
    -e "MINIO_ACCESS_KEY=admin" \
    -e "MINIO_SECRET_KEY=admin123456" \
    -v /data/minio/data:/data \
    -v /data/minio/config:/root/.minio \
    sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/minio:latest \
    server /data --console-address ":9090"

# 安装 redis
sudo mkdir -p /data/redis/data
cat <<EOF > /data/redis/redis.conf
daemonize no
port 6379
bind 0.0.0.0
appendonly yes
EOF

sudo docker run -p 6379:6379 --name some-redis -d \
    -v /data/redis/redis.conf:/etc/redis/redis.conf \
    -v /data/redis/data:/data \
    sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/redis:5.0 \
    redis-server /etc/redis/redis.conf
```

3. 修改 `3kctl/conf/softwares.yaml` 文件，按需选择安装的组件，具体配置参考文件内注释（可选）

4. 修改 `deployment/values_online/juicefs-csi.yaml` 文件，配置上面示例中 minio 以及 redis 的信息，此处如果使用已有的对象存储，则替换为相应的配置

5. 修改 `deployment/values_online/sxcloud.yaml` 文件，具体配置参考文件内注释

6. 修改 `deployment/values_online/cpodoperator.yaml` 文件，具体配置参考文件内注释
   
7. 部署 sx3k 平台：
```bash
cd 3k/deployment/online
make
```