# Gateway 算想云网关

## 路由配置
etc目录下的配置文件中修改
```yaml
- Path: "/api/userJob/cpod_jobs"   // 原路径
  ToPath: "/cpod/job"              // 目标路径，不填则为同原路径
  Auth: Yes                        // 是否检查鉴权
  Server: scheduler                // 转发的目标服务
```

## docker build and push
pull request合并进main主干时，如果pr的title有`build:vX.X.X`这样的内容，就会自动构建Docker镜像，tag设置为`go-gateway:vX.X.X`

## docker run
根据环境修改S_ENV，生产 prod 测试 test

1. 注意修改版本号
2. dsn和AUTH_SECRET根据实际信息替换
```bash
# 设置版本号环境变量
export OLD_VERSION=v0.0.1
export NEW_VERSION=v0.0.2
export S_ENV=test

# dsn
export S_DSN="..."

# 关闭老服务
docker rm -f go-gateway-$OLD_VERSION

# 启动新服务
docker run -d \
-e GATEWAY_ENV=$S_ENV \
-e GATEWAY_DSN=$S_DSN \
-e AUTH_SECRET=... \
-p 8080:8080 \
-v /var/docker/gateway:/var/log \
--name go-gateway-$NEW_VERSION \
sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/go-gateway:$NEW_VERSION
```
