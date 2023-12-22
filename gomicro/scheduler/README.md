# Scheduler 调度模块

## docker build and push
注意修改版本号
```bash
# 设置版本号环境变量
export VERSION=v0.0.1

# 构建镜像
docker build -t go-scheduler:$VERSION -f gomicro/scheduler/Dockerfile .

# 标记镜像
docker tag go-scheduler:$VERSION sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/go-scheduler:$VERSION

# 推送镜像
docker push sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/go-scheduler:$VERSION
```

## docker run
根据环境修改S_ENV，生产 prod 测试 test

注意修改版本号
```bash
# 设置版本号环境变量
export OLD_VERSION=v0.0.1
export NEW_VERSION=v0.0.2
export S_ENV=test

# 关闭老服务
docker rm -f go-scheduler-$OLD_VERSION

# 启动新服务
docker run -d -e SCHEDULER_ENV=$S_ENV -p 8090:80 -v /var/docker/scheduler:/var/log --name go-scheduler-$NEW_VERSION sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/go-scheduler:$NEW_VERSION
```

## Notes

### API 生成
在gomicro目录执行
```bash
goctl api go --style go_zero --home ./go-zero-template --dir ./scheduler --api ./scheduler.api
```