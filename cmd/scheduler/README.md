# Scheduler 调度模块

## docker build and push
pull request合并进main主干时，如果pr的title有`build:vX.X.X`这样的内容，就会自动构建Docker镜像，tag设置为`go-scheduler:vX.X.X`

## docker run
根据环境修改S_ENV，生产 prod 测试 test

注意修改版本号
```bash
# 设置版本号环境变量
export OLD_VERSION=v0.0.1
export NEW_VERSION=v0.0.2
export S_ENV=test

# dsn
export S_DSN="..."

# 关闭老服务
docker rm -f go-scheduler-$OLD_VERSION

# 启动新服务
docker run -d \
-e SCHEDULER_ENV=$S_ENV \
-e SCHEDULER_DSN=$S_DSN \
-p 8090:80 \
-v /var/docker/scheduler:/var/log \
--name go-scheduler-$NEW_VERSION \
sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/go-scheduler:$NEW_VERSION
```

## Notes

### API 生成
在3k根目录执行
```bash
goctl api go --style go_zero --home ./tools/go-zero-template --api ./cmd/scheduler/scheduler.api --dir ./internal/scheduler 
```
