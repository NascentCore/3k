# Scheduler
scheduler目前包含了主要的业务逻辑，用户、任务调度、计费、集群信息管理。

微服务框架 go-zero
DB mysql

## 目录结构说明
核心目录有两个
1. 3k/cmd/scheduler
2. 3k/internal/scheduler

```bash
# 3k/cmd/scheduler
├── Dockerfile                    Dockerfile
├── README.md
├── etc
│   ├── scheduler-api.yaml        本地开发配置文件
│   ├── scheduler-api_k8s.yaml    私有化配置文件
│   ├── scheduler-api_prod.yaml   生产环境配置文件
│   └── scheduler-api_test.yaml   测试环境配置文件
├── ftl                           
│   ├── README.md
│   ├── email.ftl                 验证码邮件模版
│   └── token.ftl                 注册成功下发token邮件模版
├── scheduler.api                 go-zero接口描述文件
├── scheduler.go                  go-zero启动入口
└── scheduler.json                go-zero生成的OpenAPI文件
```

```bash
# 3k/internal/scheduler
├── config                        配置加载相关的代码
├── handler                       go-zero handler层代码，工具生成
├── job                           任务逻辑相关的代码
├── logic                         go-zero logic层代码，工具生成骨架，手工编写逻辑。业务逻辑都在这里。
├── model                         go-zero orm代码，工具生成
├── pay                           计费模块
├── svc                           go-zero service层代码，工具生成
├── types                         go-zero 接口参数、返回类型定义，工具生成
└── user                          用户模块
```

## docker build and push
pull request合并进main主干时，如果pr的title有`scheduler-build:vX.X.X`这样的内容，就会自动构建Docker镜像，tag设置为`go-scheduler:vX.X.X`

## go-zero代码生成命令
以下命令都是在3k根目录下执行

### 生成model代码
```bash
goctl model mysql datasource --style=go_zero \
--home ./tools/go-zero-template \
-dir="./internal/scheduler/model/" \
-url="nascentcore:Sxwl@6868@tcp(8.140.22.241:3306)/aiadmin" \
-table="sys_cpod_node"
```

### 生成api代码
```bash
goctl api go --style go_zero \
--home ./tools/go-zero-template \
--dir . \
--api ./cmd/scheduler/scheduler.api \
--service scheduler
```

### 生成swagger
```bash
goctl api plugin -plugin goctl-swagger="swagger -filename scheduler.json" -api ./cmd/scheduler/scheduler.api -dir ./cmd/scheduler
```