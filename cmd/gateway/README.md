# Gateway 算想云网关
gateway实现了jwt验签、路由转发。

微服务框架 go-zero

## 目录结构
3k/cmd/gateway
```bash
├── Dockerfile                   Dockerfile
├── README.md
├── etc
│   ├── gateway-api.yaml         本地开发路由配置
│   ├── gateway-api_k8s.yaml     私有化部署路由配置
│   ├── gateway-api_prod.yaml    生产环境路由配置
│   └── gateway-api_test.yaml    测试环境路由配置
└── gateway.go                   go-zero入口文件
```

3k/internal/gateway
```bash
├── config
│   └── config.go                go-zero配置加载相关代码
├── gateway
│   ├── config.go                路由转发的配置
│   ├── handler.go               路由逻辑、jwt验签逻辑
│   └── match.go                 路由匹配
└── svc
    └── service_context.go       go-zero service层代码，框架生成
```

## 路由配置
etc目录下的配置文件中修改
```yaml
- Path: "/api/userJob/cpod_jobs"   // 原路径
  ToPath: "/cpod/job"              // 目标路径，不填则为同原路径
  Auth: Yes                        // 是否检查鉴权
  Server: scheduler                // 转发的目标服务
```

## docker build and push
pull request合并进main主干时，如果pr的title有`gateway-build:vX.X.X`这样的内容，就会自动构建Docker镜像，tag设置为`go-gateway:vX.X.X`

