# AIAdmin

IntelJIDEA 导入 aiadmin 目录，AppRun 运行主命令

aiadmin-system/src/main/java/nascentcore/ai/AppRun.java 是可执行程序的入口

SprintBoot 开发的云市场入口的后端业务逻辑；主要是 CRUD（数据库增删改查的操作）

1. 注册用户
2. Email 验证 sendEmail
4. AccessKey 发送到用户 Email 作为用户调用 API 的 key

GCP部署云市场后端java（https://tricorder.feishu.cn/wiki/QryJwf9EiirI9Tkxxi6c8Wjhncg）按照指南启动本地服务；然后打开 http://8.140.22.241:8012/doc.html 就可以
