## AiChat
- 使用 fastapi 搭建本地 server，转发前端会话请求以及完成图片上传
- 前端为 H5 页面

## 构建镜像
```bash
git clone https://github.com/NascentCore/3k.git
cd chatui
docker buildx build --platform linux/amd64 -t sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/chatui:latest .
```