# 算想云 Web UI

这里是算想云 Web UI 代码，后端代码位于 aiadmin

## 编译构建

**推荐 node 版本：12-16**

```
# 安装依赖
npm install

# 启动服务 localhost:8013
npm run dev

# 打包项目
# 不管是将项目部署到 nginx 还是其他服务器，都需要先将项目打包

npm run build:prod

# 打包完成后会在根目录生成 dist 文件夹，我们需要将他上传到服务器中，
# 举例上传到/home/aiadmin中
# 在 nginx/conf/nginx.conf 添加配置
server {
    listen 80;
    server_name 域名;
    location / {
       root   /home/aiadmin/dist; #dist上传的路径
       index  index.html;
    }
}

#重启Nginx
systemctl restart nginx

重启 nginx 后，访问你的域名
```

## 国际化

```
marketplace/aiadmin-web/src/components/language/zh.js
marketplace/aiadmin-web/src/components/language/en.js
```
