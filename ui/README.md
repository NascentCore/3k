# 中科苏州 Web UI

这里是中科苏州 Web UI 代码

## 国际化配置目录

src/locales

## 编译构建

\*\*推荐 node 版本：18 及以上

```
# 安装依赖
yarn install

# 启动服务 localhost:8000
yarn start

# 打包项目
# 不管是将项目部署到 nginx 还是其他服务器，都需要先将项目打包

yarn build

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
