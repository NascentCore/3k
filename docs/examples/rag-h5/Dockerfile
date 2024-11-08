# 使用官方的 Nginx 镜像作为基础镜像
FROM nginx:alpine

# 将本地的网页文件复制到 Nginx 的默认静态文件目录
COPY index.html /usr/share/nginx/html/
COPY styles.css /usr/share/nginx/html/
COPY script.js /usr/share/nginx/html/
COPY 21book.png /usr/share/nginx/html/

# 如果有其他静态资源（如图标），请确保也复制到容器中
COPY voice-icon.png /usr/share/nginx/html/
COPY send-icon.png /usr/share/nginx/html/
COPY user-avatar.png /usr/share/nginx/html/
COPY assistant-avatar.png /usr/share/nginx/html/

# 暴露 Nginx 的默认端口
EXPOSE 80

# 启动 Nginx
CMD ["nginx", "-g", "daemon off;"]