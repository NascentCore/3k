# 阶段 1: 构建
# 使用Node官方镜像作为构建环境
FROM node:18 as build-stage

# 设置工作目录
WORKDIR /app

# 复制package.json和yarn.lock文件到工作目录
COPY package.json ./

# 安装项目依赖
RUN yarn install

# 复制剩余的项目文件到工作目录
COPY . .

# 构建项目
RUN yarn build

# 阶段 2: 部署
# 使用官方Nginx镜像作为运行环境
FROM nginx:latest as production-stage

# 将构建好的文件从构建阶段复制到Nginx容器
COPY --from=build-stage /app/dist /usr/share/nginx/html

# （可选）如果你有自定义的Nginx配置文件，可以用它来替换默认的配置
COPY --from=build-stage /app/nginx.conf /etc/nginx/conf.d/default.conf

# 暴露容器的80端口
EXPOSE 80

# 使用Nginx镜像的默认启动命令启动Nginx
CMD ["nginx", "-g", "daemon off;"]
