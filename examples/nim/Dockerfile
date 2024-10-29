FROM registry.cn-hangzhou.aliyuncs.com/miclon/py-nodejs:latest as build
WORKDIR /app
COPY . /app
RUN pnpm config set registry https://registry.npmmirror.com/
RUN cd /app/chat-ui/web && pnpm install && pnpm run build

FROM sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/llamafactory:inference-v2.1
WORKDIR /app
COPY chat-ui/api/requirements.txt /app/web/requirements.txt
RUN pip install --no-cache-dir -r web/requirements.txt
COPY --from=build /app/chat-ui/web/dist /app/web/api/dist
COPY run.sh /app
COPY chat-ui/api /app/web/api
COPY model/meta-llama-3.1-8b-instruct /mnt/models
CMD ["/bin/bash", "run.sh"]

