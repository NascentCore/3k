FROM swr.cn-east-3.myhuaweicloud.com/sxwl/torch-base:latest

RUN DEBIAN_FRONTEND=noninteractive apt install -y tzdata

#安装ssh服务
RUN apt-get install -y passwd openssh-server

#设置初始密码
RUN echo "root:root"|chpasswd

RUN mkdir /var/run/sshd
RUN sed -i 's/UsePAM yes/UsePAM no/g' /etc/ssh/sshd_config
RUN sed -i "s/.*PermitRootLogin.*/PermitRootLogin yes/" /etc/ssh/sshd_config
RUN sed -i 's/^.*Port 22$/Port 14343/g' /etc/ssh/sshd_config

ADD . /workspace
WORKDIR /workspace

ENTRYPOINT /workspace/entrypoint.sh
