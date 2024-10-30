mkdir -p /opt/3k/deploy
cp -a ../../3kctl /opt/3k/cli
cp -a ../../3kctl/conf /opt/3k/
cp -a ../values_online /opt/3k/deploy/
cp -a ../yaml_apps_online /opt/3k/deploy/
wget https://sxwl-ai.oss-cn-beijing.aliyuncs.com/artifacts/packages.tgz -O /tmp/packages.tgz
tar zxf /tmp/packages.tgz -C /opt/3k/
cp /opt/3k/bin/kubectl-assert /usr/local/bin/
apt update
apt install -y python3.8-venv
python3 -m venv /opt/.venv
/opt/.venv/bin/pip install -r /opt/3k/cli/requirements.txt
ln -s /opt/3k/cli/3kctl.py /usr/local/bin/3kctl
chmod +x /usr/local/bin/3kctl
helm repo add harbor https://sxwl-ai.oss-cn-beijing.aliyuncs.com/charts/
helm repo update
cd /opt/3k
3kctl deploy install --online operators