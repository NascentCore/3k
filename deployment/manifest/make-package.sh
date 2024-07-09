#!/bin/bash
ARTIFACT_DIR=./artifacts
PACKAGES_URL=https://sxwl-ai.oss-cn-beijing.aliyuncs.com/artifacts/packages.tgz

# 创建目录
mkdir -p ${ARTIFACT_DIR}/cli ${ARTIFACT_DIR}/conf ${ARTIFACT_DIR}/deploy/values ${ARTIFACT_DIR}/deploy/yaml_apps ${ARTIFACT_DIR}/deploy/models ${ARTIFACT_DIR}/deploy/datasets ${ARTIFACT_DIR}/packages

# 下载并解压包
wget -O /tmp/packages.tgz ${PACKAGES_URL}
tar -zxf /tmp/packages.tgz -C ${ARTIFACT_DIR}
rm -f /tmp/packages.tgz

# 复制文件
cp ../values/* ${ARTIFACT_DIR}/deploy/values
cp ../yaml_apps/* ${ARTIFACT_DIR}/deploy/yaml_apps
cp -a ../../3kctl/* ${ARTIFACT_DIR}/cli/
cp ../../3kctl/conf/* ${ARTIFACT_DIR}/conf/
cp manifest.yaml ${ARTIFACT_DIR}
chmod +x ${ARTIFACT_DIR}/cli/3kctl.py

# 生成预加载文件列表
grep -v '^#' ../models/pre_loaded.txt | while read name line; do basename $name; done | sed 's/.git$//' > ${ARTIFACT_DIR}/deploy/models/pre_load.txt
grep -v '^#' ../datasets/pre_loaded.txt | while read name line; do basename $name; done | sed 's/.git$//' > ${ARTIFACT_DIR}/deploy/datasets/pre_load.txt

# 导出工件
cd ${ARTIFACT_DIR}
export KKZONE=cn
sudo ./bin/kk artifact export -m manifest.yaml -o 3k.tar.gz
mv 3k.tar.gz ./packages

# 打包离线包
cd ..
tar -zcf 3k-artifacts.tar.gz artifacts
mv 3k-artifacts.tar.gz ${ARTIFACT_DIR}