#bin/sh
OFFLINE_DIR=/home/cairong/3k/offline-deploy
ARTIFACT_DIR=${OFFLINE_DIR}/artifact

cp ../values/* ${ARTIFACT_DIR}/deploy/values
cp ../yaml_apps/* ${ARTIFACT_DIR}/deploy/yaml_apps
cp -a ../../3kctl/* ${ARTIFACT_DIR}/cli/
cp ../../3kctl/conf/softwares.yaml ${ARTIFACT_DIR}/conf/
cp manifest.yaml ${OFFLINE_DIR}
chmod +x ${ARTIFACT_DIR}/cli/3kctl.py

mkdir -p ${ARTIFACT_DIR}/deploy/models
mkdir -p ${ARTIFACT_DIR}/deploy/datasets
grep -v '^#' ../models/pre_loaded.txt | while read line;do basename $line;done | sed 's/.git$//' > ${ARTIFACT_DIR}/deploy/models/pre_load.txt
grep -v '^#' deployment/datasets/pre_loaded.txt | while read line;do basename $line;done | sed 's/.git$//' > ${ARTIFACT_DIR}/deploy/datasets/pre_load.txt

cd ${OFFLINE_DIR}
export KKZONE=cn
sudo ./kk artifact export -m manifest.yaml -o 3k.tar.gz
mv 3k.tar.gz ${ARTIFACT_DIR}/packages
