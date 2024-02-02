#bin/sh
OFFLINE_DIR=/home/cairong/3k/offline-deploy
ARTIFACT_DIR=${OFFLINE_DIR}/artifact

cp ../values/* ${ARTIFACT_DIR}/deploy/values
cp ../yaml_apps/* ${ARTIFACT_DIR}/deploy/yaml_apps
cp -a ../../3kctl/* ${ARTIFACT_DIR}/cli/
cp manifact.yaml ${OFFLINE_DIR}

cd ${OFFLINE_DIR}
export KKZONE=cn
sudo ./kk artifact export -m manifest.yaml -o 3k.tar.gz
mv 3k.tar.gz ${ARTIFACT_DIR}/packages
