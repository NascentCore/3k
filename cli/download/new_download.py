from kubernetes import client, config
from kubernetes.client.rest import ApiException


def create_pvc(api_instance, pvc_name, storage_size):
    body = {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {"name": pvc_name},
        "spec": {
            "accessModes": ["ReadWriteOnce"],
            "resources": {"requests": {"storage": storage_size}},
        },
    }

    try:
        api_response = api_instance.create_namespaced_persistent_volume_claim(
            namespace="default", body=body
        )
        print("PVC created. Status: %s" % str(api_response.status))
        return api_response
    except ApiException as e:
        print("Error creating PVC: %s" % e)
        raise e  # 让调用者处理异常



def create_golang_pod(api_instance, pod_name, pvc_name):
    container = client.V1Container(
        name="golang-container",
        image="golang:latest",
        command=["/bin/sh", "-c"],
        args=["echo 'Hello, PVC!' > /mnt/data/file.txt; sleep 3600"],
        volume_mounts=[client.V1VolumeMount(name="pvc-volume", mount_path="/mnt/data")],
    )

    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels={"app": pod_name}),
        spec=client.V1PodSpec(containers=[container], volumes=[client.V1Volume(
            name="pvc-volume",
            persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                claim_name=pvc_name
            ),
        )]),
    )

    spec = client.V1PodSpec(containers=[container], volumes=[client.V1Volume(
        name="pvc-volume",
        persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
            claim_name=pvc_name
        ),
    )])

    body = client.V1Pod(
        api_version="v1",
        kind="Pod",
        metadata=client.V1ObjectMeta(name=pod_name),
        spec=spec,
    )

    try:
        api_response = api_instance.create_namespaced_pod(
            namespace="default", body=body
        )
        print("Pod created. Status: %s" % str(api_response.status))
    except ApiException as e:
        print("Error creating Pod: %s" % e)


def create_readonly_pod(api_instance, pod_name, pvc_name):
    container = client.V1Container(
        name="readonly-container",
        image="alpine:latest",
        command=["/bin/sh", "-c"],
        args=["cat /mnt/data/file.txt"],
        volume_mounts=[client.V1VolumeMount(name="pvc-volume", mount_path="/mnt/data")],
    )

    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels={"app": pod_name}),
        spec=client.V1PodSpec(containers=[container], volumes=[client.V1Volume(
            name="pvc-volume",
            persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                claim_name=pvc_name
            ),
        )]),
    )

    spec = client.V1PodSpec(containers=[container], volumes=[client.V1Volume(
        name="pvc-volume",
        persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
            claim_name=pvc_name
        ),
    )])

    body = client.V1Pod(
        api_version="v1",
        kind="Pod",
        metadata=client.V1ObjectMeta(name=pod_name),
        spec=spec,
    )

    try:
        api_response = api_instance.create_namespaced_pod(
            namespace="default", body=body
        )
        print("Read-only Pod created. Status: %s" % str(api_response.status))
    except ApiException as e:
        print("Error creating Read-only Pod: %s" % e)


def main():
    config.load_kube_config()
    api_instance = client.CoreV1Api()

    pvc_name = "my-pvc"
    storage_size = "1Gi"
    try:
        create_pvc(api_instance, pvc_name, storage_size)
    except ApiException as e:
        print("Exception: %s" % e)


    golang_pod_name = "golang-pod"
    create_golang_pod(api_instance, golang_pod_name, pvc_name)

    readonly_pod_name = "readonly-pod"
    create_readonly_pod(api_instance, readonly_pod_name, pvc_name)


if __name__ == "__main__":
    main()
