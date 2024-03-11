import hashlib

# Placeholder constants - replace these with your actual values
MODEL = "model"
DATASET = "dataset"
OSS_USER_MODEL_PATH = "models/{}"
OSS_PUBLIC_MODEL_PATH = "models/public/{}"
OSS_USER_DATASET_PATH = "datasets/{}"
OSS_PUBLIC_DATASET_PATH = "datasets/public/{}"


def resource_to_oss_path(resource_type: str, resource: str) -> str:
    if resource_type == MODEL:
        if resource.startswith("user-"):
            return OSS_USER_MODEL_PATH.format(resource)
        else:
            return OSS_PUBLIC_MODEL_PATH.format(resource)
    elif resource_type == DATASET:
        if resource.startswith("user-"):
            return OSS_USER_DATASET_PATH.format(resource)
        else:
            return OSS_PUBLIC_DATASET_PATH.format(resource)

    return ""


def model_crd_name(oss_path: str) -> str:
    return f"model-storage-{hash_data(oss_path)}"


def dataset_crd_name(oss_path: str) -> str:
    return f"dataset-storage-{hash_data(oss_path)}"


def model_pvc_name(oss_path: str) -> str:
    return f"pvc-model-{hash_data(oss_path)}"


def dataset_pvc_name(oss_path: str) -> str:
    return f"pvc-dataset-{hash_data(oss_path)}"


def model_download_job_name(oss_path: str) -> str:
    return f"download-model-{hash_data(oss_path)}"


def dataset_download_job_name(oss_path: str) -> str:
    return f"download-dataset-{hash_data(oss_path)}"


def oss_path_to_oss_url(bucket: str, oss_path: str) -> str:
    return f"oss://{bucket}/{oss_path}"


def hash_data(data: str) -> str:
    hasher = hashlib.sha1()
    hasher.update(data.encode('utf-8'))
    hashed = hasher.hexdigest()
    return hashed[:16]
