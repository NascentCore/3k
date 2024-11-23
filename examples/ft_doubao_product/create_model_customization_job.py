# customization_job.py
import os
import datetime
import json
from request_utils import request

SERVICE = "ark"
ACTION = "CreateModelCustomizationJob"
VERSION = "2024-01-01"
REGION = "cn-beijing"
HOST = "open.volcengineapi.com"
CONTENT_TYPE = "application/json"
PATH = "/"

def create_model_customization_job():
    data = {
        "Name": "sxwl_ft_test1",
        "CustomizationType": "FinetuneLoRA",
        "ModelReference": {
            "FoundationModel": {
                "Name": "doubao-pro-32k",
                "ModelVersion": "240828"
            }
        },
        "Hyperparameters": [
            {"Name": "epoch", "Value": "1"},
            {"Name": "learning_rate", "Value": "0.0001"}
        ],
        "SaveModelLimit": 1,
        "Data": {
            "TrainingSet": {
                "TosBucket": "ark-auto-2101752952-cn-beijing-default",
                "TosPaths": [
                    "model-customization-job-training-set/1731424681570/doubao_wenan_tiny_train_data.jsonl"
                ]
            }
        }
    }
    data_json = json.dumps(data)
    current_date = datetime.datetime.utcnow()
    response = request(
        method="POST",
        date=current_date,
        query={"Action": ACTION, "Version": VERSION},
        header={},
        ak=os.getenv("AK", ""),
        sk=os.getenv("SK", ""),
        body=data_json,
        service=SERVICE,
        region=REGION,
        host=HOST,
        path=PATH,
        content_type=CONTENT_TYPE,
    )
    response_content = response.content.decode('utf-8')
    return response_content

