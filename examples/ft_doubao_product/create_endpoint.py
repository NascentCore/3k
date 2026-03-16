# customization_job.py
import os
import datetime
import json
from request_utils import request

SERVICE = "ark"
ACTION = "CreateEndpoint"
VERSION = "2024-01-01"
REGION = "cn-beijing"
HOST = "open.volcengineapi.com"
CONTENT_TYPE = "application/json"
PATH = "/"

def create_endpoint(endpoint_name, custom_model_id):
    #data = {
    #        "Name": endpoint_name,
    #        "ModelReference": {
    #          "CustomModelId": custom_model_id,
    #        },
    #        "RateLimit": {
    #          "Rpm": 651,
    #          "Tpm": 798
    #        }
    #     }

    data = {
            "Name": endpoint_name,
            "ModelReference": {
              "CustomModelId": custom_model_id,
            },
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

