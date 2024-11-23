# customization_job.py
import os
import datetime
import json
from request_utils import request

SERVICE = "ark"
ACTION = "ListModelCustomizationJobs"
VERSION = "2024-01-01"
REGION = "cn-beijing"
HOST = "open.volcengineapi.com"
CONTENT_TYPE = "application/json"
PATH = "/"

def list_model_customization_jobs(job_ids):
    data = {
        "PageNumber": 1,
        "PageSize": 10,
        "ProjectName": "default",
        "Filter": {
            "Ids": job_ids
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

