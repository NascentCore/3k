import json
import os
import sys
import time
import requests


class APITester:
    """API Tester"""
    def __init__(self, base_url, token, feishu_webhook):
        self.base_url = base_url
        self.headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        self.feishu_webhook = feishu_webhook
        self.init_job_session()

    def post_job(self, endpoint, data):
        """Post job to API"""
        url = f"{self.base_url}/{endpoint}"
        response = requests.post(url, json=data, headers=self.headers)
        return response.json()

    def get_jobs(self, endpoint):
        """Get jobs from API"""
        url = f"{self.base_url}/{endpoint}"
        response = requests.get(url, headers=self.headers)
        res = response.json()
        return res['content'] if 'content' in res else res['data']

    def check_job_existence(self, job_id, endpoint):
        """Check if job exists in API"""
        jobs = self.get_jobs(endpoint)
        return any(job['jobName'] == job_id if 'jobName' in job else job['job_name'] == job_id if 'job_name' in job else job['service_name'] == job_id for job in jobs)

    def wait_for_status(self, job_id, endpoint, expected_status, interval=10, timeout=600):
        """Wait for job to reach expected status"""
        start_time = time.time()
        while True:
            jobs = self.get_jobs(endpoint)
            for job in jobs:
                if job['jobName'] == job_id if 'jobName' in job else job['job_name'] == job_id if 'job_name' in job else job['service_name'] == job_id:
                    status = job['status'] if 'status' in job else job['workStatus']
                    break
            if status == expected_status:
                print(f"Job {job_id} reached status: {status}")
                break
            elif status != 0 and status != 'deploying' and status != 'waitdeploy':
                self.send_alert(f"Job {job_id} reached unexpected status: {status}")
                break
            elif time.time() - start_time > timeout:
                self.send_alert(
                    f"Job {job_id} did not reach status {expected_status} \
                        within the timeout period."
                    )
                break
            else:
                print(f"Job {job_id} status is {status}. Waiting...")
                time.sleep(interval)

    def test_get_requests(self, endpoints):
        """Test GET requests"""
        results = {}
        for endpoint in endpoints:
            url = f"{self.base_url}/{endpoint}"
            response = requests.get(url, headers=self.headers)
            results[endpoint] = {
                'status_code': response.status_code,
                'is_empty': len(response.json()) == 0
            }
        return results

    def send_alert(self, message):
        """Send alert to Feishu"""
        headers = {'Content-Type': 'application/json'}
        data = {
            "msg_type": "text",
            "content": {
                "text": message
            }
        }
        response = requests.post(self.feishu_webhook, headers=headers, json=data)
        return response.json()

    def save_job_session(self, job_type, job_id):
        """Save job session to file"""
        self.job_session[job_type] = job_id
        with open('session.json', 'w') as f:
            json.dump(self.job_session, f)

    def load_job_session(self, job_type):
        """Load job session from file"""
        if job_type in self.job_session:
            return self.job_session[job_type]
        else:
            return None

    def init_job_session(self):
        """Initialize job session"""
        self.job_session = {}
        if os.path.exists('session.json'):
            with open('session.json', 'r') as f:
                self.job_session = json.load(f)

    def delete_job(self, job_id, endpoint, data):
        """Delete job from API"""
        if endpoint in ['finetune', 'training']:
            url = f"{self.base_url}/api/userJob/job_del"
            response = requests.post(url, headers=self.headers, data=json.dumps(data))
        else:
            url = f"{self.base_url}/api/job/{endpoint}?service_name={job_id}"
            response = requests.delete(url, headers=self.headers, data=json.dumps(data))
        if response.status_code == 200:
            print(f"Job {job_id} deleted successfully.")
            self.save_job_session(endpoint, None)


if __name__ == "__main__":
    url = os.getenv('SXCLOUD_API_URL')
    token = os.getenv('SXCLOUD_API_TOKEN')
    feishu_webhook = os.getenv('FEISHU_WEBHOOK')

    if not url or not token or not feishu_webhook:
        print("API Url, Token or Feishu Webhook not found.")
        sys.exit()

    api_tester = APITester(url, token, feishu_webhook)

    try:
        # 测试微调任务
        finetune_job_id = api_tester.load_job_session('finetune')
        if not finetune_job_id:
            data = {
                "model":"google/gemma-2b-it",
                "training_file":"llama-factory/alpaca_data_zh_short",
                "gpu_model":"NVIDIA-GeForce-RTX-3090",
                "gpu_count":1,
                "hyperparameters":{
                    "n_epochs":"3.0",
                    "batch_size":"4",
                    "learning_rate_multiplier":"5e-5"
                    },
                "model_is_public":True,
                "dataset_is_public":True
                }
            finetune_response = api_tester.post_job('api/job/finetune', data)
            finetune_job_id = finetune_response['job_id']
            api_tester.save_job_session('finetune', finetune_job_id)
        print(finetune_job_id)
        assert api_tester.check_job_existence(
            finetune_job_id,
            'api/job/training?current=1&size=100'
            ), "Finetune job not found."
        api_tester.wait_for_status(finetune_job_id, 'api/job/training?current=1&size=100', 2)

        # 测试训练任务
        # training_job_id = api_tester.load_job_session('training')
        # if not training_job_id:
        #     data = {
        #         "ckptPath":"/workspace/checkpoint",
        #         "ckptVol":10240,
        #         "modelPath":"/workspace/saved-model",
        #         "modelVol":10240,
        #         "gpuNumber":1,
        #         "gpuType":"NVIDIA-GeForce-RTX-3090",
        #         "imagePath":
        #             "sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/modelscope_gpt3_2h16g4tp:v2",
        #         "jobType":"Pytorch",
        #         "pretrainedModelId":"model-storage-a175dee1c23700ff",
        #         "pretrainedModelPath":"/workspace/nlp_gpt3_text-generation_1.3B",
        #         "datasetId":"dataset-storage-e505673234392893",
        #         "datasetPath":"/workspace/chinese-poetry-collection",
        #         "runCommand":"torchrun --nnodes=1 --nproc_per_node=1 finetune_poetry.py",
        #         "modelIsPublic":True,
        #         "datasetIsPublic":True
        #         }
        #     training_response = api_tester.post_job('api/job/training', data)
        #     training_job_id = training_response['job_id']
        #     api_tester.save_job_session('training', training_job_id)
        # print(training_job_id)
        # assert api_tester.check_job_existence(
        #     training_job_id,
        #     'api/job/training?current=1&size=10'
        #     ), "Training job not found."
        # api_tester.wait_for_status(training_job_id, 'api/job/training?current=1&size=10', 2)

        # 测试推理任务
        inference_job_id = api_tester.load_job_session('inference')
        if not inference_job_id:
            data = {
                "model_name": "google/gemma-2b-it", 
                "gpu_model": "NVIDIA-GeForce-RTX-3090", 
                "gpu_count": 1
                }
            inference_response = api_tester.post_job('api/job/inference', data)
            inference_job_id = inference_response['service_name']
            api_tester.save_job_session('inference', inference_job_id)
        print(inference_job_id)
        assert api_tester.check_job_existence(
            inference_job_id,
            'api/job/inference'
            ), "Inference job not found."
        api_tester.wait_for_status(inference_job_id, 'api/job/inference', 'deployed')

        # JupyterLab 实例测试
        jupyterlab_job_id = api_tester.load_job_session('jupyterlab')
        if not jupyterlab_job_id:
            instance_name = f"test-{int(time.time())}"
            data = {
                "instance_name":instance_name,
                "cpu_count":2,
                "memory":2147483648,
                "gpu_count":1,
                "gpu_product":"NVIDIA-GeForce-RTX-3090",
                "data_volume_size":1073741824,
                "model_id":"model-storage-0ce92f029254ff34",
                "model_path":"/model",
                "user_id":181,
                "model_name":"google/gemma-2b-it"
                }
            jupyterlab_response = api_tester.post_job('api/job/jupyterlab', data)
            jupyterlab_job_id = jupyterlab_response['message'].split(':')[1].split()[0]
            api_tester.save_job_session('jupyterlab', jupyterlab_job_id)
        print(jupyterlab_job_id)
        assert api_tester.check_job_existence(
            jupyterlab_job_id,
            'api/job/jupyterlab'
            ), "JupyterLab job not found."
        api_tester.wait_for_status(jupyterlab_job_id, 'api/job/jupyterlab', 'deployed')

        # 定义需要测试的 GET 请求端点
        endpoints = [
            'api/resource/models',
            'api/resource/gpus',
            'api/resource/datasets'
        ]

        # 进行测试
        test_results = api_tester.test_get_requests(endpoints)
        for endpoint, result in test_results.items():
            print(f"Endpoint: {endpoint} - Status Code: {result['status_code']} \
                  - Is Empty: {'Yes' if result['is_empty'] else 'No'}"
                  )
            if result['status_code'] != 200 or result['is_empty']:
                api_tester.send_alert(f"Endpoint {endpoint} returned status code \
                                      {result['status_code']} or is empty.")

        # 完成测试，删除测试任务，发送通知
        api_tester.delete_job(finetune_job_id, 'finetune', {"job_id":finetune_job_id})
        # api_tester.delete_job(training_job_id, 'training', {"job_id":training_job_id})
        api_tester.delete_job(inference_job_id, 'inference', {"service_name":inference_job_id})
        api_tester.delete_job(jupyterlab_job_id, 'jupyterlab', {"job_name":jupyterlab_job_id})
        api_tester.send_alert("API Tester finished testing.")

    except AssertionError as e:
        api_tester.send_alert(f"Assertion Error: {str(e)}")

    except Exception as e:
        api_tester.send_alert(f"Unexpected Error: {str(e)}")
