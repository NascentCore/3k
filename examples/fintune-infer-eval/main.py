import json
import time
import requests
from sxwl_client import ApiClient, Configuration
from sxwl_client.api import SchedulerApiApi
from sxwl_client.models.finetune_req import FinetuneReq
from sxwl_client.models.inference_deploy_req import InferenceDeployReq
from config import API_HOST, API_TOKEN, EVALUATION_CONFIG, SX_USER_ID, FINETUNE_CONFIG
from calculate_score import rouge_score
import jieba  # 因为rouge_score函数依赖jieba

def log_message(message: str) -> None:
    """统一的日志输出函数"""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}", flush=True)

def main():
    # 初始化配置
    configuration = Configuration(
        host=API_HOST,
    )

    with ApiClient(configuration) as api_client:
        api_client.default_headers["Authorization"] = API_TOKEN
        api_instance = SchedulerApiApi(api_client)
        
        # 查询模型
        model_name = FINETUNE_CONFIG["model_name"]
        model_info = api_instance.model_by_name(model_name=model_name)
        log_message(f"Model Info: {model_info}")

        # 查询数据集
        dataset_name = FINETUNE_CONFIG["dataset_name"]
        dataset_info = api_instance.dataset_by_name(dataset_name=dataset_name)
        log_message(f"Dataset Info: {dataset_info}")
        
        # 更新 FINETUNE_CONFIG
        finetune_params = FINETUNE_CONFIG.copy()
        finetune_params.update({
            "model_id": model_info.model_id,
            "model_name": model_info.model_name,
            "model_size": model_info.model_size,
            "model_is_public": model_info.model_is_public,
            "model_template": model_info.model_template,
            "model_category": model_info.model_category,
            "model_meta": model_info.model_meta,
            "dataset_id": dataset_info.dataset_id,
            "dataset_name": dataset_info.dataset_name,
            "dataset_size": dataset_info.dataset_size,
            "dataset_is_public": dataset_info.dataset_is_public,
        })
        
        # 记录微调开始时间
        finetune_start_time = time.time()
        
        # 创建微调任务
        finetune_request = FinetuneReq(**finetune_params)
        finetune_response = api_instance.finetune(sx_user_id=SX_USER_ID, body=finetune_request)
        log_message(f"Finetune Response: {finetune_response}")

        # 轮询微调任务状态
        while True:
            job_id=finetune_response.job_id
            task_status = api_instance.finetune_status(sx_user_id=SX_USER_ID, job_id=job_id)
            log_message(f"Task Status: {task_status}")
            if task_status.status == "succeeded":
                finetune_end_time = time.time()
                log_message("Finetune completed")
                break
            else:
                log_message(f"Finetune is {task_status.status}, waiting...")
                time.sleep(10)

        # 记录部署开始时间
        deploy_start_time = time.time()
        
        # 部署推理服务
        inference_dict = {
            **FINETUNE_CONFIG,
            "model_id": model_info.model_id,
            "model_name": model_info.model_name,
            "model_size": model_info.model_size,
            "model_is_public": model_info.model_is_public,
            "model_template": model_info.model_template,
            "model_category": model_info.model_category,
            "model_meta": model_info.model_meta,
            "adapter_id": task_status.adapter_id,
            "adapter_name": task_status.adapter_name,
            "adapter_size": task_status.adapter_size,
            "adapter_is_public": task_status.adapter_is_public
        }
        inference_params = InferenceDeployReq(**inference_dict)

        deploy_response = api_instance.inference_deploy(sx_user_id=SX_USER_ID, body=inference_params)
        log_message(f"Deploy Response: {deploy_response}")

        while True:
            inference_status = api_instance.inference_status(sx_user_id=SX_USER_ID, service_name=deploy_response.service_name)
            log_message(f"Inference Status: {inference_status}")
            if inference_status.status == "running":
                deploy_end_time = time.time()
                log_message("Inference completed")
                log_message(f"Inference Service: {inference_status.service_name}")
                break
            else:
                log_message(f"Inference is {inference_status.status}, waiting...")
                time.sleep(10)

        # 计算并输出耗时统计
        finetune_duration = finetune_end_time - finetune_start_time
        deploy_duration = deploy_end_time - deploy_start_time
        total_duration = finetune_duration + deploy_duration
        
        log_message(f"\nInference Service: {inference_status.service_name} is ready")
        log_message(f"Inference API URL: {inference_status.api_url}")
        log_message(f"Inference Chat URL: {inference_status.chat_url}")
        log_message("\n耗时统计:")
        log_message(f"微调训练耗时: {finetune_duration:.2f} 秒 ({finetune_duration/60:.2f} 分钟)")
        log_message(f"推理部署耗时: {deploy_duration:.2f} 秒 ({deploy_duration/60:.2f} 分钟)")
        log_message(f"总耗时: {total_duration:.2f} 秒 ({total_duration/60:.2f} 分钟)")

        # 使用评测集调用推理接口生成数据
        # inference_api_url = inference_status.api_url
        inference_api_url = "http://test.llm.sxwl.ai:30004/inference/api/infer-6451530b-792b-4ad0-b08f-8ec37a3fe483/v1/chat/completions"
        with open(EVALUATION_CONFIG["evaluation_file"], "r") as file:
            eval_data = json.load(file)
            
            # 创建一个列表存储所有的评测结果
            evaluation_results = []
            
            # 对每个评测项进行处理
            for idx, eval_item in enumerate(eval_data):
                log_message(f"处理第 {idx + 1}/{len(eval_data)} 个评测项")
                
                try:
                    response = requests.post(
                        inference_api_url, 
                        json={
                            "model": EVALUATION_CONFIG["model"],
                            "messages": [
                                {
                                    "role": "user",
                                    "content": eval_item["user_content"]
                                },
                            ],
                            "temperature": EVALUATION_CONFIG["temperature"],
                            "top_k": EVALUATION_CONFIG["top_k"],
                            "stream": EVALUATION_CONFIG["stream"]
                        }
                    )
                    
                    result = response.json()
                    
                    # 保存评测结果
                    evaluation_results.append({
                        "user_content": eval_item["user_content"],
                        "original_response": eval_item["original_response"],
                        "model_response": result["choices"][0]["message"]["content"],
                        "score": rouge_score(eval_item["original_response"], result)
                    })
                    
                    log_message(f"评测项 {idx + 1} 完成")
                    
                except Exception as e:
                    log_message(f"处理评测项 {idx + 1} 时发生错误: {str(e)}")
                    continue
                
                # 添加短暂延迟，避免请求过于频繁
                time.sleep(1)
            
            # 将评测结果保存到文件
            output_file = EVALUATION_CONFIG["output_file"]
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
            
            # 打印一下评测结果的最高score、最低score、平均score
            scores = [result["score"] for result in evaluation_results]
            log_message(f"评测结果最高score: {max(scores)}")
            log_message(f"评测结果最低score: {min(scores)}")
            log_message(f"评测结果平均score: {sum(scores) / len(scores)}")
            
            log_message(f"评测结果已保存到: {output_file}")

        # 终止推理服务
        api_instance.inference_stop(service_name=deploy_response.service_name)
        log_message(f"Inference Service: {deploy_response.service_name} is stopped")

if __name__ == "__main__":
    main()