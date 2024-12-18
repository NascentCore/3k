import os
import json

from typing import Dict, Optional, List
import logging

from pathlib import Path
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse

from ray import serve

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_engine import LoRAModulePath
from vllm.entrypoints.openai.serving_engine import BaseModelPath
from vllm.utils import FlexibleArgumentParser

logger = logging.getLogger("ray.serve")

app = FastAPI()

def parse_vllm_args(cli_args: Dict[str, str]):
    """Parses vLLM args based on CLI inputs.

    Currently uses argparse because vLLM doesn't expose Python models for all of the
    config options we want to support.
    """
    parser = FlexibleArgumentParser(description="vLLM CLI")
    parser = make_arg_parser(parser)
    arg_strings = []
    for key, value in cli_args.items():
        if key == "extra-params":
            try:
                extra_params = json.loads(value)
                for param_key, param_value in extra_params.items():
                    if str(param_value) == "":
                        arg_strings.append(f"--{param_key}")
                    else:
                        arg_strings.extend([f"--{param_key}", str(param_value)])
            except json.JSONDecodeError:
                logger.error(f"无法解析 extra-params: {value}")
                raise
        else:
            arg_strings.extend([f"--{key}", str(value)])
    logger.info(arg_strings)
    parsed_args = parser.parse_args(args=arg_strings)
    return parsed_args


@serve.deployment(
    name="VLLMDeployment",
    # autoscaling_config={
    #     "min_replicas": 1,
    #     "max_replicas": 4,
    #     "target_ongoing_requests": 3,
    # },
    # ray_actor_options={
    #     "num_gpus": 4,  # 指定需要 1 个 GPU
    # },
    # max_ongoing_requests=5,
)
@serve.ingress(app)
class VLLMDeployment:
    def __init__(
        self,
        cli_args: Dict[str, str],  # 修改为接收原始参数
        response_role: str = None,
        lora_modules: Optional[List[LoRAModulePath]] = None,
        chat_template: Optional[str] = None,
    ):
        # 将解析逻辑移到 worker 节点的初始化函数中
        parsed_args = parse_vllm_args(cli_args)
        engine_args = AsyncEngineArgs.from_cli_args(parsed_args)
        engine_args.served_model_name = parsed_args.served_model_name
        if isinstance(engine_args.served_model_name, list):
            engine_args.served_model_name = engine_args.served_model_name[0] 
        engine_args.worker_use_ray = True
        engine_args.trust_remote_code = True

        model_path = Path(engine_args.model)
        if not model_path.exists():
            raise ValueError(f"模型路径不存在: {model_path}, engine_args.model: {engine_args.model}")
        
        logger.info(f"Starting with engine args: {engine_args}")
        self.openai_serving_chat = None
        self.engine_args = engine_args
        self.response_role = parsed_args.response_role
        self.lora_modules = parsed_args.lora_modules
        self.chat_template = parsed_args.chat_template
        try:
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        except Exception as e:
            logger.error(f"初始化 LLM 引擎失败: {str(e)}")
            raise
        logger.info("初始化 LLM 成功")

    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        if not self.openai_serving_chat:
            model_config = await self.engine.get_model_config()
            
            # 创建两个 BaseModelPath，一个用于实际路径，一个用于显示名称
            actual_model_path = BaseModelPath(
                name=self.engine_args.model,
                model_path=self.engine_args.model
            )
            
            display_model_path = BaseModelPath(
                name=self.engine_args.served_model_name,  # 使用指定的显示名称
                model_path=self.engine_args.model  # 实际的模型路径
            )
        
            self.openai_serving_chat = OpenAIServingChat(
                self.engine,
                model_config,
                base_model_paths=[actual_model_path, display_model_path],
                response_role=self.response_role,
                lora_modules=self.lora_modules,
                chat_template=self.chat_template,
                prompt_adapters=None,
                request_logger=None,
            )
            
        logger.info(f"Request: {request}")
        generator = await self.openai_serving_chat.create_chat_completion(
            request, raw_request
        )
        if isinstance(generator, ErrorResponse):
            return JSONResponse(
                content=generator.model_dump(), status_code=generator.code
            )
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())



def build_app(cli_args: Dict[str, str]) -> serve.Application:
    """构建 Serve 应用"""
    # 直接将参数传递给 VLLMDeployment
    if "accelerator" in cli_args.keys():
        accelerator = cli_args.pop("accelerator")
    else:
        accelerator = "GPU"
    tp = int(cli_args["tensor-parallel-size"])
    logger.info(f"Tensor parallelism = {tp}")
    pg_resources = []
    for i in range(tp):
        pg_resources.append({"CPU": 1, accelerator: 1})  # for the vLLM actors
    return VLLMDeployment.options(
            placement_group_bundles=pg_resources, placement_group_strategy="STRICT_PACK"
    ).bind(cli_args)


model = build_app(
    {"model": os.environ.get('MODEL_ID','/mnt/models'), 
     "served-model-name": os.environ.get('MODEL_NAME', "/mnt/models"),
     "tensor-parallel-size": os.environ.get('TENSOR_PARALLELISM',1), 
     "pipeline-parallel-size": os.environ.get('PIPELINE_PARALLELISM',1),
     "extra-params": os.environ.get('EXTRA_PARAMS', "{}"),
     "max-model-len": os.environ.get('MAX_MODEL_LEN', 4096),
    }
    )
