import os
import json
import asyncio
from typing import List, Dict, Any
import httpx

API_KEY = os.environ.get('API_KEY')
if proxy := os.environ.get('HTTPS_PROXY'):
    PROXIES = {"https://": proxy}
else:
    PROXIES = None


def check_api_key(func):
    def wrapper(*args, **kwargs):
        api_key = kwargs.get('api_key', API_KEY) or API_KEY
        # if not api_key:
        #     raise ValueError('API key is required')
        # if not api_key.startswith('sk-'):
        #     raise ValueError('API key must start with "sk-"')
        kwargs['api_key'] = api_key
        return func(*args, **kwargs)

    return wrapper


@check_api_key
async def completions(message, api_key=None):
    """Get completions for the message."""
    # print('message:', message)
    url = os.environ.get('API_URL')
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    async with httpx.AsyncClient(proxies=PROXIES) as client:
        response = await client.post(
            url,
            json=message.dict(),
            headers=headers,
            timeout=60,
        )
        # print('response:', response.json())
        return response.json()


async def completions_turbo(message, api_key=None):
    """逐步读取并发送响应的每个流式内容块"""
    url = os.environ.get('API_URL')
    print(url)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    message_data = message.dict(exclude_none=True)
    message_data['stream'] = True
    clean_previous_image_urls(message_data["messages"])
    print(message_data)

    async with httpx.AsyncClient(proxies=PROXIES) as client:
        async with client.stream("POST", url, json=message_data, headers=headers) as response:
            async for chunk in response.aiter_text():
                if chunk.strip():
                    if not chunk.strip().startswith("data:"):
                        yield f"data: {chunk.strip()}\n\n"
                    else:
                        yield f"{chunk.strip()}\n\n"


@check_api_key
async def credit_summary(api_key=None):
    """Get the credit summary for the API key."""
    url = os.environ.get('API_URL')
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    async with httpx.AsyncClient(proxies=PROXIES) as client:
        response = await client.get(
            url,
            headers=headers,
            timeout=60,
        )
        return response.json()

def clean_previous_image_urls(messages: List[Dict[str, Any]]):
    last_image_url_index = None

    # 找到最后一个包含 image_url 的 content 的索引
    for index, message in enumerate(messages):
        if isinstance(message.get("content"), list):
            for item in message["content"]:
                if item.get("type") == "image_url":
                    last_image_url_index = index

    # 如果找到了最后一个 image_url 的位置
    if last_image_url_index is not None:
        # 遍历所有 message，删除除了最后一个的 image_url 项
        for index, message in enumerate(messages):
            if isinstance(message.get("content"), list):
                # 只保留最后一个 image_url 项
                if index != last_image_url_index:
                    message["content"] = [
                        item for item in message["content"] if item.get("type") != "image_url"
                    ]