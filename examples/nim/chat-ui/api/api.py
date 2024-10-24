import os

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


@check_api_key
async def completions_turbo(message, api_key=None):
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
