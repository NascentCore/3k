import os
from pathlib import Path

# 基础路径配置
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# 图片相关配置
IMAGE_DIR = DATA_DIR / "images"  # 替换为实际的图片目录
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.JPG'}

# OSS配置
OSS_PREFIX = "https://sxwl-ai.oss-cn-beijing.aliyuncs.com/airplane-demo/images"

# 数据库配置
DB_PATH = DATA_DIR / "images.db"

# 模型配置
MODEL_BASE_URL = "http://master.llm.sxwl.ai:30005/inference/infer-b277210a-a7e8-4f95-a24b-51acda83dfb8/"
API_BASE_URL = "http://master.llm.sxwl.ai:30005/inference/api/infer-b277210a-a7e8-4f95-a24b-51acda83dfb8/"

# API配置
API_KEY = os.getenv("MULTIMODAL_API_KEY", "your_api_key")  # 优先使用环境变量
API_URL = API_BASE_URL + "v1/chat/completions"  # 替换为实际的API地址
UPLOAD_URL = MODEL_BASE_URL + "upload"

# 处理配置
MAX_WORKERS = 4  # 最大线程数 

# 搜索配置
USE_LLM_SEARCH = True  # 是否使用大模型处理搜索词

# 航司和机型列表
PROBABLE_AIRLINE_AIRCRAFT = '''
Possible airline companies include:
- ANA - All Nippon Airways
- Lufthansa
- Star Alliance
- IA - Air China
- YTO - Yunda Express
- TJU - Tianjiao Airlines
- CEA - China Eastern Airlines
- CSN - China Southern Airlines
- CTU - Chengdu Airlines
- UNI - Uni Air

Possible aircraft models include:
- Airbus 340
- Boeing 777
- Boeing 787
- 中国商飞 ARJ21
- 中国商飞 C919
- 武直-10
'''

# 多模态模型提示词
PROMPT_TEMPLATE = '''
Based on the content of the picture, identify the airline company and the aircraft model.
If the airline company or aircraft model cannot be identified, please fill in None.
Always return the results in the following exact format:

```
Company: Name of the airline company or None
Aircraft: Aircraft model or None
```

Do not add any extra content to the result.
''' + PROBABLE_AIRLINE_AIRCRAFT

# 添加提取关键词的提示模板
EXTRACT_KEYWORDS_PROMPT = '''
Please extract the airline company name and aircraft type from the following query.
If found, return in the following format:

Company: [Airline Company Name]
Aircraft: [Aircraft Type]

If any information is not present, fill in None for the corresponding value.

Change the airline company name and aircraft type to the format of the following list:
''' + PROBABLE_AIRLINE_AIRCRAFT

