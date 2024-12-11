import os
import sqlite3
from typing import Tuple, List
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
import requests
from config import (
    IMAGE_DIR, DB_PATH, API_KEY, API_URL,
    SUPPORTED_FORMATS, MAX_WORKERS, PROMPT_TEMPLATE
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImagePreprocessor:
    def __init__(self, image_dir: str = IMAGE_DIR, 
                 db_path: str = DB_PATH, 
                 api_key: str = API_KEY,
                 prompt: str = PROMPT_TEMPLATE):
        """
        初始化图片预处理器
        
        Args:
            image_dir: 图片目录路径，默认使用配置中的路径
            db_path: SQLite数据库路径，默认使用配置中的路径
            api_key: 多模态大模型API密钥，默认使用配置中的密钥
            prompt: 提示词模板，默认使用配置中的模板
        """
        self.image_dir = Path(image_dir)
        self.db_path = db_path
        self.api_key = api_key
        self.prompt = prompt
        self._init_db()

    def _init_db(self):
        """初始化SQLite数据库"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_url TEXT NOT NULL,
                    airline TEXT,
                    aircraft_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
    def _call_multimodal_api(self, image_url: str) -> Tuple[str, str]:
        """调用多模态大模型API获取图片信息
        
        Args:
            image_url: 图片URL地址
            
        Returns:
            Tuple[str, str]: 返回(航空公司, 飞机型号)元组
        """
        try:
            headers = {
                'accept': 'application/json',
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}'
            }
            
            payload = {
                "model": "/mnt/models",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self.prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url
                                }
                            }
                        ]
                    }
                ]
            }
            
            response = requests.post(
                API_URL,
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                # 解析航空公司和飞机型号
                import re
                airline = re.search(r'Company:\s*([^\n]+)', content)
                aircraft = re.search(r'Aircraft:\s*([^\n]+)', content)
                
                # 如果解析失败，记录原始响应内容
                if not airline or not aircraft:
                    logger.warning(f"无法从API响应中提取信息，原始响应内容: \n{content}")
                
                return (
                    airline.group(1) if airline else None,
                    aircraft.group(1) if aircraft else None
                )
            else:
                logger.error(f"API调用失败，状态码: {response.status_code}, 错误信息: {response.text}, image_url: {image_url}")
                return None, None
            
        except Exception as e:
            logger.error(f"处理图片 {image_url} 时发生错误: {str(e)}")
            return None, None

    def _process_single_image(self, image_url: str):
        """处理单张图片
        
        Args:
            image_url: 图片的URL地址
        """
        try:
            airline, aircraft_type = self._call_multimodal_api(image_url)
            if airline and aircraft_type:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        'INSERT INTO images (image_url, airline, aircraft_type) VALUES (?, ?, ?)',
                        (image_url, airline, aircraft_type)  # 将image_path改为存储image_url
                    )
                logger.info(f"成功处理图片: {image_url}")
            else:
                logger.warning(f"无法识别图片信息: {image_url}")
        except Exception as e:
            logger.error(f"处理图片 {image_url} 时发生错误: {str(e)}")

    def process_images(self, max_workers: int = MAX_WORKERS):
        """处理图片目录下的所有图片（包括子目录）
        
        Args:
            max_workers: 最大线程数
        """
        logger.info(f"开始扫描图片目录: {self.image_dir}")
        
        # 获取所有支持格式的图片文件（包括子目录）
        image_files = []
        for format in SUPPORTED_FORMATS:
            # 移除format中的点号，因为glob不需要
            format = format.lstrip('.')
            # **/ 确保递归搜索所有子目录
            pattern = f"**/*.{format}"
            image_files.extend(list(self.image_dir.glob(pattern)))
        
        if not image_files:
            logger.warning(f"在目录 {self.image_dir} 及其子目录中未找到支持的图片文件")
            return
        
        # 将本地路径转换为OSS URL
        image_urls = []
        from config import OSS_PREFIX
        
        for image_path in image_files:
            try:
                # 获取相对于IMAGE_DIR的路径
                relative_path = image_path.relative_to(self.image_dir)
                # 将路径分隔符统一转换为URL格式的���斜杠
                url_path = str(relative_path).replace('\\', '/')
                from urllib.parse import quote
                # 对路径进行URL编码，保留正斜杠
                encoded_path = quote(url_path)
                # 拼接完整的OSS URL
                oss_url = f"{OSS_PREFIX}/{encoded_path}"
                image_urls.append(oss_url)
                logger.debug(f"转换本地路径 {image_path} 为OSS URL: {oss_url}")
            except Exception as e:
                logger.error(f"处理图片路径时出错 {image_path}: {str(e)}")
                continue
        
        logger.info(f"找到 {len(image_urls)} 个图片文件")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(self._process_single_image, image_urls)

def main():
    # 使用配置中的默认值创建预处理器
    preprocessor = ImagePreprocessor()
    preprocessor.process_images()

if __name__ == "__main__":
    main() 