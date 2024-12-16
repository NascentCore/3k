from fastapi import FastAPI, Query, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import List, Optional
from search import ImageSearcher
import requests
import re
from config import (
    API_KEY, 
    API_URL, 
    EXTRACT_KEYWORDS_PROMPT,
    PROMPT_TEMPLATE,
    UPLOAD_URL,
    MODEL_BASE_URL,
    MODEL_TEMPERATURE,
    MODEL_TOP_P,
    MODEL_MAX_TOKENS,
)

app = FastAPI(title="飞机图片检索服务")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件目录（用于提供图片访问）
app.mount("/images", StaticFiles(directory=Path("data/images")), name="images")

# 创建搜索器实例
searcher = ImageSearcher()

@app.get("/api/search")
async def search_images(
    keyword: str = Query(None, description="搜索关键词"),
    use_optimized_model: bool = Query(False, description="是否使用优化模型"),
    image_url: str = Query(None, description="图片URL")
):
    """
    搜索图片接口，支持关键词搜索
    """
    if image_url:
        # 使用大模型解析图片, 使用类似_call_multimodal_api的方法从多模态模型中提取图片中的航司和机型信息
        extracted_info = _call_multimodal_api(image_url)
        if extracted_info:
            airline, aircraft_type = extracted_info
            results = searcher.search_by_text(airline, aircraft_type)
            return {"status": "success", "data": results}
    
    if not keyword:
        return {"status": "success", "data": []}
        
    try:
        if use_optimized_model:
            # 使用大模型解析关键词
            extracted_info = _extract_search_info(keyword)
            if extracted_info:
                airline, aircraft_type = extracted_info
                results = searcher.search_by_text(airline, aircraft_type)
                return {"status": "success", "data": results}
        
        # 如果未启用大模型，或大模型解析失败，尝试按空格分割关键词
        keywords = keyword.strip().split()
        if len(keywords) > 1:
            # 如果有多个关键词，假设第一个是航司，第二个是机型
            results = searcher.search_by_text(keywords[0], keywords[1])
        else:
            # 单个关键词时使用原有的关键搜索
            results = searcher.search_by_keyword(keyword)
        return {"status": "success", "data": results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _call_multimodal_api(image_url: str) -> Optional[tuple[str, str]]:
    """
    调用多模态API解析图片信息
    
    Args:
        image_url: 图片URL地址
        
    Returns:
        Optional[tuple[str, str]]: 返回(航空公司, 飞机型号)元组，如果解析失败返回None
    """
    try:
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {API_KEY}'
        }
        
        payload = {
            "model": "/mnt/models",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": PROMPT_TEMPLATE
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        }
                    ]
                }
            ],
            "temperature": MODEL_TEMPERATURE,
            "top_p": MODEL_TOP_P,
            "max_tokens": MODEL_MAX_TOKENS,
        }
        
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            content = response.json().get('choices', [{}])[0].get('message', {}).get('content', '')
            
            # 解析航空公司和飞机型号
            airline = re.search(r'Company:\s*([^\n]+)', content)
            aircraft = re.search(r'Aircraft:\s*([^\n]+)', content)
            
            airline = airline.group(1) if airline else None
            aircraft = aircraft.group(1) if aircraft else None
            
            # 如果提取的值是"None"，转换为None
            airline = None if airline == "None" else airline
            aircraft = None if aircraft == "None" else aircraft
            
            if airline or aircraft:
                return airline, aircraft
        else:
            print(f"多模态API返回 状态码: {response.status_code} 错误: {response.json()}")
                
        return None
        
    except Exception as e:
        print(f"处理图片 {image_url} 时发生错误: {str(e)}")
        return None

def _extract_search_info(keyword: str) -> Optional[tuple[str, str]]:
    """
    使用大模型API解析搜索关键词
    """
    try:
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {API_KEY}'
        }
        
        payload = {
            "model": "/mnt/models",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{EXTRACT_KEYWORDS_PROMPT}\nQuery：{keyword}"
                        }
                    ]
                }
            ],
            "temperature": MODEL_TEMPERATURE,
            "top_p": MODEL_TOP_P,
            "max_tokens": MODEL_MAX_TOKENS,
        }
        
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            content = response.json().get('choices', [{}])[0].get('message', {}).get('content', '')
            
            airline = re.search(r'Company:\s*([^\n]+)', content)
            aircraft = re.search(r'Aircraft:\s*([^\n]+)', content)
            
            airline = airline.group(1) if airline else None
            aircraft = aircraft.group(1) if aircraft else None
            
            # 如果提取的值是"None"，转换为None
            airline = None if airline == "None" else airline
            aircraft = None if aircraft == "None" else aircraft
            
            if airline or aircraft:
                return airline, aircraft
                
        return None
        
    except Exception as e:
        print(f"解析关键词时出错: {str(e)}")
        return None

@app.get("/api/airlines")
async def get_airlines():
    """获取所有航空公司列表"""
    try:
        airlines = searcher.get_all_airlines()
        return {"status": "success", "data": airlines}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/aircraft-types")
async def get_aircraft_types():
    """获取所有飞机型号列表"""
    try:
        types = searcher.get_all_aircraft_types()
        return {"status": "success", "data": types}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_image(image: UploadFile = File(...)):
    """处理图片上传的接口"""
    try:
        # 读取上传的文件内容
        content = await image.read()
        
        # 打印调试信息
        print(f"开始上传文件: {image.filename}, 内容类型: {image.content_type}")
        
        # 构建multipart/form-data请求
        files = {
            'image': (image.filename, content, image.content_type)
        }
        
        # 发送请求到上传服务器，添加错误处理和超时设置
        try:
            response = requests.post(
                UPLOAD_URL,
                files=files,
                verify=False,
                timeout=30
            )
            
            # 检查响应状态
            response.raise_for_status()
            
            # 尝试解析JSON响应
            result = response.json()
            print(f"上传成功，服务器响应: {result}")
            
            # 统一返回格式
            return {
                "url": f"{MODEL_BASE_URL}{result.get('file_path')}"  # 添加完整的访问URL
            }
            
        except requests.exceptions.RequestException as e:
            print(f"上传请求失败: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": f"上传服务器错误: {str(e)}"
                }
            )
            
    except Exception as e:
        error_msg = f"处理上传请求时发生错误: {str(e)}"
        print(error_msg)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": error_msg
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 