import hashlib
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, List
import os
from face_processor import FaceProcessor
from oss2 import Auth, Bucket
from config import OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET, OSS_BUCKET_NAME, OSS_ENDPOINT, CACHE_DIR, OSS_ROOT_DIR
from pydantic import BaseModel
import logging
from db import db_manager
import mimetypes

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI()

# 配置 FastAPI 的日志级别
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 确保缓存目录存在
os.makedirs(CACHE_DIR, exist_ok=True)

# 初始化OSS
auth = Auth(OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET)
bucket = Bucket(auth, OSS_ENDPOINT, OSS_BUCKET_NAME)

# 初始化FaceProcessor
face_processor = FaceProcessor()

class ImageResponse(BaseModel):
    id: int
    original_filename: str
    oss_url: str
    size: int
    sha1: str

class PaginationParams(BaseModel):
    page: Optional[int] = 1
    page_size: Optional[int] = 10

@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    filename = file.filename
    original_name = os.path.splitext(filename)[0]
    file_ext = os.path.splitext(filename)[1]

    # 使用短 uuid (前8位) 作为唯一标识
    unique_id = str(uuid.uuid4())[:8]
    unique_filename = f"{original_name}_{unique_id}{file_ext}"

    file_size = file.size

    # 计算文件的 sha1
    file_content = file.file.read()
    sha1 = hashlib.sha1(file_content).hexdigest()
    
    # 保存文件临时位置
    temp_path = os.path.join(CACHE_DIR, unique_filename)
    with open(temp_path, "wb") as buffer:
        buffer.write(file_content)
    
    # 检查文件是否已存在
    existing_image = db_manager.get_image_by_sha1(sha1)
    if existing_image:
        return JSONResponse(
            status_code=200,
            content={
                "file_id": original_name,
                "oss_url": existing_image["oss_url"],
                "message": "文件已存在",
                "duplicate": True,
                "status": "warning"
            }
        )
    
    # 上传到OSS
    oss_url = f'https://{OSS_BUCKET_NAME}.{OSS_ENDPOINT}/{OSS_ROOT_DIR}/{unique_filename}'
    with open(temp_path, 'rb') as f:
        bucket.put_object(os.path.join(OSS_ROOT_DIR, unique_filename), f)
    
    # 保存到数据库
    id = db_manager.insert_image(filename, oss_url, sha1, file_size)
    
    # 用FaceProcessor处理图片
    process_result = face_processor.extract_face(temp_path)
    os.remove(temp_path)

    if process_result[0]:
        # 保存到milvus
        db_manager.insert_face_vectors(id, process_result[1]['embeddings'])
        # 保存人脸坐标
        for i, box in enumerate(process_result[1]['boxes']):
            db_manager.insert_face(id, i, box[0], box[1], box[2], box[3])
        return JSONResponse(
            status_code=200,
            content={
                "file_id": original_name,
                "oss_url": oss_url,
                "message": "图片处理成功",
                "status": "success"
            }
        )
    else:
        return JSONResponse(
            status_code=500,
            content={
                "message": process_result[1],
                "status": "error"
            }
        )

@app.post("/api/search")
async def search(file: UploadFile = File(...)):
    logging.info(f"接收到搜索请求，文件名：{file.filename}")
    logging.info(f"文件内容类型：{file.content_type}")
    
    # 验证文件是否存在
    if not file:
        return JSONResponse(
            status_code=400,
            content={
                "message": "请上传文件",
                "status": "error"
            }
        )
    
    # 获取文件扩展名
    file_ext = os.path.splitext(file.filename)[1]
    
    # 保存上传的文件到临时目录
    temp_filename = f"search_{uuid.uuid4()}{file_ext}"
    temp_path = os.path.join(CACHE_DIR, temp_filename)
    
    try:
        # 重置文件指针位置
        await file.seek(0)
        
        # 保存文件
        contents = await file.read()
        with open(temp_path, "wb") as buffer:
            buffer.write(contents)
        
        # 使用 FaceProcessor 提取人脸特征
        process_result = face_processor.extract_face(temp_path)
        
        if not process_result[0]:
            return JSONResponse(
                status_code=400,
                content={
                    "message": "无法从图片中提取人脸特征",
                    "status": "error"
                }
            )
        
        # 获取人脸特征向量
        face_embedding = process_result[1]['embeddings']
        if len(face_embedding) == 0:
            return JSONResponse(
                status_code=200,
                content={
                    "data": [],
                    "message": "未找到相似人脸",
                    "status": "success"
                }
            )
        elif len(face_embedding) > 1:
            return JSONResponse(
                status_code=400,
                content={
                    "message": "图片中有多张人脸",
                    "status": "error"
                }
            )
            
        # 使用 milvus 搜索相似人脸
        similar_faces = db_manager.search_similar_faces(face_embedding[0].tolist())
            
        if not similar_faces:
            return JSONResponse(
                status_code=200,
                content={
                    "data": [],
                    "message": "未找到相似人脸",
                    "status": "success"
                }
            )
        
        # 获取相似图片的详细信息
        image_ids = list(set(face['id'] for face in similar_faces))  # 去重image_ids
        similar_images = db_manager.get_images_by_ids(image_ids)
        # 获取人脸坐标
        for face in similar_faces:
            face['face_coords'] = db_manager.get_face_coords(face['id'], face['face_index'])
        
        # 重组数据，将相同图片的多个人脸信息合并
        result_images = []
        for img in similar_images:
            # 找出所有属于这张图片的人脸
            matching_faces = [
                {
                    'similarity': face['similarity'],
                    'face_index': face['face_index'],
                    'face_coords': face['face_coords']
                }
                for face in similar_faces 
                if face['id'] == img['id']
            ]
            
            # 将人脸信息添加到图片数据中
            img_data = img.copy()
            img_data['faces'] = matching_faces
            result_images.append(img_data)
        
        return JSONResponse(
            status_code=200,
            content={
                "data": result_images,
                "message": "搜索完成",
                "status": "success"
            }
        )
            
    except Exception as e:
        logging.error(f"搜索失败: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "message": f"搜索失败: {str(e)}",
                "status": "error"
            }
        )
    finally:
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/api/images")
async def get_images(
    page: Optional[int] = 1,
    page_size: Optional[int] = 10
):
    images, total = db_manager.get_images_paginated(page, page_size)
    return {
        "data": images,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size
    }

@app.delete("/api/images/{id}")
async def delete_image(id: int = Path(...)):
    try:
        oss_url = db_manager.delete_image(id)
        if not oss_url:
            return JSONResponse(
                status_code=404,
                content={
                    "message": "图片不存在",
                    "status": "error"
                }
            )
            
        # 从OSS删除文件
        object_name = oss_url.split(f"{OSS_ENDPOINT}/")[-1]
        
        try:
            # 从OSS删除文件
            bucket.delete_object(object_name)
        except Exception as e:
            logging.error(f"删除OSS文件失败: {str(e)}")
            
        # 删除 Milvus 中的向量
        db_manager.delete_face_vectors(id)
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "图片删除成功",
                "status": "success"
            }
        )
    except Exception as e:
        logging.error(f"删除图片失败: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "message": f"删除图片失败: {str(e)}",
                "status": "error"
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# 在应用启动时添加
logging.info("应用启动成功，日志系统正常工作")
