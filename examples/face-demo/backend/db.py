import os
import sqlite3
from contextlib import contextmanager
import logging
from typing import List, Tuple, Optional, Dict, Any
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import numpy as np
from config import (
    DB_PATH, MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME,
    FACE_VECTOR_DIM, INDEX_PARAMS, FACE_SIMILARITY_THRESHOLD,
    SEARCH_PARAMS, FACE_SEARCH_LIMIT
)

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self._init_sqlite()
        self._init_milvus()
    
    def _init_sqlite(self):
        """初始化 SQLite 数据库"""
        with self.get_sqlite_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY,
                    original_filename TEXT,
                    oss_url TEXT,
                    sha1 TEXT,
                    size BIGINT
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS faces (
                    id INTEGER PRIMARY KEY,
                    image_id INTEGER,
                    face_index INTEGER,
                    x1 FLOAT,
                    y1 FLOAT,
                    x2 FLOAT,
                    y2 FLOAT
                )
            ''')
            conn.commit()

    def _init_milvus(self):
        """初始化 Milvus 数据库"""
        try:
            connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
            
            photo_id = FieldSchema(
                name="photo_id",
                dtype=DataType.INT64,
                is_primary=False,  # 作为辅助字段
                description="The ID of the photo containing the face"
            )

            face_index = FieldSchema(
                name="face_index",
                dtype=DataType.INT64,
                is_primary=False,  # 作为辅助字段
                description="The index of the face in the photo"
            )

            face_vector = FieldSchema(
                name="face_vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=FACE_VECTOR_DIM,  # 替换为你的向量维度
                description="The vector representation of the face"
            )

            unique_id = FieldSchema(
                name="unique_id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True,  # 自动生成 ID
                description="Unique identifier for each face record"
            )

            # 定义 schema
            schema = CollectionSchema(
                fields=[unique_id, photo_id, face_index, face_vector],
                description="A collection of face vectors for face recognition"
            )

            # 检查集合是否存在
            if utility.has_collection(COLLECTION_NAME):
                logger.info(f"使用已存在的 Milvus 集合: {COLLECTION_NAME}")
                self.collection = Collection(COLLECTION_NAME)
            else:
                logger.info(f"创建新的 Milvus 集合: {COLLECTION_NAME}")
                self.collection = Collection(
                    name=COLLECTION_NAME, 
                    schema=schema
                )

            # 检查索引是否存在
            if not self.collection.has_index():
                logger.info("创建向量索引...")
                self.collection.create_index(
                    field_name="face_vector",
                    index_params=INDEX_PARAMS
                )
                logger.info("向量索引创建完成")

            # 加载集合
            self.collection.load()
            
        except Exception as e:
            logger.error(f"Milvus 初始化失败: {str(e)}")
            raise

    @contextmanager
    def get_sqlite_connection(self):
        """SQLite 连接上下文管理器"""
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        try:
            yield conn
        finally:
            conn.close()

    # SQLite 操作方法
    def insert_image(self, filename: str, oss_url: str, sha1: str, size: int) -> int:
        """插入图片记录"""
        with self.get_sqlite_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO images (original_filename, oss_url, sha1, size) VALUES (?, ?, ?, ?)",
                (filename, oss_url, sha1, size)
            )
            conn.commit()
            return cursor.lastrowid  # 返回新插入记录的 id 值

    def insert_face(self, image_id: int, face_index: int, x1: float, y1: float, x2: float, y2: float) -> int:
        """插入人脸记录"""
        with self.get_sqlite_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO faces (image_id, face_index, x1, y1, x2, y2) VALUES (?, ?, ?, ?, ?, ?)",
                (image_id, face_index, x1, y1, x2, y2)
            )
            conn.commit()
            return cursor.lastrowid  # 返回新插入记录的 id 值

    def get_image_by_sha1(self, sha1: str) -> Optional[Dict[str, Any]]:
        """通过 SHA1 获取图片信息"""
        with self.get_sqlite_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, original_filename, oss_url, sha1, size FROM images WHERE sha1=?",
                (sha1,)
            )
            row = cursor.fetchone()
            if row:
                return {
                    "id": row[0],
                    "original_filename": row[1],
                    "oss_url": row[2],
                    "sha1": row[3],
                    "size": row[4]
                }
            return None

    def get_images_paginated(self, page: int, page_size: int) -> Tuple[List[Dict[str, Any]], int]:
        """获取分页的图片列表"""
        with self.get_sqlite_connection() as conn:
            cursor = conn.cursor()
            
            # 获取总数
            cursor.execute("SELECT COUNT(*) FROM images")
            total = cursor.fetchone()[0]
            
            # 获取分页数据
            offset = (page - 1) * page_size
            cursor.execute("""
                SELECT id, original_filename, oss_url, sha1, size 
                FROM images 
                ORDER BY id DESC 
                LIMIT ? OFFSET ?
            """, (page_size, offset))
            
            rows = cursor.fetchall()
            images = [{
                "id": row[0],
                "original_filename": row[1],
                "oss_url": row[2],
                "sha1": row[3],
                "size": row[4]
            } for row in rows]
            
            return images, total

    def delete_image(self, image_id: int) -> Optional[str]:
        """删除图片记录，返回 oss_url"""
        with self.get_sqlite_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT oss_url FROM images WHERE id=?", (image_id,))
            result = cursor.fetchone()
            
            if result:
                cursor.execute("DELETE FROM images WHERE id=?", (image_id,))
                conn.commit()
                return result[0]
            return None

    # Milvus 操作方法
    def insert_face_vectors(self, photo_id: int, embeddings: np.ndarray) -> bool:
        """插入人脸向量到 Milvus"""
        try:
            # 准备实体数据列表
            entities = []
            for i in range(len(embeddings)):
                entity = {
                    "photo_id": photo_id,
                    "face_index": i,
                    "face_vector": embeddings[i].tolist()
                }
                entities.append(entity)
            
            # 插入数据
            self.collection.insert(entities)
            # 确保数据被写入
            self.collection.flush()
            logger.info(f"成功插入{len(embeddings)}个人脸向量")
            return True
        except Exception as e:
            logger.error(f"插入人脸向量失败: {str(e)}")
            return False

    def search_similar_faces(self, face_embedding):
        """在 Milvus 中搜索相似人脸"""
        try:
            # 确保集合已加载
            self.collection.load()
            
            # 确保索引存在
            if not self.collection.has_index():
                raise Exception("向量索引未创建")
            
            # 确保 face_embedding 是列表格式
            search_vector = face_embedding if isinstance(face_embedding, list) else face_embedding.tolist()
            
            results = self.collection.search(
                data=[search_vector],
                anns_field="face_vector",
                param=SEARCH_PARAMS,
                limit=FACE_SEARCH_LIMIT,
                expr=None,
                output_fields=["photo_id", "face_index"]
            )
            
            similar_faces = []
            for j, result in enumerate(results[0]):
                distance = result.distance
                photo_id = result.entity.get('photo_id')
                face_index = result.entity.get('face_index')
                similarity = 1 / (1 + distance)  # 将距离转换为相似度分数
                if similarity >= FACE_SIMILARITY_THRESHOLD:
                    similar_faces.append({
                        "id": photo_id,
                        "similarity": similarity,
                        "face_index": face_index
                    })
            
            return similar_faces
        except Exception as e:
            logger.error(f"搜索失败: {str(e)}")
            raise

    def delete_face_vectors(self, photo_id: int) -> bool:
        """删除指定图片的所有脸向量"""
        try:
            # 先加载集合
            self.collection.load()
            
            expr = f"photo_id == {photo_id}"
            self.collection.delete(expr)
            # 确保删除操作被持久化
            self.collection.flush()
            # # 执行压缩操作以回收空间
            # self.collection.compact()
            
            logger.info(f"成功删除图片ID {photo_id} 的人脸向量")
            return True
        except Exception as e:
            logger.error(f"删除人脸向量失败: {str(e)}")
            return False

    def get_images_by_ids(self, image_ids):
        """
        根据图片ID列表获取图片信息
        """
        if not image_ids:
            return []
        
        placeholders = ','.join('?' * len(image_ids))
        query = f"""
            SELECT id, original_filename, oss_url, size, sha1
            FROM images 
            WHERE id IN ({placeholders})
        """
        
        with self.get_sqlite_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, image_ids)
            results = cursor.fetchall()
            
        return [dict(row) for row in results]

    def get_face_coords(self, image_id: int, face_index: int) -> Optional[Dict[str, float]]:
        """
        获取指定图片中特定人脸的坐标信息
        
        Args:
            image_id (int): 图片ID
            face_index (int): 人脸索引
            
        Returns:
            Optional[Dict[str, float]]: 包含人脸坐标的字典，如果未找到则返回 None
        """
        with self.get_sqlite_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT x1, y1, x2, y2 
                FROM faces 
                WHERE image_id = ? AND face_index = ?
                """,
                (image_id, face_index)
            )
            row = cursor.fetchone()
            
            if row:
                return {
                    "x1": row[0],
                    "y1": row[1],
                    "x2": row[2],
                    "y2": row[3]
                }
            return None

# 创建全局数据库管理器实例
db_manager = DatabaseManager()
