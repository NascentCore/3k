import sqlite3
from pathlib import Path
from typing import List, Dict
from config import DB_PATH
import jieba

class ImageSearcher:
    def __init__(self, db_path: str = DB_PATH):
        """
        初始化图片搜索器
        
        Args:
            db_path: SQLite数据库路径
        """
        self.db_path = db_path
        # 加载航空公司和机型的词典
        self._load_custom_dict()

    def _load_custom_dict(self):
        """加载自定义词典"""
        with sqlite3.connect(self.db_path) as conn:
            # 加载所有航空公司名称
            airlines = self.get_all_airlines()
            for airline in airlines:
                if airline:
                    jieba.add_word(airline)
            
            # 加载所有飞机型号
            aircraft_types = self.get_all_aircraft_types()
            for aircraft in aircraft_types:
                if aircraft:
                    jieba.add_word(aircraft)

    def search_by_text(self, airline: str = None, aircraft_type: str = None) -> List[Dict]:
        """
        根据文本条件搜索图片
        
        Args:
            airline: 航空公司名称
            aircraft_type: 飞机型号
            
        Returns:
            List[Dict]: 搜索结果列表
        """
        query = "SELECT * FROM images WHERE 1=1"
        params = []
        
        if airline:
            query += " AND airline LIKE ?"
            params.append(f"%{airline}%")
            
        if aircraft_type:
            query += " AND aircraft_type LIKE ?"
            params.append(f"%{aircraft_type}%")
            
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            results = cursor.fetchall()
            
        return [dict(row) for row in results]

    def get_all_airlines(self) -> List[str]:
        """获取所有航空公司列表"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT DISTINCT airline FROM images WHERE airline IS NOT NULL")
            return [row[0] for row in cursor.fetchall()]

    def get_all_aircraft_types(self) -> List[str]:
        """获取所有飞机型号列表"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT DISTINCT aircraft_type FROM images WHERE aircraft_type IS NOT NULL")
            return [row[0] for row in cursor.fetchall()]

    def search_by_keyword(self, keyword: str) -> List[dict]:
        """
        使用分词后的关键词进行模糊搜索
        
        Args:
            keyword: 搜索关键词
            
        Returns:
            List[dict]: 搜索结果列表
        """
        # 对关键词进行分词
        words = jieba.cut(keyword)
        words = [w.strip() for w in words if w.strip()]  # 去除空白字符
        
        if not words:
            return []
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # 构建查询条件
            conditions = []
            params = []
            for word in words:
                conditions.extend([
                    "airline LIKE ?",
                    "aircraft_type LIKE ?"
                ])
                params.extend([f"%{word}%", f"%{word}%"])
            
            # 使用OR连接所有条件
            query = f"""
            SELECT DISTINCT image_url, airline, aircraft_type 
            FROM images 
            WHERE {" OR ".join(conditions)}
            """
            
            cursor.execute(query, params)
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "image_url": row["image_url"],
                    "airline": row["airline"],
                    "aircraft_type": row["aircraft_type"]
                })
            
            return results