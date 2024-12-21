import os
import torch

# 基础配置
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# 数据库配置
DB_PATH = os.path.join(BASE_DIR, "face_demo.db")

# 缓存目录
CACHE_DIR = os.path.join(BASE_DIR, "cache")

# 阿里云OSS配置
OSS_ACCESS_KEY_ID = os.getenv('OSS_ACCESS_KEY_ID')
OSS_ACCESS_KEY_SECRET = os.getenv('OSS_ACCESS_KEY_SECRET')
OSS_BUCKET_NAME = 'sxwl-ai'
OSS_ENDPOINT = 'oss-cn-beijing.aliyuncs.com'
OSS_ROOT_DIR = 'face-demo'

# Milvus配置
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "face_vectors"
INDEX_PARAMS = {
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 1024}
}
SEARCH_PARAMS = {
    "metric_type": "L2",
    "params": {"nprobe": 128}
}
FACE_SIMILARITY_THRESHOLD = 0.65 # 人脸相似度阈值
FACE_SEARCH_LIMIT = 1000 # 人脸搜索返回的最大结果数

# 模型配置
FACENET_WEIGHTS = "./data/models/20180402-114759-vggface2.pt"  # FaceNet模型权重路径
FACE_VECTOR_DIM = 512  # FaceNet输出维度

# 自动检测并设置设备
def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'  # M1 Mac 的 Metal 加速
    return 'cpu'

DEVICE = get_device()

# 支持的图片格式
SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
