import os

class Config:
    MILVUS_HOST = os.getenv('MILVUS_HOST', '127.0.0.1')
    MILVUS_PORT = os.getenv('MILVUS_PORT', '19530')
    MILVUS_COLLECTION_NAME = 'text_collection'
    LLAMA2_CHAT_URL = os.getenv('LLAMA2_CHAT_URL', 'http://10.233.50.150/v1/chat/completions')
    OPENCHAT_URL = os.getenv('OPENCHAT_URL', 'http://openchat.llm.sxwl.ai:30005/v1/chat/completions')
    ID_TEXT_DIR = os.getenv('ID_TEXT_DIR', '/data/id_text')