"""
向量存储服务 —— 封装 Chroma 向量检索。

基于 vector_stores.py，提供单例模式供全局使用。
"""
import threading
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import config_data as config


class VectorStoreService:
    """向量存储服务（单例）"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, embedding=None):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._initialized = True

        self.embedding = embedding or OpenAIEmbeddings(
            openai_api_base=config.address,
            model=config.embedding_model_name,
            # 必须关闭长度检查：默认开启时 langchain 会先把文本转成 tiktoken
            # 的 token id 再发给 API——OpenAI 官方接口能还原，但第三方兼容
            # 接口（SiliconFlow/bge-m3）词表不同，收到的是语义错乱的 token，
            # 向量与真实语义脱钩（实测同文本与直连 API 余弦仅 0.29）
            check_embedding_ctx_length=False,
        )

        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )

    def get_retriever(self):
        """返回向量检索器，方便加入 LangChain chain"""
        return self.vector_store.as_retriever(
            search_kwargs={"k": config.retrieval_top_k}
        )


# 全局单例
vector_store_service = VectorStoreService()
