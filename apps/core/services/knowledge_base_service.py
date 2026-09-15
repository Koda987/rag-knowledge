"""
知识库服务 —— 封装文档上传、分块、向量化。

单例模式；上传记录同步到 Django 数据库（KnowledgeDocument）。
返回结构化结果（dict），路由层据此判断状态，不解析字符串。
"""
import hashlib
import logging
import os
import threading
from datetime import datetime
from langchain_chroma import Chroma
import config_data as config
from . import chunking

logger = logging.getLogger(__name__)


def _get_string_md5(input_str: str, encoding: str = 'utf-8') -> str:
    """将传入的字符串转换为 md5 字符串"""
    str_bytes = input_str.encode(encoding=encoding)
    md5_obj = hashlib.md5()
    md5_obj.update(str_bytes)
    return md5_obj.hexdigest()


def _check_md5(md5_str: str) -> bool:
    """检查 md5 是否已处理过"""
    if not os.path.exists(config.md5_path):
        with open(config.md5_path, 'w', encoding='utf-8'):
            pass
        return False
    with open(config.md5_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == md5_str:
                return True
    return False


def _save_md5(md5_str: str) -> None:
    """将 md5 字符串记录到文件"""
    with open(config.md5_path, 'a', encoding='utf-8') as f:
        f.write(md5_str + '\n')


class KnowledgeBaseService:
    """知识库服务（单例）"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._initialized = True

        os.makedirs(config.persist_directory, exist_ok=True)

        # 复用 vector_store_service 的嵌入器（单一配置来源）：
        # 该处已关闭 check_embedding_ctx_length——此处曾自建默认配置的
        # 嵌入器，token-id 问题导致入库向量与检索端不在同一向量空间
        from .vector_store_service import vector_store_service as _vss
        self.chroma = Chroma(
            collection_name=config.collection_name,
            embedding_function=_vss.embedding,
            persist_directory=config.persist_directory,
        )

    def upload_by_str(self, data: str, filename: str,
                      chunk_size: int | None = None,
                      overlap: int | None = None,
                      mode: str | None = None,
                      clean: bool | None = None) -> dict:
        """
        将传入字符串进行向量化并存入向量数据库。

        Args:
            data: 文本内容
            filename: 来源文件名
            chunk_size / overlap / mode / clean: 分段参数（Dify 式向导传入）；
                None 落回 config_data 默认值

        Returns:
            结构化结果：
            {"status": "success" | "skipped" | "error",
             "chunks": 分块数, "message": 人类可读描述}
        """
        md5_hex = _get_string_md5(data)

        if _check_md5(md5_hex):
            self._save_to_db(filename, md5_hex, len(data), 0)
            return {
                "status": "skipped",
                "chunks": 0,
                "message": f"内容已存在于知识库中，跳过入库 (MD5: {md5_hex[:8]}...)",
            }

        # 文本分块（切块统一走 chunking 模块，支持上传时自定义参数）
        knowledge_chunks = chunking.split_text(data, chunk_size, overlap, mode, clean)

        metadata = {
            "source": filename,
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator": config.operator,
        }

        chunk_count = len(knowledge_chunks)
        self.chroma.add_texts(
            knowledge_chunks,
            metadatas=[metadata.copy() for _ in knowledge_chunks],
        )

        _save_md5(md5_hex)
        self._save_to_db(filename, md5_hex, len(data), chunk_count)

        return {
            "status": "success",
            "chunks": chunk_count,
            "message": f"内容已载入向量库，共 {chunk_count} 个分块",
        }

    def upload_by_file_content(self, content: bytes, filename: str,
                               chunk_size: int | None = None,
                               overlap: int | None = None,
                               mode: str | None = None,
                               clean: bool | None = None) -> dict:
        """
        直接从文件字节内容上传（FastAPI 文件上传接口调用）。

        utf-8 优先，失败回退 gbk。分段参数透传给 upload_by_str。
        """
        try:
            text = content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                text = content.decode('gbk')
            except UnicodeDecodeError:
                return {
                    "status": "error",
                    "chunks": 0,
                    "message": "文件编码不支持，请使用 UTF-8 编码的文本文件",
                }
        return self.upload_by_str(text, filename, chunk_size, overlap, mode, clean)

    def stats(self) -> dict:
        """
        知识库统计：向量块数（Chroma）与文档记录（Django）。

        Returns:
            {"chunks": 向量块数, "documents": 文档记录数或 None,
             "last_updated": 最近入库时间或 None}
        """
        # langchain-chroma 未封装 count()，借道底层 collection 的原生接口
        chunks = self.chroma._collection.count()
        documents = None
        last_updated = None
        try:
            from apps.core.models import KnowledgeDocument
            from django.utils import timezone
            documents = KnowledgeDocument.objects.count()
            latest = KnowledgeDocument.objects.order_by('-created_at').first()
            if latest:
                # created_at 存的是 UTC，展示前转本地时区
                last_updated = timezone.localtime(latest.created_at).strftime('%Y-%m-%d %H:%M')
        except Exception as e:
            logger.warning("读取 Django 上传记录失败: %s", e)
        return {"chunks": chunks, "documents": documents, "last_updated": last_updated}

    def documents(self) -> list[dict]:
        """
        已录入文档列表（来自 Django 上传记录，按入库时间倒序）。

        chunk_count 以 Chroma 实际分块数为准（去重跳过的重复上传
        会在 DB 里记 0 块，直接展示会误导）。

        Returns:
            [{filename, chunk_count, content_length, created_at}, ...]
        """
        try:
            from apps.core.models import KnowledgeDocument
            from django.utils import timezone
            # 统计每个来源文件在 Chroma 里的真实块数
            counts = {}
            try:
                got = self.chroma._collection.get(include=["metadatas"])
                for meta in got.get("metadatas") or []:
                    src = (meta or {}).get("source", "未知来源")
                    counts[src] = counts.get(src, 0) + 1
            except Exception as e:
                logger.warning("统计 Chroma 分块数失败: %s", e)
            qs = KnowledgeDocument.objects.all().order_by('-created_at')
            return [{
                "filename": d.filename,
                "chunk_count": counts.get(d.filename, d.chunk_count),
                "content_length": d.content_length,
                "created_at": timezone.localtime(d.created_at).strftime('%Y-%m-%d %H:%M'),
            } for d in qs]
        except Exception as e:
            logger.warning("读取文档列表失败: %s", e)
            return []

    def preview_chunks(self, limit: int = 12, source: str | None = None) -> list[dict]:
        """
        知识块内容预览：取最近入库的 limit 块（可按来源文件过滤）。

        Args:
            limit: 最多返回的块数
            source: 来源文件名（传入则只返回该文件的知识块）

        Returns:
            [{id, source, create_time, text}, ...]（text 为完整原文，前端负责截断/展开）
        """
        got = self.chroma._collection.get(include=["documents", "metadatas"])
        items = []
        for cid, text, meta in zip(
            got.get("ids") or [],
            got.get("documents") or [],
            got.get("metadatas") or [],
        ):
            meta = meta or {}
            if source and meta.get("source") != source:
                continue
            items.append({
                "id": cid,
                "source": meta.get("source", "未知来源"),
                "create_time": meta.get("create_time", ""),
                "text": text or "",
            })
        # Chroma 的 get 无排序保证；create_time 为 %Y-%m-%d %H:%M:%S 格式，字典序即时间序
        items.sort(key=lambda x: x["create_time"], reverse=True)
        return items[:max(1, limit)]

    def delete_document(self, filename: str) -> dict:
        """
        删除指定来源文档：Chroma 知识块 + Django 上传记录 + MD5 去重记录一并清除。

        Args:
            filename: 来源文件名

        Returns:
            {"success": True/False, "deleted_chunks": 删除的知识块数,
             "deleted_records": 删除的上传记录数, "message": 描述}
        """
        deleted_chunks = 0
        try:
            # Chroma 按 metadata.source 删除对应知识块
            result = self.chroma._collection.delete(where={"source": filename})
            # delete 的返回值在不同版本里可能是 ID 列表或 None
            if isinstance(result, list):
                deleted_chunks = len(result)
        except Exception as e:
            logger.warning("删除 Chroma 知识块失败（%s）: %s", filename, e)

        deleted_records = 0
        md5_hashes = []
        try:
            from apps.core.models import KnowledgeDocument
            qs = KnowledgeDocument.objects.filter(filename=filename)
            md5_hashes = list(qs.values_list('md5_hash', flat=True))
            if md5_hashes:
                deleted_records = qs.count()
                qs.delete()
        except Exception as e:
            logger.warning("删除 Django 上传记录失败（%s）: %s", filename, e)

        # 同步清理 MD5 去重记录，使同名内容之后可以重新入库
        if md5_hashes:
            try:
                if os.path.exists(config.md5_path):
                    with open(config.md5_path, encoding='utf-8') as f:
                        lines = [ln.strip() for ln in f if ln.strip()]
                    kept = [ln for ln in lines if ln not in md5_hashes]
                    with open(config.md5_path, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(kept) + ('\n' if kept else ''))
            except Exception as e:
                logger.warning("清理 MD5 记录失败: %s", e)

        return {
            "success": True,
            "deleted_chunks": deleted_chunks,
            "deleted_records": deleted_records,
            "message": f"已删除「{filename}」",
        }

    def _save_to_db(self, filename: str, md5_hash: str, content_length: int, chunk_count: int):
        """将上传记录同步到 Django 数据库（可选，失败不影响入库）"""
        try:
            from apps.core.models import KnowledgeDocument
            KnowledgeDocument.objects.update_or_create(
                md5_hash=md5_hash,
                defaults={
                    'filename': filename,
                    'content_length': content_length,
                    'chunk_count': chunk_count,
                    'operator': config.operator,
                }
            )
        except Exception as e:
            logger.warning("上传记录写入 Django 失败（%s）: %s", filename, e)


# 全局单例
knowledge_base_service = KnowledgeBaseService()
