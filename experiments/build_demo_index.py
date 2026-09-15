"""
构建 BYOK 演示页的静态向量索引。

用法：在项目根目录、配置了 OPENAI_API_KEY（SiliconFlow）的终端执行
    python experiments/build_demo_index.py

产出：experiments/demo-index.js
    内容为 `window.DEMO_INDEX = {...}`（比 .json 多一层包装，
    是为了 file:// 直开也能用 <script> 加载，不受浏览器 fetch 限制）

产出物拷贝到作品集仓库 assets/ 下即可被 demo 页使用。
无密钥时会执行"干跑"：只切块、不调 API，用于校验切块逻辑。
"""
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apps.core.services.chunking import split_text  # noqa: E402

# 与项目线上一致的切块参数（chunk 实验实测最优）
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
EMBED_MODEL = "BAAI/bge-m3"
EMBED_BASE = "https://api.siliconflow.cn/v1"
BATCH = 32          # 每批嵌入的块数
SLEEP = 0.4         # 批间隔，防限流

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_PATH = Path(__file__).resolve().parent / "demo-index.js"


def build_chunks():
    """读取 data/ 下全部文档并切块"""
    all_chunks = []
    for path in sorted(DATA_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        pieces = split_text(text, CHUNK_SIZE, CHUNK_OVERLAP, "paragraph", True)
        for p in pieces:
            all_chunks.append({"source": path.name, "text": p})
        print(f"  {path.name}: {len(pieces)} 块")
    return all_chunks


def embed_all(chunks):
    """调 SiliconFlow bge-m3 逐批嵌入（需环境变量 OPENAI_API_KEY）"""
    from langchain_openai import OpenAIEmbeddings

    embedder = OpenAIEmbeddings(
        openai_api_base=EMBED_BASE,
        model=EMBED_MODEL,
        # 发送原文而非 token id：第三方接口词表不同，token id 会产生
        # 语义错乱的向量；浏览器端查询走原始 API，两边必须同一空间
        check_embedding_ctx_length=False,
    )
    vectors = []
    texts = [c["text"] for c in chunks]
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        vectors.extend(embedder.embed_documents(batch))
        print(f"  嵌入进度 {min(i + BATCH, len(texts))}/{len(texts)}")
        time.sleep(SLEEP)
    return vectors


def main():
    print("=== 1. 读取并切块（300/50，与线上一致） ===")
    chunks = build_chunks()
    print(f"  共 {len(chunks)} 块")

    if not os.environ.get("OPENAI_API_KEY"):
        print("\n[干跑模式] 未检测到 OPENAI_API_KEY：切块逻辑校验完成，未生成向量。")
        print("请在配置过密钥的终端重新运行本脚本以生成完整索引。")
        return

    print("\n=== 2. 调用 bge-m3 生成向量 ===")
    vectors = embed_all(chunks)
    assert len(vectors) == len(chunks)
    # 保留 5 位小数：对余弦相似度的影响可忽略，文件体积小约三成
    vectors = [[round(x, 5) for x in v] for v in vectors]

    print("\n=== 3. 写出索引 ===")
    index = {
        "model": EMBED_MODEL,
        "provider": "siliconflow",
        "built_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "dim": len(vectors[0]),
        "chunks": [
            {"source": c["source"], "text": c["text"], "vector": v}
            for c, v in zip(chunks, vectors)
        ],
    }
    payload = "window.DEMO_INDEX = " + json.dumps(index, ensure_ascii=False) + ";"
    OUT_PATH.write_text(payload, encoding="utf-8")
    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"  已写出 {OUT_PATH}（{size_kb:.0f} KB，{len(chunks)} 块 × {index['dim']} 维）")
    print("\n下一步：把 demo-index.js 拷贝到作品集仓库的 assets/ 目录。")


if __name__ == "__main__":
    main()
