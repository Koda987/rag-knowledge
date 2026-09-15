"""
chunk 参数对比实验 —— 用数据回答"chunk_size / overlap 怎么定"

用法（在项目根目录、配置过 OPENAI_API_KEY 的终端里执行）：
    python experiments/chunk_experiment.py

做什么：
    1. 用 16 道评估题（每题标注正确答案来自哪篇文档）衡量检索质量
    2. 在 6 组 chunk 参数下重建知识库并逐题检索，统计命中率
    3. 彩蛋：用英文 embedding 模型跑一遍基准参数（验证中文模型选型）
    4. 结果写入 experiments/results.md

指标说明：
    - hit@1：检索第一名就命中正确文档的比例（本实验的主指标）
    - MRR：第一名=1分、第二名=0.5分……的平均（衡量整体排序质量）
    - 不统计 hit@5：语料只有 3 篇文档，Top-5 几乎必然全覆盖，没有区分度
"""
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

# 让本脚本能 import 项目根目录的 config_data
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config_data as config

# Windows 控制台中文兼容
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ============================================================
# 1. 评估集：16 道题 × 正确答案所在文档
#    （含大量"换述题"——问题和原文用词不同，考语义检索而非关键词匹配）
# ============================================================
EVAL = [
    # —— 尺码推荐 ——
    ("我身高178cm、体重140斤，应该穿什么尺码？", "尺码推荐.txt"),
    ("体重200斤的人选多大码？", "尺码推荐.txt"),
    ("最大的尺码是多少，适合什么身高体重？", "尺码推荐.txt"),
    ("160cm、95斤穿S还是M？", "尺码推荐.txt"),
    # —— 洗涤养护 ——
    ("真丝连衣裙可以机洗吗？", "洗涤养护.txt"),
    ("羊毛衫能用洗衣机洗吗？", "洗涤养护.txt"),
    ("羽绒服洗完怎么恢复蓬松？", "洗涤养护.txt"),
    ("厚羊毛大衣为什么不能水洗？", "洗涤养护.txt"),
    ("牛仔裤第一次下水要注意什么？", "洗涤养护.txt"),
    ("什么材质的衣服收纳时要放樟脑丸？", "洗涤养护.txt"),
    ("冰丝T恤能用热水洗吗？", "洗涤养护.txt"),
    ("衣服上沾了血渍怎么处理？", "洗涤养护.txt"),
    # —— 颜色选择 ——
    ("黄皮肤的人适合穿什么颜色？", "颜色选择.txt"),
    ("参加面试应该穿什么颜色？", "颜色选择.txt"),
    ("怎么搭配颜色显得高？", "颜色选择.txt"),
    ("红色和绿色能搭配在一起吗？", "颜色选择.txt"),
]

# ============================================================
# 2. 参数矩阵：(chunk_size, overlap, 备注)
# ============================================================
CONFIGS = [
    (100, 30, "小块"),
    (200, 50, ""),
    (300, 0, "无重叠（对照组：验证 overlap 的价值）"),
    (300, 50, "项目现状（基准）"),
    (500, 50, "大块"),
    (800, 100, "超大块"),
]

EMBED_MODEL = config.embedding_model_name      # BAAI/bge-m3
EMBED_MODEL_EN = "BAAI/bge-large-en-v1.5"      # 彩蛋：英文模型对照
TOP_K = config.retrieval_top_k                 # 与线上一致

DATA_DIR = config.BASE_DIR / "data"
EXP_DIR = config.BASE_DIR / "experiments"
# 每组参数一个独立子目录：Windows 下 sqlite 句柄未关闭时文件删不掉（WinError 32），
# 分目录让各组互不影响，全部跑完后再尽力统一清理
TMP_ROOT = EXP_DIR / "chroma_tmp"


def load_docs():
    """读取 data/ 下的全部知识库文档"""
    return [(p.name, p.read_text(encoding="utf-8")) for p in sorted(DATA_DIR.glob("*.txt"))]


def with_retry(fn, *args, **kwargs):
    """API 偶发失败重试（最多 3 次）"""
    for attempt in range(3):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt == 2:
                raise
            print(f"    [重试 {attempt + 1}/2] {type(e).__name__}: {e}")
            time.sleep(2)


def run_config(chunk_size, overlap, model_name, note="", slug=None):
    """一组参数：重建知识库 → 逐题检索 → 返回统计结果"""
    tag = f"chunk={chunk_size} overlap={overlap} model={model_name}"
    print(f"\n=== {tag} {note} ===")

    db_dir = TMP_ROOT / (slug or f"c{chunk_size}_o{overlap}")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=config.separators,
        length_function=len,
    )
    embeddings = OpenAIEmbeddings(
        openai_api_base=config.address,
        model=model_name,
        # 发送原文而非 token id：第三方接口词表不同，token id 会产生
        # 语义错乱的向量（详见 vector_store_service.py 的注释）
        check_embedding_ctx_length=False,
    )

    try:
        if db_dir.exists():
            shutil.rmtree(db_dir)  # 清掉上次运行的残留（进程已退出，句柄已释放）
        store = Chroma(
            collection_name=f"exp_{slug or f'{chunk_size}_{overlap}'}",
            embedding_function=embeddings,
            persist_directory=str(db_dir),
        )

        # —— 入库 ——
        n_chunks = 0
        for name, text in load_docs():
            chunks = splitter.split_text(text)
            # 分批入库，控制请求大小
            for i in range(0, len(chunks), 16):
                batch = chunks[i:i + 16]
                with_retry(
                    store.add_texts, batch,
                    metadatas=[{"source": name}] * len(batch),
                )
                time.sleep(0.3)
            n_chunks += len(chunks)
        print(f"  知识库就绪：{n_chunks} 个块")

        # —— 逐题检索 ——
        retriever = store.as_retriever(search_kwargs={"k": TOP_K})
        hit1 = 0
        mrr_sum = 0.0
        fails = []
        for q, expected in EVAL:
            docs = with_retry(retriever.invoke, q)
            sources = [d.metadata["source"] for d in docs]
            rank = next((i + 1 for i, s in enumerate(sources) if s == expected), None)
            if rank:
                mrr_sum += 1.0 / rank
                if rank == 1:
                    hit1 += 1
            if rank != 1:
                # 记录"第一名未命中"的题（而非 Top-5 全脱靶——3 篇语料下后者几乎不发生）
                fails.append(q + ("（Top-5 全脱靶）" if rank is None else f"（正确文档排第 {rank} 名）"))
            time.sleep(0.2)

        result = {
            "tag": tag, "note": note, "model": model_name,
            "chunk_size": chunk_size, "overlap": overlap,
            "chunks": n_chunks,
            "hit1": hit1, "hit1_rate": hit1 / len(EVAL),
            "mrr": mrr_sum / len(EVAL),
            "fails": fails,
        }
        print(f"  hit@1 = {hit1}/{len(EVAL)} ({result['hit1_rate']:.0%})  MRR = {result['mrr']:.3f}")
        return result
    except Exception as e:
        print(f"  [该配置执行失败] {type(e).__name__}: {e}")
        raise  # 残留目录已被 gitignore，交由 main 决定是否继续


def write_report(rows):
    """把结果写成 Markdown 报告"""
    lines = [
        "# chunk 参数对比实验报告",
        "",
        f"- 实验时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- 语料：data/ 下 3 篇文档（尺码推荐 / 洗涤养护 / 颜色选择）",
        f"- 评估集：{len(EVAL)} 道题（含大量换述题，考语义检索而非字面匹配）",
        f"- 嵌入模型：{EMBED_MODEL}（经 SiliconFlow API）",
        f"- 指标：hit@1 = 检索第一名命中正确文档的比例；MRR = 排序质量",
        f"- 不统计 hit@5：语料仅 3 篇，Top-5 几乎必然全覆盖，无区分度",
        "",
        "## 结果总表",
        "",
        "| chunk_size | overlap | 模型 | 总块数 | hit@1 | MRR | 备注 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['chunk_size']} | {r['overlap']} | {r['model']} | {r['chunks']} "
            f"| {r['hit1']}/{len(EVAL)} ({r['hit1_rate']:.0%}) | {r['mrr']:.3f} | {r['note']} |"
        )
    lines += ["", "## 各配置第一名未命中的题目（诊断明细）", ""]
    for r in rows:
        if r["fails"]:
            lines.append(f"**{r['tag']}**（hit@1 {r['hit1_rate']:.0%}）未命中：")
            lines += [f"- {q}" for q in r["fails"]]
            lines.append("")
        else:
            lines.append(f"**{r['tag']}**：全部命中 🎯")
            lines.append("")
    report = EXP_DIR / "results.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告已写入：{report}")


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("缺少环境变量 OPENAI_API_KEY——请在配置过密钥的终端里运行（和跑 run.py 的同一个终端即可）")
        sys.exit(1)

    EXP_DIR.mkdir(exist_ok=True)
    rows = []
    for size, overlap, note in CONFIGS:
        try:
            rows.append(run_config(size, overlap, EMBED_MODEL, note, slug=f"c{size}_o{overlap}"))
        except Exception:
            print("  [跳过该配置，继续下一组]")
        write_report(rows)  # 每跑完一组就落盘，中途中断也不丢已有数据

    # 彩蛋：英文模型对照（平台若无此模型则跳过）
    print("\n=== 彩蛋：英文模型对照 ===")
    try:
        rows.append(run_config(300, 50, EMBED_MODEL_EN, "英文模型对照（验证中文语料选中文模型的必要性）", slug="c300_o50_en"))
        write_report(rows)
    except Exception as e:
        print(f"  英文模型不可用，跳过：{type(e).__name__}: {e}")

    # 尽力清理临时向量库（最后一个库的句柄可能仍占用，清不掉也无害：已被 gitignore）
    shutil.rmtree(TMP_ROOT, ignore_errors=True)
    if TMP_ROOT.exists():
        print("提示：experiments/chroma_tmp 有残留，可稍后手动删除（不影响任何功能）")

    write_report(rows)
    print("\n全部完成！把 results.md 交给 Claude 分析即可。")


if __name__ == "__main__":
    main()
