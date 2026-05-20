"""
全局配置：路径、API、超参数
"""
import os

# ============================================================
# 项目根目录（自动推断）
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# 数据路径
# ============================================================
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# --- DPR 下载的原始文件（只读，下载后不再改动）---
DPR_DIR             = os.path.join(DATA_DIR, "dpr_download")
DPR_NQ_TRAIN        = os.path.join(DPR_DIR, "nq-train.json")
DPR_TRIVIA_TRAIN    = os.path.join(DPR_DIR, "trivia-train.json")
DPR_NQ_TEST         = os.path.join(DPR_DIR, "nq-test.csv")
DPR_TRIVIA_TEST     = os.path.join(DPR_DIR, "trivia-test.csv")

# --- 各步骤的输出目录 ---
QUERIES_DIR   = os.path.join(DATA_DIR, "queries")      # step1 输出
RETRIEVED_DIR = os.path.join(DATA_DIR, "retrieved")     # step2 输出
ANNOTATED_DIR = os.path.join(DATA_DIR, "annotated")     # step3 输出
FINAL_DIR     = os.path.join(DATA_DIR, "final")         # step4/5 输出

# ============================================================
# GPT annotation API config
# ============================================================
GPT_API_URL = "https://api.xty.app/v1/chat/completions"
# GPT_API_URL = "https://hk.xty.app/v1"
GPT_API_KEY = os.environ.get("GPT_API_KEY", "")

GPT_MODEL   = "gpt-5.4"

# ============================================================
# 数据规模
# ============================================================
QUERIES_PER_DATASET  = 6000          # 每个数据集抽取的初始 query 数量
DATASETS             = ["nq", "triviaqa"]
CONFIDENCE_THRESHOLD = 0.75          # distracting 标注的最低置信度
TRAIN_RATIO          = 0.8           # 训练集占比

# ============================================================
# 检索配置
# ============================================================
DPR_QUESTION_ENCODER = "facebook/dpr-question_encoder-single-nq-base"
RETRIEVAL_TOP_K      = 1             # 检索 top-k 文档

# ============================================================
# 目标 LLM（用于后续隐藏状态提取）
# ============================================================
TARGET_LLM = "meta-llama/Llama-3.1-8B-Instruct"