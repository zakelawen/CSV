"""Global configuration for paths, APIs, and experiment constants."""

import os


# ============================================================
# Project root
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# Data paths
# ============================================================
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Raw DPR files. These are treated as read-only after download.
DPR_DIR = os.path.join(DATA_DIR, "dpr_download")
DPR_NQ_TRAIN = os.path.join(DPR_DIR, "nq-train.json")
DPR_TRIVIA_TRAIN = os.path.join(DPR_DIR, "trivia-train.json")
DPR_NQ_TEST = os.path.join(DPR_DIR, "nq-test.csv")
DPR_TRIVIA_TEST = os.path.join(DPR_DIR, "trivia-test.csv")

# Step outputs.
QUERIES_DIR = os.path.join(DATA_DIR, "queries")
RETRIEVED_DIR = os.path.join(DATA_DIR, "retrieved")
ANNOTATED_DIR = os.path.join(DATA_DIR, "annotated")
FINAL_DIR = os.path.join(DATA_DIR, "final")


# ============================================================
# LLM annotation API
# ============================================================
GPT_API_URL = os.environ.get(
    "GPT_API_URL",
    "https://api.xty.app/v1/chat/completions",
)
GPT_API_KEY = os.environ.get("GPT_API_KEY", "")
GPT_MODEL = os.environ.get("GPT_MODEL", "gpt-5.4")


# ============================================================
# Dataset construction
# ============================================================
QUERIES_PER_DATASET = 6000
DATASETS = ["nq", "triviaqa"]
CONFIDENCE_THRESHOLD = 0.75
TRAIN_RATIO = 0.8


# ============================================================
# Retrieval
# ============================================================
DPR_QUESTION_ENCODER = "facebook/dpr-question_encoder-single-nq-base"
RETRIEVAL_TOP_K = 1


# ============================================================
# Default target LLM identifier
# ============================================================
TARGET_LLM = "meta-llama/Llama-3.1-8B-Instruct"
