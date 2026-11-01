import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

INDEX_DIR = "index_store"
CHECKPOINT_PATH = "embedding_model.pth"
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MIN_IMAGE_SCORE = 0.50
MIN_TEXT_SCORE = 0.35
UNCERTAIN_THRESHOLD = 0.15
FALLBACK_K = 20
ALPHA = 0.7
MAX_RESULTS = 200
PAGE_SIZE = 12
JUDGE_TOP_N = 20
JUDGE_THRESHOLD = 0.5
W_RETRIEVAL = 0.6
W_JUDGE = 0.4