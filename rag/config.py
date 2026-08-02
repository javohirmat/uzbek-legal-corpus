"""Single place to edit paths, models and retrieval knobs."""
import os

# ---- corpus source -----------------------------------------------------
# "local" reads ../data/articles/*.jsonl straight out of this repo (no HF auth,
# no download, no schema drift). "hf" pulls javohirmat/uzbek-legal-corpus.
CORPUS_SOURCE = os.getenv("CORPUS_SOURCE", "local")
LOCAL_ARTICLES = os.getenv(
    "LOCAL_ARTICLES",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "articles"),
)
HF_DATASET = "javohirmat/uzbek-legal-corpus"
HF_SPLIT = "train"

# ---- build artifacts ---------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_JSON = os.path.join(HERE, "index.json")   # deterministic layer
CHROMA_DIR = os.path.join(HERE, "chroma")       # vector store
COLLECTION = "uz_law"

# ---- embeddings --------------------------------------------------------
EMBED_MODEL = "BAAI/bge-m3"
BUILD_DEVICE = os.getenv("BUILD_DEVICE", "cuda")  # build BEFORE vLLM starts -> GPU is free
QUERY_DEVICE = os.getenv("QUERY_DEVICE", "cpu")   # cpu = zero VRAM contention with vLLM
                                                  # set to "cuda" if you run vLLM at
                                                  # --gpu-memory-utilization 0.85 or lower

# ---- vLLM (OpenAI-compatible endpoint) ---------------------------------
VLLM_MODEL = os.getenv("VLLM_MODEL", "tomaris")   # the DPO LoRA adapter name
VLLM_BASE = os.getenv("VLLM_BASE", "http://localhost:8001/v1")
VLLM_KEY = "EMPTY"
TEMPERATURE = 0.1
MAX_TOKENS = 1200

# The base model is a reasoning model: left alone it emits a long hidden <think>
# trace before every answer. On the RAG path the article text is already supplied
# verbatim, so that reasoning mostly re-derives what it was handed -- it is the
# dominant cost per request. The chat template exposes `enable_thinking`.
ENABLE_THINKING = os.getenv("ENABLE_THINKING", "false").lower() == "true"

# ---- retrieval ---------------------------------------------------------
# Customer-configured "ask X -> answer Y" rules. Checked before everything else.
OVERRIDES_JSON = os.getenv("OVERRIDES_JSON", os.path.join(HERE, "overrides.json"))

TOP_K = 6                 # articles handed to the model on the semantic path
CANDIDATES = 20           # per-retriever candidates before rank fusion
RRF_K = 60                # reciprocal-rank-fusion constant
MAX_ARTICLE_CHARS = 6000  # guard for the few very long articles (e.g. Soliq 483)
HISTORY_TURNS = 6         # how far back to look for a code name in a chat thread
