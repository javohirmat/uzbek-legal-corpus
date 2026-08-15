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
# General chat is not grounded in retrieved text, so the low legal temperature
# only hurts it: at 0.1 the model loops on a phrase when asked for anything
# creative ("alla aytib ber" came back repeating one word). Legal generation
# keeps TEMPERATURE; conversation uses this.
CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", "0.7"))
MAX_TOKENS = 900        # caps worst-case latency: 900 tokens / ~23 tok/s ~= 39s

# Reasoning mode:
#   "auto"  -> send nothing, let the chat template and the model decide per
#              question (a greeting gets none, a hard legal question gets a
#              full trace). This is the default.
#   "true"  -> force thinking on every request
#   "false" -> force it off; fastest, but the UI's Reasoning panel stays empty
# Forcing it off also requires dropping vLLM's --reasoning-parser: with no
# closing </think> the parser files the whole answer under reasoning_content
# and returns content=null.
THINKING_MODE = os.getenv("THINKING_MODE", "auto").lower()

# ---- retrieval ---------------------------------------------------------
# Customer-configured "ask X -> answer Y" rules. Checked before everything else.
OVERRIDES_JSON = os.getenv("OVERRIDES_JSON", os.path.join(HERE, "overrides.json"))

TOP_K = 6                 # articles handed to the model on the semantic path
CANDIDATES = 20           # per-retriever candidates before rank fusion
RRF_K = 60                # reciprocal-rank-fusion constant
PER_CODE_CAP = int(os.getenv("PER_CODE_CAP", "2"))  # max articles per code in TOP_K
# Situational rewrite: one extra 27B call, thinking off, hard timeout so a
# slow rewrite cannot stall a story behind the main answer.
REWRITE_TIMEOUT = float(os.getenv("REWRITE_TIMEOUT", "8"))
REWRITE_MAX_TOKENS = int(os.getenv("REWRITE_MAX_TOKENS", "200"))
SITUATION_MAX_QUERIES = int(os.getenv("SITUATION_MAX_QUERIES", "5"))
MAX_ARTICLE_CHARS = 6000  # guard for the few very long articles (e.g. Soliq 483)
HISTORY_TURNS = 6         # how far back to look for a code name in a chat thread

# Cosine distance below which the corpus is considered to have a real answer.
# Above it, a question with no explicit legal signal is treated as general chat.
# Raise -> more questions routed to law; lower -> more to chat.
LEGAL_DISTANCE = float(os.getenv("LEGAL_DISTANCE", "0.45"))

# Colloquial -> formal legal wording, applied to the retrieval query only.
SYNONYMS_JSON = os.getenv("SYNONYMS_JSON", os.path.join(HERE, "synonyms.json"))

# Durable transcript of every question and answer (survives service restarts).
CHAT_LOG = os.getenv("CHAT_LOG", "/workspace/chat-history.jsonl")

# How often the SSE stream emits a keep-alive comment while an answer is being
# generated. Must stay below any client/proxy inactivity timeout.
HEARTBEAT_SECONDS = float(os.getenv("HEARTBEAT_SECONDS", "5"))
