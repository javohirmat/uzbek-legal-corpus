#!/bin/bash
# Fresh Vast.ai box -> running Tomaris stack. Run as root on the instance:
#
#   git clone https://github.com/javohirmat/uzbek-legal-corpus.git /workspace/repo
#   bash /workspace/repo/rag/bootstrap.sh
#
# Requires: CUDA driver >= 12.8, disk >= 150 GB, and `hf auth login` already
# done (the model repos are private; this script will stop and tell you).
set -euo pipefail

W=/workspace
REPO=$W/repo

say() { printf "\n\033[1m== %s\033[0m\n" "$1"; }

say "0. preflight"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
CUDA=$(nvidia-smi | grep -oE "CUDA Version: [0-9]+\.[0-9]+" | grep -oE "[0-9]+\.[0-9]+")
awk -v c="$CUDA" 'BEGIN{if (c+0 < 12.8) {print "ERROR: CUDA " c " < 12.8. Torch/vLLM wheels will fail. Rent a newer box."; exit 1}}'
df -h $W | tail -1
if ! /venv/main/bin/hf auth whoami >/dev/null 2>&1; then
  echo "ERROR: not logged in to Hugging Face. Run:  /venv/main/bin/hf auth login"
  exit 1
fi

say "1. virtualenvs (separate on purpose: vLLM pins torch, sentence-transformers upgrades it)"
# Guarded so the script can be re-run. `uv venv` refuses to touch an existing
# environment, and under `set -e` that aborts the whole run -- so a bootstrap
# interrupted anywhere after this point (a dropped 54 GB download is the likely
# one) could not simply be started again. The pip installs below stay
# unconditional: they are idempotent and near-instant against uv's cache.
[ -x $W/venv-vllm/bin/python ] || uv venv $W/venv-vllm --python 3.12 -q
[ -x $W/venv-rag/bin/python ]  || uv venv $W/venv-rag  --python 3.12 -q
uv pip install --python $W/venv-vllm/bin/python -q vllm huggingface_hub
uv pip install --python $W/venv-rag/bin/python  -q -r $REPO/rag/requirements.txt

say "2. weights (~54 GB, ~8 min)"
[ -f $W/model/config.json ]        || /venv/main/bin/hf download tomaris/Tomaris.ai     --local-dir $W/model
[ -f $W/dpo_adapter/adapter_config.json ] || /venv/main/bin/hf download tomaris/Tomaris-dpo-v1 --local-dir $W/dpo_adapter

say "3. vector index (before vLLM starts, so the GPU is free) — ~2 min"
cd $REPO/rag && $W/venv-rag/bin/python build_index.py

say "4. register services + expose the API"
# Per-customer keys. Unset = open box (fine for your own testing, NOT for a
# key you hand out). Set it in the Vast instance env, before this runs:
#   TOMARIS_API_KEYS="azizbek:t7-$(openssl rand -hex 12):200"
# name:key:daily_request_limit, comma-separated. The service refuses to start
# if the value is set but unparseable, rather than silently serving everyone.
if [ -n "${TOMARIS_API_KEYS:-}" ]; then
  echo "   API keys: configured ($(echo "$TOMARIS_API_KEYS" | tr ',' '\n' | wc -l | tr -d ' ') entry/entries)"
else
  echo "   API keys: NONE -- this box will answer anyone who finds the port."
  echo "   To meter a customer:  export TOMARIS_API_KEYS=\"name:key:200\"  and re-run step 4."
fi
PORT=$(vast-capabilities 2>/dev/null | python3 -c "
import json,sys
free=[p for p in json.load(sys.stdin)['instance']['open_ports']
      if not p['in_use'] and not p.get('self_mapped') and p['container_port']>1024]
print(free[0]['container_port'] if free else '')" 2>/dev/null || echo "")
[ -z "$PORT" ] && { echo "no free normal port; pick one manually: vast-capabilities | jq '.instance.open_ports'"; exit 1; }
bash $REPO/rag/install_services.sh "$PORT"

say "5. waiting for vLLM (~3 min to load 51 GB)"
for i in $(seq 1 60); do
  curl -s --max-time 5 localhost:8001/v1/models >/dev/null 2>&1 && break
  sleep 10
done
supervisorctl status | grep tomaris

# set -u would abort the whole run here if Vast has not exported these -- after
# the 54 GB download and a working install, which reads as "the deploy failed"
# and loses the URL the frontend needs. Degrade to a placeholder instead.
PUB=$(eval echo "\${VAST_TCP_PORT_${PORT}:-}")
[ -z "$PUB" ] && PUB="$PORT"
IP="${PUBLIC_IPADDR:-<box-public-ip>}"
say "6. verify"
curl -s "http://localhost:${PORT}/health"; echo
curl -s -X POST "http://localhost:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"JK 999-modda"}]}' \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['retrieval_mode'],'|',d['choices'][0]['message']['content'][:70])"

cat <<EOF

DONE. Give the frontend this as VAST_API_URL (base URL, no path):

    http://${IP}:${PUB}

Live transcript:  tail -f /workspace/chat-history.jsonl
Service logs:     tail -f /var/log/portal/tomaris-rag.log
Restart after a code change:
    cd /workspace/repo && git pull && supervisorctl restart tomaris-rag
EOF
