#!/bin/bash
# Register vLLM + the RAG API as supervisor services on a Vast.ai instance,
# and expose ONLY the RAG API through the Caddy token-auth edge.
#
#   bash install_services.sh [EXTERNAL_PORT]
#
# EXTERNAL_PORT must be a free "normal" open port on the instance:
#   vast-capabilities | jq '.instance.open_ports[]|select(.in_use==false)'
#
# AUTH=on  (default) publish through Caddy; every request needs
#          Authorization: Bearer $OPEN_BUTTON_TOKEN. The token is printed at
#          the end -- the frontend needs it as well as the URL.
# AUTH=off bind the API straight to the mapped port: no Caddy, no token,
#          reachable by anyone with the URL. Only for a closed demo.
set -euo pipefail

EXT_PORT="${1:-10100}"
AUTH="${AUTH:-on}"
if [ "$AUTH" = "off" ]; then
  BIND_HOST=0.0.0.0; BIND_PORT="$EXT_PORT"; PORTAL_GUARD=""
else
  BIND_HOST=127.0.0.1; BIND_PORT=8000
  PORTAL_GUARD='. "${utils}/exit_portal.sh" "Tomaris RAG"'
fi
REPO="${REPO:-/workspace/repo}"
VLLM_VENV="${VLLM_VENV:-/workspace/venv-vllm}"
RAG_VENV="${RAG_VENV:-/workspace/venv-rag}"
MODEL="${MODEL:-/workspace/model}"
ADAPTER="${ADAPTER:-/workspace/dpo_adapter}"

if [ -z "$(eval echo "\${VAST_TCP_PORT_${EXT_PORT}:-}")" ]; then
  echo "VAST_TCP_PORT_${EXT_PORT} is not set — ${EXT_PORT} is not an open port here." >&2
  exit 1
fi

# --- vLLM: internal only. No portal entry: it has no auth of its own. --------
cat > /opt/supervisor-scripts/tomaris-vllm.sh << SH
#!/bin/bash
utils=/opt/supervisor-scripts/utils
. "\${utils}/logging.sh"
. "\${utils}/cleanup_generic.sh"
. "\${utils}/environment.sh"

cd /workspace
pty ${VLLM_VENV}/bin/vllm serve ${MODEL} \\
  --enable-lora --lora-modules tomaris=${ADAPTER} --max-lora-rank 64 \\
  --dtype bfloat16 --max-model-len 8192 --port 8001 --host 127.0.0.1 \\
  --served-model-name tomaris-base --gpu-memory-utilization 0.88 \\
  --language-model-only --reasoning-parser deepseek_r1 \\
  --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1,2,4,8]}' 2>&1
SH

# --- RAG API: internal 8000, published on EXT_PORT behind Caddy auth --------
cat > /opt/supervisor-scripts/tomaris-rag.sh << SH
#!/bin/bash
utils=/opt/supervisor-scripts/utils
. "\${utils}/logging.sh"
. "\${utils}/cleanup_generic.sh"
. "\${utils}/environment.sh"
${PORTAL_GUARD}
cd ${REPO}/rag
pty ${RAG_VENV}/bin/uvicorn server:app --host ${BIND_HOST} --port ${BIND_PORT} 2>&1
SH

chmod +x /opt/supervisor-scripts/tomaris-vllm.sh /opt/supervisor-scripts/tomaris-rag.sh

for n in vllm rag; do
cat > "/etc/supervisor/conf.d/tomaris-${n}.conf" << CONF
[program:tomaris-${n}]
environment=PROC_NAME="%(program_name)s"
command=/opt/supervisor-scripts/tomaris-${n}.sh
autostart=true
autorestart=true
startsecs=30
stopwaitsecs=60
killasgroup=true
stdout_logfile=/dev/stdout
redirect_stderr=true
stdout_logfile_maxbytes=0
CONF
done

/venv/main/bin/python - "$EXT_PORT" "$AUTH" << 'PY'
import sys, yaml
port, auth = int(sys.argv[1]), sys.argv[2]
d = yaml.safe_load(open("/etc/portal.yaml")) or {"applications": {}}
if auth == "off":
    # No Caddy entry: the app owns the mapped port directly, so no token.
    d["applications"].pop("Tomaris RAG", None)
    print(f"portal: no entry -> API bound directly to {port}, NO token required")
else:
    d["applications"]["Tomaris RAG"] = {
        "hostname": "localhost", "external_port": port,
        "internal_port": 8000, "open_path": "/health", "name": "Tomaris RAG",
    }
    print(f"portal: Tomaris RAG -> external {port} / internal 8000 (token required)")
yaml.safe_dump(d, open("/etc/portal.yaml", "w"), sort_keys=False)
PY

supervisorctl reread && supervisorctl update && supervisorctl restart caddy >/dev/null
PUB=$(eval echo "\$VAST_TCP_PORT_${EXT_PORT}")
echo
echo "services registered. vLLM takes ~3 min to load; watch:"
echo "  supervisorctl status | grep tomaris"
echo
echo "Frontend VAST_API_URL:  http://${PUBLIC_IPADDR}:${PUB}"
if [ "$AUTH" = "off" ]; then
  echo "Auth: NONE - anyone with the URL can use the GPU."
else
  echo "Auth: REQUIRED - send this header on every request:"
  echo "  Authorization: Bearer ${OPEN_BUTTON_TOKEN}"
fi
