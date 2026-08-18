"""Unit tests for per-customer API keys: auth, daily caps, usage, persistence.

No LLM, no GPU, no network (TestClient runs in-process).

    python test_api_keys.py
"""
import json
import tempfile
import threading

from api_keys import KeyAuth, KeyAuthError

fails = []


def check(label, got, want=True):
    ok = got == want
    print(f'  {"PASS" if ok else "FAIL"}  {label:52} -> {got!r}')
    if not ok:
        fails.append(f"{label}: got {got!r}, want {want!r}")


class StubReq:
    def __init__(self, auth=None):
        self.headers = {"authorization": auth} if auth else {}


def make(env_extra=None, state=None):
    env = {"TOMARIS_API_KEYS": "", "API_USAGE_FILE": state or _state()}
    env.update(env_extra or {})
    return KeyAuth.from_env(env)


def _state():
    return tempfile.mkdtemp(prefix="api-keys-test-") + "/usage.json"


print("1. open mode: zero keys configured behaves exactly like today")
ka = make()
check("disabled when no keys", ka.enabled, False)
check("authorize returns None (open)", ka.authorize(StubReq()), None)
check("no key, no header, still open", ka.authorize(StubReq("garbage")), None)


print("\n2. parsing TOMARIS_API_KEYS")
ka = make({"TOMARIS_API_KEYS": "azizbek:t7-prod-AAA:200,qa:t7-qa-BBB,,no-colon-here"})
check("two valid keys loaded", len(ka._by_secret), 2)
check("malformed entry skipped", "no-colon-here" not in ka._by_secret, True)
check("limit parsed", ka.limit_for("azizbek"), 200)
check("qa has no limit", ka.limit_for("qa"), 0)
ka2 = make({"TOMARIS_API_KEYS": "a:key-X:5,b:key-X:9"})
check("duplicate secret first-wins", ka2.limit_for("a"), 5)
check("duplicate second ignored", ka2.limit_for("b"), 0)


print("\n3. bearer auth: 401s, case-insensitive scheme")
ka = make({"TOMARIS_API_KEYS": "azizbek:t7-prod-AAA:2"})
try:
    ka.authorize(StubReq())
    check("missing key raises", "no raise", "KeyAuthError")
except KeyAuthError as e:
    check("missing key -> 401", (e.status, e.code), (401, "invalid_api_key"))
    check("error body is OpenAI-shaped", "error" in e.body(), True)
try:
    ka.authorize(StubReq("Bearer wrong-key"))
    check("bad key raises", "no raise", "KeyAuthError")
except KeyAuthError as e:
    check("bad key -> 401", e.status, 401)
check("good key accepted", ka.authorize(StubReq("Bearer t7-prod-AAA")), "azizbek")
check("lowercase 'bearer' accepted",
      ka.authorize(StubReq("bearer t7-prod-AAA")), "azizbek")


print("\n4. daily request cap")
try:
    for _ in range(3):
        name = ka.authorize(StubReq("Bearer t7-prod-AAA"))
        ka.count_request(name)
    check("third request raises", "no raise", "KeyAuthError")
except KeyAuthError as e:
    check("over cap -> 429", (e.status, e.code), (429, "daily_limit_exceeded"))
    check("message names the reset day", "Resets" in e.message, True)
snap = ka.usage_for("azizbek")
check("requests counted up to cap", snap["requests"], 2)
check("remaining is zero", snap["remaining"], 0)
check("limit exposed", snap["daily_limit"], 2)
check("open-mode key not counted", ka.count_request(None) is None, True)


print("\n5. usage accounting accumulates real upstream tokens")
ka.add_usage("azizbek", prompt_tokens=1200, completion_tokens=180)
ka.add_usage("azizbek", prompt_tokens=300, completion_tokens=20)
snap = ka.usage_for("azizbek")
check("prompt tokens summed", snap["prompt_tokens"], 1500)
check("completion tokens summed", snap["completion_tokens"], 200)
ka.add_usage(None, 99, 99)
check("open-mode usage ignored", ka.usage_for("azizbek")["prompt_tokens"], 1500)


print("\n6. counters survive a restart")
state = _state()
ka = make({"TOMARIS_API_KEYS": "azizbek:t7-prod-AAA:0", "API_USAGE_FILE": state})
ka.count_request(ka.authorize(StubReq("Bearer t7-prod-AAA")))
ka.add_usage("azizbek", 500, 60)
reloaded = make({"TOMARIS_API_KEYS": "azizbek:t7-prod-AAA:0", "API_USAGE_FILE": state})
snap = reloaded.usage_for("azizbek")
check("requests reloaded", snap["requests"], 1)
check("tokens reloaded", (snap["prompt_tokens"], snap["completion_tokens"]), (500, 60))
with open(state, encoding="utf-8") as f:
    check("state file is plain JSON", isinstance(json.load(f)["days"], dict), True)


print("\n7. concurrent counting loses nothing")
ka = make({"TOMARIS_API_KEYS": "azizbek:t7-prod-AAA:0"})
def hit():
    for _ in range(50):
        ka.count_request("azizbek")
threads = [threading.Thread(target=hit) for _ in range(8)]
[t.start() for t in threads]
[t.join() for t in threads]
check("8 threads x 50 = 400", ka.usage_for("azizbek")["requests"], 400)


print("\n8. FastAPI wiring end-to-end (installed fastapi, in-process)")
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

app = FastAPI()
keys = make({"TOMARIS_API_KEYS": "azizbek:t7-prod-AAA:2"})

@app.post("/v1/chat/completions")
async def chat(request: Request):
    try:
        key_id = keys.admit(request)
    except KeyAuthError as e:
        return JSONResponse(status_code=e.status, content=e.body())
    return JSONResponse({"choices": [{"message": {"content": "ok"}}]})

@app.get("/v1/usage")
async def usage(request: Request):
    try:
        key_id = keys.authorize(request, check_cap=False)
    except KeyAuthError as e:
        return JSONResponse(status_code=e.status, content=e.body())
    if not key_id:
        return JSONResponse({"error": "no keys configured"})
    return JSONResponse(keys.usage_for(key_id))

client = TestClient(app)
r = client.post("/v1/chat/completions", json={"messages": []})
check("no header -> 401 JSON", (r.status_code, r.json()["error"]["code"]),
      (401, "invalid_api_key"))
r = client.post("/v1/chat/completions", json={"messages": []},
                headers={"Authorization": "Bearer t7-prod-AAA"})
check("good key -> 200", r.status_code, 200)
r = client.post("/v1/chat/completions", json={"messages": []},
                headers={"Authorization": "Bearer t7-prod-AAA"})
check("second request within cap", r.status_code, 200)
r = client.post("/v1/chat/completions", json={"messages": []},
                headers={"Authorization": "Bearer t7-prod-AAA"})
check("third request -> 429", (r.status_code, r.json()["error"]["code"]),
      (429, "daily_limit_exceeded"))
r = client.get("/v1/usage", headers={"Authorization": "Bearer t7-prod-AAA"})
check("usage endpoint reports the cap", (r.json()["requests"], r.json()["remaining"]),
      (2, 0))


print("\n9. admit() is atomic — concurrent callers cannot overshoot the cap")
# Production change that would fail this: authorize() then count_request()
# on two threads both seeing remaining=1.
ka = make({"TOMARIS_API_KEYS": "azizbek:t7-prod-AAA:10"})
admitted = []
statuses = []

def race():
    try:
        ka.admit(StubReq("Bearer t7-prod-AAA"))
        admitted.append(1)
    except KeyAuthError as e:
        statuses.append(e.status)
    except AttributeError:
        statuses.append("no-admit")

threads = [threading.Thread(target=race) for _ in range(20)]
[t.start() for t in threads]
[t.join() for t in threads]
check("exactly 10 admitted under a 10-cap", sum(admitted), 10)
check("the other 10 are 429", statuses.count(429), 10)
check("counter matches admits", ka.usage_for("azizbek")["requests"], 10)
check("open mode admit is a no-op", make().admit(StubReq()), None)


print("\n10. eval_recall --url sends the partner Bearer when TOMARIS_API_KEY is set")
from eval_recall import retrieve_http_headers
check("no env, no Authorization",
      "Authorization" in retrieve_http_headers({}), False)
check("TOMARIS_API_KEY becomes Bearer",
      retrieve_http_headers({"TOMARIS_API_KEY": " t7-secret "})["Authorization"],
      "Bearer t7-secret")
check("empty key stays omitted",
      "Authorization" in retrieve_http_headers({"TOMARIS_API_KEY": "  "}), False)


print()
if fails:
    print(f"{len(fails)} FAILURES:")
    for f in fails:
        print("  -", f)
    raise SystemExit(1)
print("all api_keys tests pass")
