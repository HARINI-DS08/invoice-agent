"""
A2A Invoice Agent (best-effort, spec-compliant) for TDS GA3 Q2.
- A2A JSON-RPC endpoint at /a2a/ (message/send, tasks/get, tasks/cancel)
- Agent card at /.well-known/agent-card.json (and /a2a/.well-known/agent-card.json)
- Logs EVERY incoming request to SQLite -> view at /debug/requests
- Durable task + payment state in SQLite; payments idempotent per invoice
"""
import json, os, re, sqlite3, time, uuid, hashlib
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx

DB = os.environ.get("DB_PATH", "state.db")
LLM_URL = os.environ.get("LLM_URL", "https://aipipe.org/openai/v1/chat/completions")
LLM_KEY = os.environ.get("LLM_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
APPROVAL_LIMIT = float(os.environ.get("APPROVAL_LIMIT", "10000"))
PUBLIC_URL = os.environ.get("PUBLIC_URL") or os.environ.get("RENDER_EXTERNAL_URL") or "https://your-app.onrender.com"

app = FastAPI()


def db():
    c = sqlite3.connect(DB)
    c.execute("CREATE TABLE IF NOT EXISTS reqlog(ts REAL, method TEXT, path TEXT, body TEXT)")
    c.execute("CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY, data TEXT)")
    c.execute("CREATE TABLE IF NOT EXISTS payments(key TEXT PRIMARY KEY, receipt TEXT)")
    return c


def log(method, path, body):
    with db() as c:
        c.execute("INSERT INTO reqlog VALUES(?,?,?,?)", (time.time(), method, path, body[:20000]))


@app.middleware("http")
async def log_all(request: Request, call_next):
    body = (await request.body()).decode("utf-8", "replace")
    if not request.url.path.startswith("/debug"):
        log(request.method, str(request.url), body)
    return await call_next(request)


@app.get("/debug/requests")
def debug_requests(n: int = 30):
    with db() as c:
        rows = c.execute("SELECT ts, method, path, body FROM reqlog ORDER BY ts DESC LIMIT ?", (n,)).fetchall()
    return [{"ts": r[0], "method": r[1], "path": r[2], "body": r[3]} for r in rows]


def agent_card():
    return {
        "protocolVersion": "0.3.0",
        "name": "Invoice Agent",
        "description": "Reads invoice packages, decides pay/hold/reject/escalate with cited evidence, pays idempotently.",
        "url": PUBLIC_URL.rstrip("/") + "/a2a/",
        "preferredTransport": "JSONRPC",
        "version": "1.0.0",
        "capabilities": {"streaming": False, "pushNotifications": False, "stateTransitionHistory": True},
        "defaultInputModes": ["text/plain", "application/json"],
        "defaultOutputModes": ["application/json", "text/plain"],
        "skills": [{
            "id": "invoice_decision",
            "name": "Invoice decision",
            "description": "Decide and safely execute the business action for an invoice package.",
            "tags": ["invoice", "accounts-payable", "payments"],
        }],
    }


@app.get("/.well-known/agent-card.json")
@app.get("/.well-known/agent.json")
@app.get("/a2a/.well-known/agent-card.json")
@app.get("/a2a/.well-known/agent.json")
def card():
    return agent_card()


# ---------- invoice reasoning ----------

def collect_text(message):
    out = []
    for p in message.get("parts", []):
        if p.get("kind") == "text" or "text" in p:
            out.append(p.get("text", ""))
        elif p.get("kind") == "data" or "data" in p:
            out.append(json.dumps(p.get("data")))
        elif p.get("kind") == "file" or "file" in p:
            f = p.get("file", {})
            out.append(json.dumps({k: v for k, v in f.items() if k != "bytes"}))
    return "\n".join(out)


SYSTEM = """You are an accounts-payable agent. Read the invoice package (invoice, PO, receipt/goods-received note,
vendor records, approval policy, prior payments). Decide ONE action:
- "pay": invoice matches PO and receipt, amount within approval limit, not already paid, vendor/bank details valid.
- "hold": missing or mismatched evidence (no receipt, quantity/price mismatch, bank details changed, unclear).
- "reject": invoice is invalid or a true duplicate of an invoice already paid (same invoice number+vendor).
- "escalate": amount exceeds the approval limit or approval is missing.
A payment for a DIFFERENT invoice (different invoice number) is NOT a duplicate.
Return JSON only: {"invoice_id": str, "vendor": str, "amount": number, "currency": str,
"action": "pay|hold|reject|escalate", "reason": str,
"evidence": [{"document": str, "field": str, "value": str}]}"""


def llm_decide(text):
    if not LLM_KEY:
        return None
    try:
        r = httpx.post(LLM_URL, headers={"Authorization": f"Bearer {LLM_KEY}"}, timeout=60, json={
            "model": LLM_MODEL,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text[:30000]}],
        })
        return json.loads(r.json()["choices"][0]["message"]["content"])
    except Exception as e:
        print("LLM error", e)
        return None


def rule_decide(text):
    inv = re.search(r"invoice[_\s-]*(?:id|no|number)?\"?\s*[:#=]?\s*\"?([A-Z0-9][A-Z0-9-]{2,})", text, re.I)
    amt = re.search(r"(?:total|amount)[^0-9]{0,15}([0-9][0-9,]*\.?[0-9]*)", text, re.I)
    amount = float(amt.group(1).replace(",", "")) if amt else 0.0
    action = "escalate" if amount > APPROVAL_LIMIT else ("pay" if "receipt" in text.lower() else "hold")
    return {"invoice_id": inv.group(1) if inv else "unknown", "vendor": "", "amount": amount, "currency": "",
            "action": action, "reason": "rule-based fallback",
            "evidence": [{"document": "package", "field": "amount", "value": str(amount)}]}


def execute(decision):
    """Carry out only approved actions. Payment is idempotent per vendor+invoice."""
    if decision.get("action") != "pay":
        return {"executed": False, "status": decision.get("action")}
    key = hashlib.sha256(f"{decision.get('vendor','').lower()}|{decision.get('invoice_id','')}".encode()).hexdigest()
    with db() as c:
        row = c.execute("SELECT receipt FROM payments WHERE key=?", (key,)).fetchone()
        if row:
            r = json.loads(row[0])
            return {"executed": False, "status": "already_paid", "duplicate_prevented": True, "receipt": r}
        receipt = {"receipt_id": "rcpt-" + uuid.uuid4().hex[:12], "invoice_id": decision.get("invoice_id"),
                   "amount": decision.get("amount"), "currency": decision.get("currency"),
                   "idempotency_key": key, "paid_at": time.time()}
        c.execute("INSERT INTO payments VALUES(?,?)", (key, json.dumps(receipt)))
    return {"executed": True, "status": "paid", "receipt": receipt}


# ---------- A2A JSON-RPC ----------

def save_task(t):
    with db() as c:
        c.execute("INSERT OR REPLACE INTO tasks VALUES(?,?)", (t["id"], json.dumps(t)))


def load_task(tid):
    with db() as c:
        row = c.execute("SELECT data FROM tasks WHERE id=?", (tid,)).fetchone()
    return json.loads(row[0]) if row else None


def rpc_ok(rid, result):
    return JSONResponse({"jsonrpc": "2.0", "id": rid, "result": result})


def rpc_err(rid, code, msg):
    return JSONResponse({"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": msg}})


@app.post("/a2a")
@app.post("/a2a/")
@app.post("/")
async def a2a(request: Request):
    try:
        req = await request.json()
    except Exception:
        return rpc_err(None, -32700, "Parse error")
    rid, method, params = req.get("id"), req.get("method"), req.get("params", {}) or {}

    if method in ("message/send", "tasks/send"):
        msg = params.get("message", {})
        msg_id = msg.get("messageId") or uuid.uuid4().hex
        # Retry safety: same messageId -> return the stored task, never re-execute
        tid = msg.get("taskId") or "task-" + hashlib.sha256(msg_id.encode()).hexdigest()[:16]
        existing = load_task(tid)
        if existing and existing["status"]["state"] in ("completed", "failed", "canceled", "rejected"):
            return rpc_ok(rid, existing)
        text = collect_text(msg)
        decision = llm_decide(text) or rule_decide(text)
        result = execute(decision)
        output = {**decision, **result}
        task = {
            "kind": "task", "id": tid, "contextId": msg.get("contextId") or tid,
            "status": {"state": "completed", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "message": {"kind": "message", "role": "agent", "messageId": uuid.uuid4().hex,
                                   "parts": [{"kind": "text", "text": f"{decision['action']}: {decision.get('reason','')}"}]}},
            "history": [msg],
            "artifacts": [{"artifactId": "decision-" + tid, "name": "invoice_decision",
                           "parts": [{"kind": "data", "data": output}]}],
        }
        save_task(task)
        return rpc_ok(rid, task)

    if method == "tasks/get":
        t = load_task(params.get("id", ""))
        return rpc_ok(rid, t) if t else rpc_err(rid, -32001, "Task not found")

    if method == "tasks/cancel":
        t = load_task(params.get("id", ""))
        if not t:
            return rpc_err(rid, -32001, "Task not found")
        if t["status"]["state"] == "completed":
            return rpc_err(rid, -32002, "Task cannot be canceled")
        t["status"]["state"] = "canceled"
        save_task(t)
        return rpc_ok(rid, t)

    return rpc_err(rid, -32601, f"Method not found: {method}")


@app.get("/")
def root():
    return {"ok": True, "card": "/.well-known/agent-card.json", "a2a": "/a2a/"}
