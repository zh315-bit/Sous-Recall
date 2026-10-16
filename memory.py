"""Dietary-preference memory.

Interface: get_preferences(person="me") -> list[str], add_preference(fact, person="me") -> None,
history() -> list[dict]. Preferences are stored per person; "me" is the default user.

Backends (env MEMORY_BACKEND):
- "surreal" (default): SurrealDB Agent Memory is the persistent store.
    * Each preference is written with POST /facts (infer="full", observed_at, scoped). The platform
      extracts it as an attribute and natively reconciles conflicts on the same key; those native
      `corrections` (old -> new) are recorded on the fact.
    * When a new fact supersedes an old one, a timestamped retraction is also written
      ("User preference superseded: <old>"), so the supersede history lives in SurrealDB too.
    * On startup (empty local cache) the ledger is rebuilt from SurrealDB recall by replaying
      preference / retraction chunks in time order.
- "local": the JSON ledger only, same behaviour.

memory.json is the fast local ledger/cache (recall takes seconds, too slow per voice turn).
Which old facts a new one supersedes is decided by the LLM, so it works the same on both backends.
"""

import json
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()  # so `python memory.py` works standalone

import llm  # noqa: E402

BACKEND = os.getenv("MEMORY_BACKEND", "surreal").lower()
LEDGER = Path(__file__).parent / "memory.json"

SURREAL_URL = os.getenv("SPECTRON_URL", "").rstrip("/")
SURREAL_CTX = os.getenv("SPECTRON_CONTEXT_ID", "")
SURREAL_KEY = os.getenv("SPECTRON_API_KEY", "")
SCOPE = os.getenv("MEMORY_SCOPE", "app/handsfree-chef/user/default").strip("/") + "/"
RECALL_QUERY = "User preference: dietary preferences, allergies, food dislikes"
PREF, RETRACT = "User preference", "User preference superseded"  # + " [person|type]: fact"
RESET = "User preference reset"  # marker: everything before it is forgotten
ME = "me"

_hydrated = False
_lock = threading.RLock()  # memory writes can run in background threads


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------- local ledger ----------

def _load() -> list[dict]:
    try:
        return json.loads(LEDGER.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save(records: list[dict]) -> None:
    LEDGER.write_text(json.dumps(records, indent=2))


def _current(records: list[dict]) -> list[dict]:
    return [r for r in records if not r.get("superseded_at")]


def norm_person(person: str | None) -> str:
    p = (person or ME).strip().lower()
    return ME if p in ("", "me", "i", "myself", "user", "the user") else p


def _record(fact: str, ts: str, person: str = ME, type: str | None = None) -> dict:
    return {"id": uuid.uuid4().hex[:8], "person": person, "fact": fact, "type": type, "created_at": ts,
            "superseded_at": None, "superseded_by": None}


def _person_of(r: dict) -> str:
    return r.get("person") or ME


def _chunk(prefix: str, person: str, fact: str, type: str | None = None) -> str:
    tag = f"{person}|{type}" if type else person
    return f"{prefix} [{tag}]: {fact}"


def _parse_chunk(text: str) -> tuple[str, str, str, str | None] | None:
    """'User preference [alex|diet]: vegetarian' -> ('pref', 'alex', 'vegetarian', 'diet').
    Kinds: pref / retract / reset. Untagged = me."""
    if text.startswith(RESET):
        return "reset", ME, "", None
    for kind, prefix in (("retract", RETRACT), ("pref", PREF)):
        if text.startswith(prefix):
            rest = text[len(prefix):]
            person, type = ME, None
            if rest.startswith(" [") and "]" in rest:
                tag, rest = rest[2:rest.index("]")], rest[rest.index("]") + 1:]
                person, _, type = tag.partition("|")
            if rest.startswith(":"):
                return kind, norm_person(person), rest[1:].strip(), type or None
    return None


# ---------- SurrealDB Agent Memory ----------

def _surreal(path: str, body: dict, timeout: float = 30) -> dict:
    r = httpx.post(
        f"{SURREAL_URL}/api/v1/{SURREAL_CTX}/{path}",
        headers={"Authorization": f"Bearer {SURREAL_KEY}"},
        json=body,
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def _surreal_remember(text: str, ts: str, infer: str = "full") -> dict:
    body = {"text": text, "infer": infer, "scopes": [SCOPE]}
    if infer != "none":  # API rejects observed_at on literal (infer=none) writes
        body["observed_at"] = ts
    return _surreal("facts", body, timeout=60)


def _hydrate() -> None:
    """Rebuild an empty local ledger from SurrealDB (e.g. new machine / deleted memory.json)."""
    global _hydrated
    if _hydrated or BACKEND != "surreal":
        return
    _hydrated = True
    if _load():
        return
    try:
        # includeDuplicates: identical chunks (e.g. a fact re-added after being superseded) must all replay
        res = _surreal("query", {"query": RECALL_QUERY, "lens": [SCOPE], "k": 50, "includeDuplicates": True},
                       timeout=60)
    except Exception as e:
        print(f"[memory] SurrealDB recall failed, starting empty: {e}")
        return
    chunks = sorted((h for h in res.get("hits", []) if h.get("text")), key=lambda h: h.get("occurredAt", ""))
    records: list[dict] = []
    for h in chunks:
        parsed = _parse_chunk(h["text"].strip())
        if not parsed:
            continue
        kind, person, fact, type = parsed
        ts = h.get("occurredAt") or _now()
        if kind == "reset":
            records = []
            continue
        same = [r for r in _current(records) if _person_of(r) == person and r["fact"].lower() == fact.lower()]
        if kind == "retract":
            for r in same:
                r["superseded_at"] = ts
        elif not same:
            records.append(_record(fact, ts, person, type))
    if records:
        _save(records)
        print(f"[memory] restored {len(records)} facts from SurrealDB")


# ---------- public interface ----------

def _surreal_write(new: dict, retracted: list[str]) -> None:
    """Persist one fact (plus retractions of the facts it supersedes) to SurrealDB."""
    if BACKEND != "surreal":
        return
    person, ts = new["person"], new["created_at"]
    try:
        for old in retracted:  # plain chunk, no extraction: just the timestamped retraction
            _surreal_remember(_chunk(RETRACT, person, old), ts, infer="none")
        res = _surreal_remember(_chunk(PREF, person, new["fact"], new.get("type")), ts)
        corrections = res.get("extraction", {}).get("corrections", [])
        if corrections:
            new["native_corrections"] = [
                {"key": c.get("key"), "old": c.get("oldValue"), "new": c.get("newValue")} for c in corrections
            ]
        new["surreal_turn_id"] = res.get("turnId")
    except Exception as e:
        print(f"[memory] SurrealDB write failed (kept in local ledger): {e}")


def add_preference(fact: str, person: str = ME, type: str | None = None) -> None:
    with _lock:
        _add_preference(fact, person, type)


def _add_preference(fact: str, person: str, type: str | None) -> None:
    fact, person = fact.strip(), norm_person(person)
    if not fact:
        return
    _hydrate()
    records = _load()
    current = [r for r in _current(records) if _person_of(r) == person]  # supersede within a person
    try:
        idx = llm.find_superseded(fact, [r["fact"] for r in current])
    except Exception as e:
        print(f"[memory] supersede check failed, appending only: {e}")
        idx = []
    ts = _now()
    new = _record(fact, ts, person, type)
    for i in idx:
        current[i]["superseded_at"] = ts
        current[i]["superseded_by"] = new["id"]
    _surreal_write(new, [current[i]["fact"] for i in idx])
    records.append(new)
    _save(records)


def add_preferences(items: list[dict], person: str = ME) -> None:
    """Batch add [{"fact", "type"}] for one person with no conflict check (used by onboarding,
    when memory is empty). Saved locally right away; SurrealDB writes run in a background thread
    (in parallel) so the voice turn doesn't wait ~4s per fact."""
    person = norm_person(person)
    ts = _now()
    new = [_record(i["fact"].strip(), ts, person, i.get("type")) for i in items if i.get("fact", "").strip()]
    if not new:
        return
    with _lock:
        _hydrate()
        _save(_load() + new)

    def persist():
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda r: _surreal_write(r, []), new))
    if BACKEND == "surreal":
        threading.Thread(target=persist, daemon=True).start()


def reset_all() -> None:
    """Forget everyone's preferences. SurrealDB gets a reset marker so a later rebuild starts empty."""
    global _hydrated
    with _lock:
        _save([])
        _hydrated = True
    if BACKEND == "surreal":
        try:
            ts = _now()  # timestamp in the text keeps each marker unique (recall dedupes identical chunks)
            _surreal_remember(f"{RESET} at {ts}: all preferences cleared", ts, infer="none")
        except Exception as e:
            print(f"[memory] SurrealDB reset marker failed: {e}")


def get_preferences(person: str = ME) -> list[str]:
    _hydrate()
    person = norm_person(person)
    return [r["fact"] for r in _current(_load()) if _person_of(r) == person]


def people() -> list[str]:
    """Everyone with at least one stored fact, "me" first."""
    _hydrate()
    seen = {_person_of(r) for r in _load()}
    return [ME] * (ME in seen) + sorted(seen - {ME})


def history() -> list[dict]:
    """All facts, newest first, including superseded ones with their timestamps."""
    _hydrate()
    return sorted(_load(), key=lambda r: r["created_at"], reverse=True)


if __name__ == "__main__":
    # Quick check: python memory.py "I love spicy food" "I don't eat spicy food anymore"
    import sys

    # "alex: vegetarian" stores for alex; plain text stores for me
    for f in sys.argv[1:]:
        who, _, fact = f.rpartition(":") if ":" in f else ("", "", f)
        add_preference(fact, who or ME)
    print("backend:", BACKEND)
    print("current:", {p: get_preferences(p) for p in people()})
    print("history:", json.dumps(history(), indent=2))
