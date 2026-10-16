import base64
import json
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # must run before importing modules that read env vars

from opentelemetry import trace  # noqa: E402

# ---------- Tracing (Arize AX) - set up before the Anthropic client is created ----------
TRACING = {"enabled": False, "project": "handsfree-chef", "error": None}
if os.getenv("ARIZE_SPACE_ID") and os.getenv("ARIZE_API_KEY"):
    try:
        from arize.otel import register
        from openinference.instrumentation.anthropic import AnthropicInstrumentor

        tracer_provider = register(
            space_id=os.getenv("ARIZE_SPACE_ID"),
            api_key=os.getenv("ARIZE_API_KEY"),
            project_name=TRACING["project"],
        )
        AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)
        TRACING["enabled"] = True
    except Exception as e:
        TRACING["error"] = str(e)
        print(f"[tracing] Arize setup failed: {e}")
else:
    TRACING["error"] = "ARIZE_SPACE_ID / ARIZE_API_KEY not set"

tracer = trace.get_tracer("handsfree-chef")

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.concurrency import run_in_threadpool  # noqa: E402
from fastapi.responses import FileResponse, StreamingResponse  # noqa: E402
from urllib.parse import quote  # noqa: E402

import llm  # noqa: E402
import memory  # noqa: E402
import voice  # noqa: E402

STATIC = Path(__file__).parent / "static"
HISTORY_TURNS = 6  # user+assistant exchanges sent to the LLM

app = FastAPI(title="Sous Recall")

# Single-user, in-memory app state. diners = who is eating this session ("me" = the user).
# mode = "onboarding" while we wait for a first-time user's allergy answer; the photo waits in pending_image.
state = {"recipe": None, "current_step": 0, "history": [], "diners": ["me"], "evals": {},
         "mode": "normal", "pending_image": None}

ONBOARD_Q = "Before we start, do you have any allergies or foods you avoid?"
NO_RESTRICTIONS = "no known allergies or restrictions"


def diner_prefs() -> dict[str, list[str]]:
    return {p: memory.get_preferences(p) for p in state["diners"]}


def prefs_text(people: list[str] | None = None) -> str:
    """Merged preferences of all diners, e.g. 'me: allergic to peanuts; alex: vegetarian'."""
    prefs = {p: memory.get_preferences(p) for p in people} if people is not None else diner_prefs()
    parts = [f"{p}: {', '.join(facts)}" for p, facts in prefs.items() if facts]
    return "; ".join(parts) or "none"


def prefs_count() -> int:
    return sum(len(f) for f in diner_prefs().values())


def tts_url(text: str) -> str:
    """Voice turns return a URL instead of base64: the browser streams it while ElevenLabs speaks."""
    return "/tts?text=" + quote(text)


def tts_b64(text: str) -> str | None:
    """TTS failures shouldn't break the request; the UI still shows the text."""
    try:
        return base64.b64encode(voice.speak(text)).decode()
    except Exception as e:
        print(f"TTS failed: {e}")
        return None


def preference_check(parent_ctx, eval_id: str, content: str) -> None:
    """LLM-as-judge eval, run after the response is sent so it adds no latency.
    Preferences are read now (not when scheduled) so a fact learned this turn, saved by an
    earlier background task, is included. Result goes to state["evals"] for the UI badge."""
    result = state["evals"][eval_id]
    preferences, n = prefs_text(), prefs_count()
    result["n"] = n
    with tracer.start_as_current_span("preference_check", context=parent_ctx) as span:
        span.set_attribute("openinference.span.kind", "EVALUATOR")
        span.set_attribute("input.value", content)
        span.set_attribute("preferences", preferences)
        if preferences == "none":
            verdict = {"violates": False, "reason": "No preferences stored."}
        else:
            try:
                verdict = llm.check_preferences(preferences, content)
            except Exception as e:
                span.set_attribute("eval.label", "error")
                span.set_attribute("eval.explanation", str(e))
                result.update(status="error", reason=str(e))
                return
        label = "fail" if verdict.get("violates") else "pass"
        span.set_attribute("eval.label", label)
        span.set_attribute("eval.explanation", verdict.get("reason", ""))
        span.set_attribute("output.value", json.dumps(verdict))
        result.update(status=label, reason=verdict.get("reason", ""))
        if label == "fail":
            print(f"[eval] preference violation: {verdict.get('reason')}")


def eval_later(bg: BackgroundTasks, content: str) -> str:
    """Schedule the eval against the diners' current preferences; returns an id the UI can poll."""
    eval_id = uuid.uuid4().hex[:10]
    state["evals"][eval_id] = {"status": "pending", "reason": "", "n": prefs_count()}
    ctx = trace.set_span_in_context(trace.get_current_span())
    bg.add_task(preference_check, ctx, eval_id, content)
    return eval_id


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"ok": True, "tracing": TRACING, "memory_backend": memory.BACKEND, "model": llm.MODEL}


def make_recipe(bg: BackgroundTasks, span, data: bytes, media_type: str) -> dict:
    """Shared by /recipe and the onboarding /talk: LLM recipe from the photo + the diners' preferences."""
    prefs = prefs_text()
    span.set_attribute("preferences", prefs)
    try:
        recipe = llm.generate_recipe(data, media_type, preferences=prefs)
    except Exception as e:
        span.record_exception(e)
        raise HTTPException(502, f"Recipe generation failed: {e}")
    state.update(recipe=recipe, current_step=0, history=[])
    span.set_attribute("dish", recipe["dish"])
    span.set_attribute("current_step", 0)
    span.set_attribute("output.value", json.dumps(recipe))
    eval_id = eval_later(bg, json.dumps(recipe))
    return {**recipe, "current_step": 0, "eval_id": eval_id, "diners": state["diners"]}


@app.post("/recipe")
async def create_recipe(bg: BackgroundTasks, image: UploadFile = File(...)):
    with tracer.start_as_current_span("recipe_request") as span:
        span.set_attribute("openinference.span.kind", "CHAIN")
        media_type = image.content_type or "image/jpeg"
        if media_type not in ("image/jpeg", "image/png", "image/gif", "image/webp"):
            raise HTTPException(400, f"Unsupported image type: {media_type} (use JPEG/PNG/WebP)")
        data = await image.read()
        if not data:
            raise HTTPException(400, "Empty image")
        if not memory.get_preferences("me"):
            # First-time user: ask about allergies before the first recipe; keep the photo for later
            state.update(mode="onboarding", pending_image=(data, media_type))
            span.set_attribute("mode", "onboarding")
            span.set_attribute("output.value", ONBOARD_Q)
            return {"needs_onboarding": True, "reply": ONBOARD_Q, "audio_url": tts_url(ONBOARD_Q)}
        # in a thread: a blocking LLM call here would also hold up the parallel /fridge request
        result = await run_in_threadpool(make_recipe, bg, span, data, media_type)
        intro = f"Let's make {result['dish']}. Step one: {result['steps'][0]}"
        return {**result, "intro": intro, "audio_url": tts_url(intro)}


@app.post("/fridge")
def fridge_items(image: UploadFile = File(...)):
    """Foods visible in the photo, for the pixel 'fridge crew'. Sync, so it runs in the threadpool next to /recipe."""
    with tracer.start_as_current_span("fridge_items") as span:
        span.set_attribute("openinference.span.kind", "CHAIN")
        data = image.file.read()
        if not data:
            raise HTTPException(400, "Empty image")
        try:
            items = llm.detect_items(data, image.content_type or "image/jpeg")
        except Exception as e:
            span.record_exception(e)
            raise HTTPException(502, f"Item detection failed: {e}")
        span.set_attribute("output.value", json.dumps(items))
        return {"items": items}


def onboarding_turn(bg: BackgroundTasks, span, transcript: str) -> dict:
    """Answer to ONBOARD_Q: store the restrictions, then cook the waiting photo."""
    span.set_attribute("mode", "onboarding")
    try:
        result = llm.onboard(transcript)
    except Exception as e:
        span.record_exception(e)
        raise HTTPException(502, f"LLM failed: {e}")
    prefs = result["preferences"] or [{"fact": NO_RESTRICTIONS, "type": None}]  # never ask again
    memory.add_preferences(prefs, "me")
    span.set_attribute("new_preference", json.dumps(prefs))
    data, media_type = state["pending_image"]
    recipe = make_recipe(bg, span, data, media_type)
    state.update(mode="normal", pending_image=None)
    step1 = recipe["steps"][0]
    reply = f"{result['ack']} Tonight let's make {recipe['dish']}. First, {step1[:1].lower() + step1[1:]}"
    state["history"] += [{"role": "user", "content": transcript},
                         {"role": "assistant", "content": json.dumps({"reply": reply, "action": "none",
                                                                     "new_preference": None, "diners": None})}]
    span.set_attribute("output.value", reply)
    return {"transcript": transcript, "reply": reply, "audio_url": tts_url(reply), "recipe": recipe,
            "current_step": 0, "preferences": prefs, "diners": state["diners"]}


@app.post("/talk")
async def talk(bg: BackgroundTasks, audio: UploadFile = File(...)):
    with tracer.start_as_current_span("voice_turn") as span:
        span.set_attribute("openinference.span.kind", "CHAIN")
        data = await audio.read()
        if not data:
            raise HTTPException(400, "Empty audio")
        try:
            transcript = voice.transcribe(data, audio.filename or "audio.webm", audio.content_type or "audio/webm")
        except Exception as e:
            span.record_exception(e)
            raise HTTPException(502, f"Transcription failed: {e}")
        span.set_attribute("transcript", transcript)
        span.set_attribute("input.value", transcript)
        if not transcript:
            reply = "Sorry, I didn't catch that. Could you say it again?"
            span.set_attribute("action", "none")
            return {"transcript": "", "reply": reply, "action": "none", "current_step": state["current_step"],
                    "new_preference": None, "diners": state["diners"], "eval_id": None,
                    "audio_url": tts_url(reply)}

        if state["mode"] == "onboarding" and state["pending_image"]:
            return onboarding_turn(bg, span, transcript)

        prefs = prefs_text()
        span.set_attribute("preferences", prefs)
        try:
            result = llm.chat(transcript, state["recipe"], state["current_step"], prefs,
                              state["history"][-HISTORY_TURNS * 2:], diners=", ".join(state["diners"]),
                              known=prefs_text(memory.people()))
        except Exception as e:
            span.record_exception(e)
            raise HTTPException(502, f"LLM failed: {e}")

        finished = False  # "next" on the last step = the dish is done (UI plays the ending)
        if result["action"] == "next" and state["recipe"]:
            last = len(state["recipe"]["steps"]) - 1
            finished = state["current_step"] >= last
            state["current_step"] = min(state["current_step"] + 1, last)

        if result["diners"]:
            diners = [memory.norm_person(d) for d in result["diners"]]
            state["diners"] = list(dict.fromkeys(["me"] + diners if "me" in diners else diners))

        new_pref = result.get("new_preference")
        if new_pref:
            new_pref["person"] = memory.norm_person(new_pref.get("person"))
            # supersede check + SurrealDB write take ~5s: do it after the reply is sent
            bg.add_task(memory.add_preference, new_pref["fact"], new_pref["person"])
            if new_pref["person"] not in state["diners"]:  # mentioned someone new: they're eating too
                state["diners"].append(new_pref["person"])

        # Store the assistant turn as JSON so the model keeps answering in JSON
        state["history"] += [
            {"role": "user", "content": transcript},
            {"role": "assistant", "content": json.dumps(result)},
        ]

        span.set_attribute("action", result["action"])
        span.set_attribute("current_step", state["current_step"])
        span.set_attribute("new_preference", json.dumps(new_pref) if new_pref else "")
        span.set_attribute("diners", ", ".join(state["diners"]))
        span.set_attribute("output.value", result["reply"])
        eval_id = eval_later(bg, result["reply"])  # checked against prefs incl. anything just learned

        return {
            "transcript": transcript,
            "reply": result["reply"],
            "action": result["action"],
            "current_step": state["current_step"],
            "new_preference": new_pref,
            "diners": state["diners"],
            "eval_id": eval_id,
            "finished": finished,
            "audio_url": tts_url(result["reply"]),
        }


@app.post("/next")
def next_step():
    """Hands-free 'next step' (OK gesture): no speech or LLM needed, just advance and read the step."""
    with tracer.start_as_current_span("gesture_next") as span:
        span.set_attribute("openinference.span.kind", "CHAIN")
        if not state["recipe"]:
            raise HTTPException(400, "No recipe yet - upload a fridge photo first")
        steps = state["recipe"]["steps"]
        finished = state["current_step"] >= len(steps) - 1
        if finished:
            reply = f"That was the last step. Your {state['recipe']['dish']} is ready. Enjoy your meal!"
        else:
            state["current_step"] += 1
            reply = f"Step {state['current_step'] + 1}: {steps[state['current_step']]}"
        state["history"] += [{"role": "user", "content": "(OK gesture: next step)"},
                             {"role": "assistant", "content": json.dumps({"reply": reply, "action": "next",
                                                                         "new_preference": None, "diners": None})}]
        span.set_attribute("action", "next")
        span.set_attribute("current_step", state["current_step"])
        span.set_attribute("output.value", reply)
        return {"reply": reply, "action": "next", "current_step": state["current_step"], "finished": finished,
                "audio_url": tts_url(reply)}


@app.get("/tts")
def tts(text: str):
    """Streamed mp3 for a reply; <audio src> starts playing on the first chunk."""
    return StreamingResponse(voice.speak_stream(text[:1000]), media_type="audio/mpeg",
                             headers={"Cache-Control": "no-store"})


@app.get("/eval/{eval_id}")
def get_eval(eval_id: str):
    if eval_id not in state["evals"]:
        raise HTTPException(404, "Unknown eval id")
    return state["evals"][eval_id]


@app.get("/preferences")
def preferences():
    current = {p: memory.get_preferences(p) for p in dict.fromkeys(memory.people() + state["diners"])}
    return {"backend": memory.BACKEND, "diners": state["diners"], "current": current, "history": memory.history()}


@app.post("/reset")
def reset():
    """Clear the cooking session (recipe, chat, diners). Long-term memory is kept."""
    state.update(recipe=None, current_step=0, history=[], diners=["me"], mode="normal", pending_image=None)
    return {"ok": True}


@app.post("/memory/reset")
def memory_reset():
    """Forget all preferences (for rehearsing the first-time onboarding demo)."""
    memory.reset_all()
    state.update(mode="normal", pending_image=None, diners=["me"])
    return {"ok": True}


@app.get("/state")
def get_state():
    return state


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
