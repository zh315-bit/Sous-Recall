import base64
import json
import os
import re

from anthropic import Anthropic

MODEL = os.getenv("MODEL", "claude-sonnet-5-5")  # recipe (vision) + onboarding
CHAT_MODEL = os.getenv("CHAT_MODEL", "claude-haiku-5-5")  # per-voice-turn controller: latency matters most
# Reads ANTHROPIC_API_KEY from env. User-level keys (sk-ant-usr-...) also need a workspace ID header.
_ws = os.getenv("ANTHROPIC_WORKSPACE_ID")
_client = Anthropic(default_headers={"anthropic-workspace-id": _ws} if _ws else None)

RECIPE_PROMPT = (
    "You are a home-cooking assistant. Look at the photo of the user's fridge or ingredients. "
    "Suggest ONE simple dish (under 30 minutes) that uses mainly what you can see; assume basic "
    "pantry items (salt, oil, pepper) are available. Respect these user preferences strictly: "
    "{preferences}. Return ONLY valid JSON, no markdown:\n"
    '{{"dish": str, "why": str (one sentence), "ingredients": [str], "steps": [str]}}\n'
    "Each step must be short enough to read aloud in under 15 seconds. Use 5-8 steps."
)


def _parse_json(text: str) -> dict:
    text = text.strip()
    # Strip ```json ... ``` fences if present
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fall back to the outermost {...} block
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            raise
        return json.loads(text[start : end + 1])


CHAT_PROMPT = (
    "You are Sous Recall, a friendly voice cooking assistant (a sous-chef that remembers what you can and can't eat). The user's hands are messy, so they "
    "talk to you instead of touching the screen. Your replies are spoken aloud: keep them under 2 "
    "sentences, conversational, no lists, no markdown, no emojis.\n"
    "Recipe: {recipe_json}\n"
    "Current step index (0-based): {current_step}\n"
    "Tonight's diners: {diners}\n"
    "Preferences of tonight's diners: {preferences}\n"
    "Everyone you remember (use this if the diners change this turn): {known}\n"
    "Decide what the user wants and return ONLY valid JSON:\n"
    '{{"reply": str, "action": "next" | "repeat" | "answer" | "none", '
    '"new_preference": {{"person": str, "fact": str}} or null, "diners": [str] or null}}\n'
    '- "next": they finished this step; reply by reading the NEXT step.\n'
    '- "repeat": read the current step again.\n'
    '- "answer": answer their question (amounts, timing, substitutions) in the context of the current step.\n'
    "- If Recipe is null there is no recipe yet: chat briefly, use action \"none\", and invite them to snap a photo of their fridge.\n"
    "- If someone states a lasting dietary preference, allergy, or dislike (e.g. 'I'm allergic to "
    "peanuts', 'Alex is vegetarian'), put it in new_preference: person is \"me\" for the speaker, "
    "otherwise that person's first name in lowercase; fact is a short fact without the name, like "
    "\"vegetarian\" or \"allergic to peanuts\". Acknowledge it in reply. Otherwise null.\n"
    "- If the user says who is eating this session (e.g. 'Tonight I'm cooking for me and Alex'), set "
    "diners to the full list of lowercase names, using \"me\" for the speaker. Otherwise null.\n"
    "- Never suggest anything that conflicts with any diner's preferences."
)


def _call_json(system: str, messages: list, required: tuple, model: str | None = None) -> dict:
    """Call the LLM, parse JSON, retry once with a correction message on failure."""
    last_err = None
    for _ in range(2):
        resp = _client.messages.create(model=model or MODEL, max_tokens=1500, system=system, messages=messages)
        text = "".join(b.text for b in resp.content if b.type == "text")
        try:
            data = _parse_json(text)
            missing = [k for k in required if not data.get(k)]
            if missing:
                raise ValueError(f"missing {missing}")
            return data
        except (json.JSONDecodeError, ValueError) as e:
            last_err = e
            messages = messages + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": "That was not valid JSON. Reply with ONLY the JSON object, nothing else."},
            ]
    raise RuntimeError(f"LLM returned invalid JSON after retry: {last_err}")


def chat(transcript: str, recipe: dict | None, current_step: int, preferences: str, history: list,
         diners: str = "me", known: str = "none") -> dict:
    system = CHAT_PROMPT.format(
        recipe_json=json.dumps(recipe), current_step=current_step, preferences=preferences,
        diners=diners, known=known,
    )
    messages = history + [{"role": "user", "content": transcript}]
    data = _call_json(system, messages, required=("reply",), model=CHAT_MODEL)
    if data.get("action") not in ("next", "repeat", "answer", "none"):
        data["action"] = "none"
    pref = data.get("new_preference")
    if isinstance(pref, str):  # tolerate the old flat shape
        pref = {"person": "me", "fact": pref}
    data["new_preference"] = pref if isinstance(pref, dict) and pref.get("fact") else None
    diners = data.get("diners")
    data["diners"] = [str(d) for d in diners] if isinstance(diners, list) and diners else None
    return data


SUPERSEDE_PROMPT = (
    "You maintain a list of a user's dietary preferences. A NEW fact was just stated. "
    "Return the indices of OLD facts that the new fact contradicts, updates, or restates "
    "(e.g. 'loves spicy food' is superseded by 'doesn't eat spicy food anymore'). "
    "Unrelated facts must NOT be included. Return ONLY JSON: {\"superseded\": [int]}"
)


def find_superseded(new_fact: str, old_facts: list[str]) -> list[int]:
    if not old_facts:
        return []
    listing = "\n".join(f"{i}: {f}" for i, f in enumerate(old_facts))
    messages = [{"role": "user", "content": f"OLD facts:\n{listing}\n\nNEW fact: {new_fact}"}]
    data = _call_json(SUPERSEDE_PROMPT, messages, required=(), model=CHAT_MODEL)
    return [i for i in data.get("superseded", []) if isinstance(i, int) and 0 <= i < len(old_facts)]


ONBOARD_PROMPT = (
    "The user was asked: 'Do you have any allergies or foods you avoid?' Extract every lasting dietary "
    "restriction from their answer as short facts. Return ONLY valid JSON:\n"
    '{"preferences": [{"fact": str, "type": "allergy" | "dislike" | "diet"}], "ack": str}\n'
    "- 'ack' is one short spoken sentence confirming what you'll avoid, e.g. 'Got it, I'll keep peanuts "
    "and shellfish out.'\n"
    "- If they have none ('no', 'nothing', 'I eat everything'), return an empty list and ack 'Great, no restrictions.'"
)


def onboard(transcript: str) -> dict:
    data = _call_json(ONBOARD_PROMPT, [{"role": "user", "content": transcript}], required=("ack",), model=CHAT_MODEL)
    prefs = data.get("preferences") if isinstance(data.get("preferences"), list) else []
    data["preferences"] = [
        {"fact": str(p["fact"]).strip(), "type": p.get("type") if p.get("type") in ("allergy", "dislike", "diet") else None}
        for p in prefs if isinstance(p, dict) and str(p.get("fact", "")).strip()
    ]
    return data


JUDGE_MODEL =os.getenv("JUDGE_MODEL", "claude-haiku-5-5")
JUDGE_PROMPT = (
    "Given these user preferences: {preferences}, does this content violate any of them? "
    'Answer JSON {{"violates": bool, "reason": str}}'
)


def check_preferences(preferences: str, content: str) -> dict:
    resp = _client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=800,
        system=JUDGE_PROMPT.format(preferences=preferences)
        + " Keep reason to one short sentence. Return ONLY the JSON.",
        messages=[{"role": "user", "content": content}],
    )
    return _parse_json("".join(b.text for b in resp.content if b.type == "text"))


def generate_recipe(image_bytes: bytes, media_type: str, preferences: str = "none") -> dict:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64.standard_b64encode(image_bytes).decode(),
                    },
                },
                {"type": "text", "text": "Here is my fridge. What should I cook?"},
            ],
        }
    ]
    system = RECIPE_PROMPT.format(preferences=preferences)
    return _call_json(system, messages, required=("dish", "steps"))
