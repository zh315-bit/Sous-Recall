import os
import time

from elevenlabs.client import ElevenLabs
from opentelemetry import trace

VOICE_ID = os.getenv("VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")  # default: "George" premade voice
_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
tracer = trace.get_tracer("handsfree-chef.voice")


def transcribe(audio_bytes: bytes, filename: str = "audio.webm", content_type: str = "audio/webm") -> str:
    with tracer.start_as_current_span("elevenlabs_stt") as span:
        span.set_attribute("openinference.span.kind", "TOOL")
        span.set_attribute("model", "scribe_v1")
        span.set_attribute("audio.bytes", len(audio_bytes))
        t0 = time.perf_counter()
        resp = _client.speech_to_text.convert(
            model_id="scribe_v1",
            file=(filename, audio_bytes, content_type),
            tag_audio_events=False,
        )
        text = (resp.text or "").strip()
        span.set_attribute("latency_ms", round((time.perf_counter() - t0) * 1000))
        span.set_attribute("output.value", text)
        return text


def speak_stream(text: str):
    """Yield mp3 chunks as ElevenLabs produces them, so playback starts ~0.15s after the request."""
    # Started manually (not as the current span): StreamingResponse pulls chunks from different
    # threads, and attaching an OTel context across yields would fail to detach.
    span = tracer.start_span("elevenlabs_tts_stream")
    span.set_attribute("openinference.span.kind", "TOOL")
    span.set_attribute("model", "eleven_flash_v2_5")
    span.set_attribute("input.value", text)
    t0, first, total = time.perf_counter(), None, 0
    try:
        for chunk in _client.text_to_speech.stream(
            voice_id=VOICE_ID, text=text, model_id="eleven_flash_v2_5", output_format="mp3_44100_128",
        ):
            if first is None:
                first = round((time.perf_counter() - t0) * 1000)
                span.set_attribute("first_chunk_ms", first)
            total += len(chunk)
            yield chunk
    finally:
        span.set_attribute("latency_ms", round((time.perf_counter() - t0) * 1000))
        span.set_attribute("audio.bytes", total)
        span.end()


def speak(text: str) -> bytes:
    with tracer.start_as_current_span("elevenlabs_tts") as span:
        span.set_attribute("openinference.span.kind", "TOOL")
        span.set_attribute("model", "eleven_flash_v2_5")
        span.set_attribute("input.value", text)
        t0 = time.perf_counter()
        chunks = _client.text_to_speech.convert(
            voice_id=VOICE_ID,
            text=text,
            model_id="eleven_flash_v2_5",
            output_format="mp3_44100_128",
        )
        audio = b"".join(chunks)  # generator: latency includes full download
        span.set_attribute("latency_ms", round((time.perf_counter() - t0) * 1000))
        span.set_attribute("audio.bytes", len(audio))
        return audio
