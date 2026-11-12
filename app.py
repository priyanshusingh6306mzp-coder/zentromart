# app.py  (project root, next to run.py)
# Run:  python app.py   ->  http://127.0.0.1:8000
# A voice turn is split in three calls so the page can show progress:
#   /api/stt (audio -> transcript)  ->  /api/answer (text -> reply)  ->  /api/tts (reply -> audio)
import os
import re
import shutil
import sys
import tempfile
import time
import uuid

from dotenv import load_dotenv
load_dotenv()

for _var in ("GNANI_API_KEY", "GROQ_API_KEY"):
    if not os.environ.get(_var):
        sys.exit(f"ERROR: {_var} is missing. Copy .env.example to .env and fill it in.")

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE, "src"))
from config import check_dataset
check_dataset()

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from groq import Groq
from pydantic import BaseModel

from agent import Conversation, process_turn
from voice import AudioError, STTAPIError, speech_to_text, text_to_speech

OUT_DIR = os.path.join(BASE, "ui_output")
os.makedirs(OUT_DIR, exist_ok=True)

SYSTEM_FAIL_REPLY = "Sorry, something went wrong on my side. Please try again in a moment."
ALLOWED_SUFFIX = {".webm", ".mp4", ".m4a", ".ogg", ".wav", ".mp3", ".opus"}

app = FastAPI(title="ZentroMart VoiceOps")
app.mount("/audio", StaticFiles(directory=OUT_DIR), name="audio")
client = Groq(timeout=20, max_retries=1)
sessions = {}


def _sid(raw: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", raw or ""):
        raise HTTPException(400, "invalid session_id")
    return raw


def _convo(sid: str) -> Conversation:
    if sid not in sessions:
        sessions[sid] = Conversation(session_id=sid)
    return sessions[sid]


def _cleanup_audio(max_age_sec: int = 3600):
    now = time.time()
    for name in os.listdir(OUT_DIR):
        p = os.path.join(OUT_DIR, name)
        try:
            if now - os.path.getmtime(p) > max_age_sec:
                os.remove(p)
        except OSError:
            pass


def _order_card(convo: Conversation):
    if not (convo.is_verified and convo.order_facts):
        return None
    f = convo.order_facts
    fields = {k: v for k, v in f.items()
              if isinstance(v, (str, int, float, bool)) and v not in ("", None)}
    fields = dict(list(fields.items())[:12])
    if isinstance(f.get("items"), list):
        fields["items"] = len(f["items"])
    return {"order_id": convo.order_id, "fields": fields}


@app.get("/")
def index():
    return FileResponse(os.path.join(BASE, "static", "index.html"))


@app.post("/api/stt")
def stt_step(audio: UploadFile = File(...), session_id: str = Form(...)):
    _sid(session_id)
    data = audio.file.read()
    if not data:
        raise HTTPException(400, "empty audio")
    if len(data) > 15_000_000:
        raise HTTPException(413, "audio too large")
    suffix = os.path.splitext(audio.filename or "")[1].lower()
    suffix = suffix if suffix in ALLOWED_SUFFIX else ".webm"

    tmp = tempfile.mkdtemp(prefix="zm_up_")
    try:
        path = os.path.join(tmp, "in" + suffix)
        with open(path, "wb") as f:
            f.write(data)
        try:
            transcript, ff_sec, stt_sec = speech_to_text(path)
        except (STTAPIError, AudioError) as e:
            err = f"{type(e).__name__}: {str(e)[:150]}"
            print(f"[stt error] {err}", flush=True)
            return {"transcript": "", "failed": True, "error": err, "timings": {}}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"transcript": transcript, "failed": False, "error": None,
            "timings": {"ffmpeg_convert": ff_sec, "stt": stt_sec}}


class AnswerIn(BaseModel):
    session_id: str
    text: str


@app.post("/api/answer")
def answer_step(body: AnswerIn):
    sid = _sid(body.session_id)
    text = body.text.strip()[:500]
    if not text:
        raise HTTPException(400, "empty text")
    convo = _convo(sid)
    timings, meta, error = {}, {}, None
    try:
        reply, meta, t = process_turn(convo, text, client)
        timings.update(t or {})
    except Exception as e:
        error = f"SYSTEM_ERROR: {type(e).__name__}: {str(e)[:120]}"
        print(f"[answer error] {error}", flush=True)
        reply = SYSTEM_FAIL_REPLY
    tools = (meta or {}).get("tools", {})
    ticket = tools.get("create_ticket")
    if ticket and ticket.get("ticket_id"):
        ticket = {**ticket, "ticket_id": str(ticket["ticket_id"])[:12]}     # short id for display
    return {
        "reply_text": reply,
        "chunks": (meta or {}).get("chunks", []),
        "cited": (meta or {}).get("cited", []),
        "order": _order_card(convo),
        "ticket": ticket,
        "verified": convo.is_verified,
        "order_in_context": bool(convo.order_id),
        "handover_done": convo.handover_done,
        "timings": timings,
        "error": error,
    }


class TtsIn(BaseModel):
    session_id: str
    text: str


@app.post("/api/tts")
def tts_step(body: TtsIn):
    sid = _sid(body.session_id)
    text = body.text.strip()[:600]
    if not text:
        raise HTTPException(400, "empty text")
    _cleanup_audio()
    name = f"{sid}_{uuid.uuid4().hex[:8]}.wav"
    try:
        ok, sec, err = text_to_speech(text, os.path.join(OUT_DIR, name))
    except Exception as e:
        ok, sec, err = False, 0.0, str(e)[:150]
    if err:
        print(f"[tts error] {err}", flush=True)
    return {"audio_url": f"/audio/{name}" if ok else None, "timings": {"tts": sec}, "error": err}


class ResetIn(BaseModel):
    session_id: str


@app.post("/api/reset")
def reset(body: ResetIn):
    sessions.pop(_sid(body.session_id), None)
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)