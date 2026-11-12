# run.py
import argparse
import glob
import json
import os
import re
import sys
import time

from dotenv import load_dotenv
load_dotenv()

for _var in ("GNANI_API_KEY", "GROQ_API_KEY"):
    if not os.environ.get(_var):
        sys.exit(f"ERROR: {_var} is missing. Copy .env.example to .env and fill it in.")

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from groq import Groq
from agent import Conversation, process_turn
from voice import speech_to_text, text_to_speech, STTAPIError, AudioError

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".aac", ".opus", ".mp4"}
FILE_TIME_BUDGET_SEC = 40.0
EMPTY_REPLY = "Sorry, I didn't catch that. Could you say it again?"
AUDIO_FAIL_REPLY = "Sorry, I'm having trouble hearing you right now. Could you try again?"
SYSTEM_FAIL_REPLY = "Sorry, something went wrong on my side. Please try again in a moment."


def natural_key(path: str):
    name = os.path.basename(path).lower()
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def session_for(base_name: str, use_conversation: bool) -> str:
    if use_conversation:
        m = re.match(r"^(.+?)_turn\d+", base_name, re.IGNORECASE)
        if m:
            return m.group(1)
    return base_name          # independent file


def speak(text: str, out_path: str):
    try:
        return text_to_speech(text, out_path)
    except Exception as e:    # never let TTS crash the batch
        return False, 0.0, str(e)[:200]


def process_batch(input_dir: str, output_dir: str, use_conversation: bool):
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(
        (p for p in glob.glob(os.path.join(input_dir, "*"))
         if os.path.isfile(p) and os.path.splitext(p)[1].lower() in AUDIO_EXTS),
        key=natural_key,
    )
    if not files:
        print(f"No audio files found in {input_dir}")
        return

    client = Groq(timeout=20, max_retries=1)
    results_log = os.path.join(output_dir, "results.jsonl")
    open(results_log, "w").close()            # fresh log for every run
    sessions = {}
    print(f"Found {len(files)} files. Mode: "
          f"{'Conversation grouping' if use_conversation else 'Independent files'}")

    for file_path in files:
        filename = os.path.basename(file_path)
        base_name = os.path.splitext(filename)[0]
        out_audio = os.path.join(output_dir, f"{base_name}_reply.wav")
        session_id = session_for(base_name, use_conversation)
        if session_id not in sessions:
            sessions[session_id] = Conversation(session_id=session_id)
        convo = sessions[session_id]

        start = time.time()
        transcript, reply_text, error_flag = "", "", None
        timings, meta = {}, {}

        try:
            transcript, ff_sec, stt_sec = speech_to_text(file_path)
            timings["ffmpeg_convert"], timings["stt"] = ff_sec, stt_sec
            if not transcript:
                reply_text = EMPTY_REPLY
            else:
                reply_text, agent_meta, agent_timings = process_turn(convo, transcript, client)
                meta.update(agent_meta or {})
                timings.update(agent_timings or {})
        except (STTAPIError, AudioError) as e:
            error_flag = f"{type(e).__name__}: {str(e)[:150]}"
            reply_text = AUDIO_FAIL_REPLY
        except Exception as e:
            error_flag = f"SYSTEM_ERROR: {type(e).__name__}: {str(e)[:150]}"
            reply_text = SYSTEM_FAIL_REPLY      # no false claims of a handover

        ok, tts_sec, tts_err = speak(reply_text, out_audio)   # ALWAYS attempt TTS
        timings["tts"] = tts_sec
        if not ok:
            msg = f"TTS_API_FAILURE: {tts_err}"
            error_flag = f"{error_flag} | {msg}" if error_flag else msg

        timings["total"] = round(time.time() - start, 2)
        entry = {
            "file": filename, "session_id": session_id, "transcript": transcript,
            "reply_text": reply_text, "reply_audio": out_audio if ok else None,
            "latency_sec": timings["total"], "over_budget": timings["total"] > FILE_TIME_BUDGET_SEC,
            "error": error_flag, "timings": timings, "metadata": meta,
        }
        with open(results_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        with open(os.path.join(output_dir, f"{base_name}_reply.txt"), "w", encoding="utf-8") as f:
            f.write(reply_text)

        print(f"[{timings['total']}s] {filename} -> {error_flag or 'SUCCESS'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZentroMart VoiceOps batch runner")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--conversation", action="store_true",
                        help="Group files named <id>_turn<N>.<ext> into one conversation")
    args = parser.parse_args()
    process_batch(args.input_dir, args.output_dir, args.conversation)