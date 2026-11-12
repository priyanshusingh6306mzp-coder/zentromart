# src/voice.py
import glob
import os
import re
import shutil
import subprocess
import tempfile
import time
import wave
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from typing import List, Optional, Tuple

FFMPEG_EXE = shutil.which("ffmpeg")
if not FFMPEG_EXE:
    try:
        import imageio_ffmpeg
        FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        FFMPEG_EXE = "ffmpeg"

# NOTE: load_dotenv() and the GNANI_API_KEY check must run BEFORE this import.
from gnani.stt import GnaniSTTClient
from gnani.tts import GnaniTTSClient

stt = GnaniSTTClient()
tts = GnaniTTSClient()

TTS_VOICE = "Saanvi"          # move to config.py; a voice from the brief's list also works
_pool = ThreadPoolExecutor(max_workers=2)


class STTAPIError(Exception): ...
class AudioError(Exception): ...


def _call_with_timeout(fn, timeout, *args, **kwargs):
    fut = _pool.submit(fn, *args, **kwargs)
    try:
        return fut.result(timeout=timeout)
    except FutTimeout:
        raise TimeoutError(f"timed out after {timeout}s")


def _run_ffmpeg(args: List[str], timeout: int = 30) -> None:
    r = subprocess.run([FFMPEG_EXE, "-y", *args], capture_output=True, timeout=timeout)
    if r.returncode != 0:
        raise AudioError(r.stderr.decode(errors="ignore")[-300:])


_TRIM = ("silenceremove=start_periods=1:start_threshold=-45dB,areverse,"
         "silenceremove=start_periods=1:start_threshold=-45dB,areverse")

def prepare_audio(input_path: str, workdir: str) -> Tuple[List[str], float]:
    t0 = time.time()
    full = os.path.join(workdir, "full.wav")
    try:
        _run_ffmpeg(["-i", input_path, "-ar", "16000", "-ac", "1", "-af", _TRIM, full])
    except AudioError:
        _run_ffmpeg(["-i", input_path, "-ar", "16000", "-ac", "1", full])
    with wave.open(full, "rb") as wf:
        seconds = wf.getnframes() / float(wf.getframerate())
    if seconds < 0.3:
        return [], round(time.time() - t0, 2)          # only silence: nothing to transcribe
    if seconds <= 59:
        chunks = [full]
    else:
        _run_ffmpeg(["-i", full, "-f", "segment", "-segment_time", "55", "-c", "copy",
                     os.path.join(workdir, "chunk_%03d.wav")])
        chunks = sorted(glob.glob(os.path.join(workdir, "chunk_*.wav")))
        if not chunks:
            raise AudioError("chunking produced no files")
    return chunks, round(time.time() - t0, 2)


def normalize_asr_digits(text: str) -> str:
    """Collapse runs of 4+ spoken or spaced digits ('2 0 1 3 0 1', 'six nine five two')."""
    def _join(m):
        toks = re.findall(_TOKEN, m.group(0), flags=re.IGNORECASE)
        return "".join(_WORD2DIG.get(t.lower(), t) for t in toks)
    return _RUN.sub(_join, text).strip()

_WORD2DIG = {"zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4",
             "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}
_TOKEN = r"(?:\d+|zero|oh|one|two|three|four|five|six|seven|eight|nine)"
_RUN = re.compile(rf"\b{_TOKEN}(?:[\s,]+{_TOKEN}){{3,}}\b", re.IGNORECASE)


def normalize_asr_digits(text: str) -> str:
    """Collapse runs of 4+ spoken or spaced digits ('2 0 1 3 0 1', 'six nine five two')."""
    def _join(m):
        toks = re.findall(_TOKEN, m.group(0), flags=re.IGNORECASE)
        return "".join(_WORD2DIG.get(t.lower(), t) for t in toks)
    return _RUN.sub(_join, text).strip()
from concurrent.futures import wait, FIRST_COMPLETED

def _transcribe_one(path: str, timeout: int = 50) -> str:
    last = None
    for attempt in range(2):
        try:
            res = _call_with_timeout(stt.transcribe, timeout, path, language_code="en-IN")
            break
        except TimeoutError:
            raise STTAPIError(f"STT timed out after {timeout}s")   # no second request
        except Exception as e:
            last = e
            if attempt == 0:
                time.sleep(1)                                      # retry only API errors
    else:
        raise STTAPIError(str(last)[:200])
    txt = res.get("transcript", "") if isinstance(res, dict) else getattr(res, "transcript", "")
    return normalize_asr_digits(txt or "")

def speech_to_text(audio_path: str) -> Tuple[str, float, float]:
    """Returns (transcript, ffmpeg_sec, stt_sec). '' = genuinely empty speech.
    API/audio failures RAISE STTAPIError / AudioError."""
    wd = tempfile.mkdtemp(prefix="zm_stt_")
    try:
        chunks, ff_sec = prepare_audio(audio_path, wd)
        t0 = time.time()
        parts = [_transcribe_one(c) for c in chunks]
        stt_sec = round(time.time() - t0, 2)
    finally:
        shutil.rmtree(wd, ignore_errors=True)     # never raise on cleanup (Windows file locks)
    transcript = " ".join(p for p in parts if p).strip()
    return (transcript if len(transcript) >= 2 else ""), ff_sec, stt_sec

def clean_text_for_speech(text: str) -> str:
    text = re.sub(r"[*#_>`]", "", text)
    text = re.sub(r"(?<=\d)\s*[-–]\s*(?=\d)", " to ", text)          # 1-3 -> 1 to 3
    text = text.replace("-", " ")                                     # 7-day -> 7 day
    text = re.sub(r"₹\s*([\d,]+(?:\.\d+)?)", r"\1 rupees", text)
    text = re.sub(r"\bRs\.?\s*([\d,]+(?:\.\d+)?)", r"\1 rupees", text)
    text = re.sub(r"\d{6,}", lambda m: " ".join(m.group(0)), text)   # order IDs, pincodes, phones
    return re.sub(r"\s+", " ", text).strip()


def text_to_speech(text: str, output_path: str, timeout: int = 20) -> Tuple[bool, float, Optional[str]]:
    """Returns (success, tts_sec, error_message)."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    clean = clean_text_for_speech(text)
    t0, err = time.time(), None
    for attempt in range(2):
        try:
            _call_with_timeout(tts.synthesize, timeout, text=clean, voice=TTS_VOICE,
                               language="en-IN", model="timbre-v2.5", output_file=output_path)
            return True, round(time.time() - t0, 2), None
        except Exception as e:
            err = str(e)[:200]
            if attempt == 0:
                time.sleep(1)
    return False, round(time.time() - t0, 2), err