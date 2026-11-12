# python stt_diag.py input/conv1_turn1.m4a
import os, subprocess, sys, time, wave
sys.path.append("src")
from dotenv import load_dotenv; load_dotenv()
from voice import stt, FFMPEG_EXE

src = sys.argv[1]
subprocess.run([FFMPEG_EXE, "-y", "-i", src, "-ar", "16000", "-ac", "1", "diag_full.wav"], capture_output=True)
trim = "silenceremove=start_periods=1:start_threshold=-45dB,areverse,silenceremove=start_periods=1:start_threshold=-45dB,areverse"
subprocess.run([FFMPEG_EXE, "-y", "-i", "diag_full.wav", "-af", trim, "diag_trim.wav"], capture_output=True)

for name in ("diag_full.wav", "diag_trim.wav"):
    with wave.open(name) as w:
        dur = w.getnframes() / w.getframerate()
    print(f"\n{name}: {dur:.1f}s of audio, {os.path.getsize(name)//1024} KB")
    for i in range(4):
        t = time.time()
        try:
            stt.transcribe(name, language_code="en-IN"); status = "OK"
        except Exception as e:
            status = "ERR " + str(e)[:70]
        print(f"  run {i+1}: {time.time()-t:5.1f}s  {status}")