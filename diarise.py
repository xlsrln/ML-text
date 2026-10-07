import os
import sys
from pathlib import Path

from pyannote.audio import Pipeline

if len(sys.argv) != 2:
    print("Användning: python diarize.py <ljudfil>")
    sys.exit(1)

audio_file = Path(sys.argv[1])

if not audio_file.exists():
    print(f"Hittar inte: {audio_file}")
    sys.exit(1)

#token = os.environ.get("HF_TOKEN")
token_file = Path(".hftoken")

if not token_file.exists():
    print("Hittar inte .hftoken")
    sys.exit(1)

token = token_file.read_text(encoding="utf-8").strip()

if not token:
    print("HF_TOKEN saknas.")
    print('Kör först: export HF_TOKEN="din-token"')
    sys.exit(1)

print("Laddar diariseringsmodell...")

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=token,
)

print("Kör diarisation...")

output = pipeline(str(audio_file))

print("\nTalare:")
for turn, speaker in output.speaker_diarization:
    print(
        f"{turn.start:8.2f} - {turn.end:8.2f}  {speaker}"
    )
