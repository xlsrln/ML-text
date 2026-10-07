import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from pyannote.audio import Pipeline

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


CHUNK_SECONDS = 2 * 60
SAMPLE_RATE = 16000


if len(sys.argv) != 2:
    print("Användning: python diarise.py <ljudfil>")
    sys.exit(1)

audio_file = Path(sys.argv[1])

if not audio_file.exists():
    print(f"Hittar inte: {audio_file}")
    sys.exit(1)


token_file = Path(".hftoken")

if not token_file.exists():
    print("Hittar inte .hftoken")
    sys.exit(1)

token = token_file.read_text(encoding="utf-8").strip()


print("Laddar diariseringsmodell...")

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=token,
)

pipeline.to(torch.device("cpu"))

#pipeline.to(torch.device("cuda"))

# Run the memory-hungry segmentation model on CPU.
pipeline._segmentation.to(torch.device("cpu"))
pipeline._segmentation.batch_size = 1

print("Modell laddad.")


# Get duration using ffprobe
result = subprocess.run(
    [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_file),
    ],
    capture_output=True,
    text=True,
    check=True,
)

duration = float(result.stdout.strip())

print(f"Ljud: {duration / 60:.1f} minuter")
print(f"Chunkstorlek: {CHUNK_SECONDS / 60:.0f} minuter")
print()


all_segments = []

chunk_start = 0.0

while chunk_start < duration:

    chunk_end = min(chunk_start + CHUNK_SECONDS, duration)

    print(
        f"Diariserar {chunk_start / 60:.1f}–"
        f"{chunk_end / 60:.1f} min..."
    )

    # Decode only this chunk with FFmpeg
    result = subprocess.run(
        [
            "ffmpeg",
            "-ss", str(chunk_start),
            "-i", str(audio_file),
            "-t", str(chunk_end - chunk_start),
            "-f", "f32le",
            "-ac", "1",
            "-ar", str(SAMPLE_RATE),
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True,
    )

    audio = np.frombuffer(
        result.stdout,
        dtype=np.float32,
    ).copy()

    waveform = torch.from_numpy(audio).unsqueeze(0)

    # Run diarisation on this chunk
    output = pipeline(
        {
            "waveform": waveform,
            "sample_rate": SAMPLE_RATE,
        }
    )

    for turn, speaker in output.speaker_diarization:

        start = chunk_start + turn.start
        end = chunk_start + turn.end

        all_segments.append(
            (start, end, speaker)
        )

    # Explicitly release chunk data
    del waveform
    del audio
    del result
    del output

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    chunk_start = chunk_end


print()
print("Talare:")

for start, end, speaker in all_segments:
    print(
        f"{start:8.2f} - "
        f"{end:8.2f}  "
        f"{speaker}"
    )

