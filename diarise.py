#!/usr/bin/env python3

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import feedparser
import numpy as np
import torch
from pyannote.audio import Pipeline
from bs4 import BeautifulSoup


# ============================================================
# Configuration
# ============================================================

FEED_URL = "https://feed.podbean.com/maratonlabbet/feed.xml"

BASE_DIR = Path(__file__).resolve().parent

AUDIO_DIR = BASE_DIR / "audio"
EPISODES_DIR = BASE_DIR / "episodes"

DEFAULT_CHUNK_SECONDS = 2 * 60
SAMPLE_RATE = 16000


# ============================================================
# Helpers
# ============================================================

def slugify(text: str) -> str:
    """Turn an episode title into a reasonably safe filename."""

    text = html.unescape(text)

    text = re.sub(r"<[^>]+>", "", text)

    replacements = {
        "å": "a",
        "ä": "a",
        "ö": "o",
        "Å": "A",
        "Ä": "A",
        "Ö": "O",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = text.lower()

    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "-", text)

    text = text.strip("-")

    return text[:100]


def clean_html(text: str) -> str:
    """Convert an RSS HTML description into plain text."""

    if not text:
        return ""

    soup = BeautifulSoup(text, "html.parser")

    return soup.get_text(" ", strip=True)


# ============================================================
# Episode
# ============================================================

@dataclass
class Episode:
    id: str
    title: str
    published: str
    published_parsed: Any
    audio_url: str
    description: str
    link: str
    index: int = 0


def read_feed() -> list[Episode]:

    print("Läser RSS-feed:")
    print(f"  {FEED_URL}")

    feed = feedparser.parse(FEED_URL)

    if feed.bozo:
        print(
            f"WARNING: RSS-parsern rapporterar: "
            f"{feed.bozo_exception}"
        )

    if not feed.entries:
        raise RuntimeError(
            "Hittade inga avsnitt i RSS-feeden."
        )

    episodes: list[Episode] = []

    for entry in feed.entries:

        enclosures = entry.get("enclosures", [])

        if not enclosures:
            print(
                f"Hoppar över utan ljudfil: "
                f"{entry.get('title', 'Unknown')}"
            )
            continue

        audio_url = enclosures[0].get("href")

        if not audio_url:
            continue

        episode_id = (
            entry.get("id")
            or entry.get("guid")
            or entry.get("link")
            or audio_url
        )

        episodes.append(
            Episode(
                id=str(episode_id),
                title=entry.get(
                    "title",
                    "Okänt avsnitt",
                ),
                published=entry.get(
                    "published",
                    "",
                ),
                published_parsed=entry.get(
                    "published_parsed"
                ),
                audio_url=audio_url,
                description=clean_html(
                    entry.get(
                        "summary",
                        entry.get(
                            "description",
                            "",
                        ),
                    )
                ),
                link=entry.get("link", ""),
            )
        )

    # Oldest -> newest.
    episodes.sort(
        key=lambda e: (
            e.published_parsed
            if e.published_parsed
            else ()
        )
    )

    for index, episode in enumerate(
        episodes,
        start=1,
    ):
        episode.index = index

    return episodes


# ============================================================
# Audio
# ============================================================

def get_duration(audio_file: Path) -> float:

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

    return float(result.stdout.strip())


def decode_chunk(
    audio_file: Path,
    start: float,
    duration: float,
) -> torch.Tensor:

    result = subprocess.run(
        [
            "ffmpeg",
            "-ss", str(start),
            "-i", str(audio_file),
            "-t", str(duration),
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

    waveform = torch.from_numpy(
        audio
    ).unsqueeze(0)

    return waveform


# ============================================================
# Diarization
# ============================================================

def diarize(
    audio_file: Path,
    pipeline: Pipeline,
    chunk_seconds: int,
) -> list[dict[str, Any]]:

    duration = get_duration(audio_file)

    print(
        f"  Ljud: {duration / 60:.1f} minuter"
    )

    print(
        f"  Chunkstorlek: "
        f"{chunk_seconds / 60:.1f} minuter"
    )

    all_segments: list[dict[str, Any]] = []

    chunk_start = 0.0

    while chunk_start < duration:

        chunk_end = min(
            chunk_start + chunk_seconds,
            duration,
        )

        print(
            f"  Diariserar "
            f"{chunk_start / 60:.1f}–"
            f"{chunk_end / 60:.1f} min..."
        )

        waveform = decode_chunk(
            audio_file,
            chunk_start,
            chunk_end - chunk_start,
        )

        output = pipeline(
            {
                "waveform": waveform,
                "sample_rate": SAMPLE_RATE,
            }
        )

        for turn, speaker in (
            output.speaker_diarization
        ):
            all_segments.append(
                {
                    "start": chunk_start + turn.start,
                    "end": chunk_start + turn.end,
                    "speaker": speaker,
                }
            )

        del waveform
        del output

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        chunk_start = chunk_end

    return all_segments


# ============================================================
# Save
# ============================================================

def write_diarization(
    output_file: Path,
    segments: list[dict[str, Any]],
) -> None:

    temporary = output_file.with_suffix(
        output_file.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            segments,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    temporary.replace(output_file)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Diarisera Maratonlabbet-avsnitt."
        )
    )

    parser.add_argument(
        "--from",
        dest="from_episode",
        type=int,
        default=None,
        help="Börja från detta avsnittsnummer.",
    )

    parser.add_argument(
        "--only",
        type=int,
        default=None,
        help="Bearbeta endast detta avsnittsnummer.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Diarisera om även om JSON redan finns."
        ),
    )

    parser.add_argument(
        "--chunk",
        type=int,
        default=DEFAULT_CHUNK_SECONDS,
        help=(
            "Chunkstorlek i sekunder "
            "(default: 120)."
        ),
    )

    args = parser.parse_args()

    AUDIO_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    EPISODES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # RSS
    # --------------------------------------------------------

    episodes = read_feed()

    print(
        f"\nHittade {len(episodes)} avsnitt."
    )

    # --------------------------------------------------------
    # Filtering
    # --------------------------------------------------------

    if args.only is not None:

        episodes = [
            e
            for e in episodes
            if e.index == args.only
        ]

    elif args.from_episode is not None:

        episodes = [
            e
            for e in episodes
            if e.index >= args.from_episode
        ]

    if not episodes:

        print(
            "Inga avsnitt matchar filtret."
        )

        return

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print("Laddar diariseringsmodell...")

    token_file = BASE_DIR / ".hftoken"

    if not token_file.exists():
        print(
            "Hittar inte .hftoken",
            file=sys.stderr,
        )
        sys.exit(1)

    token = token_file.read_text(
        encoding="utf-8"
    ).strip()

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=token,
    )

    # Keep the pipeline on CPU.
    #
    # This is intentional: the GTX 1660 was running
    # out of CUDA memory during pyannote segmentation.
    pipeline.to(torch.device("cpu"))

    # Explicitly keep the memory-hungry segmentation
    # model on CPU and use a tiny batch.
    pipeline._segmentation.to(
        torch.device("cpu")
    )

    pipeline._segmentation.batch_size = 1

    print("Modell laddad.")

    # --------------------------------------------------------
    # Episodes
    # --------------------------------------------------------

    for episode in episodes:

        print()
        print("=" * 70)
        print(
            f"AVSNITT {episode.index}: "
            f"{episode.title}"
        )
        print("=" * 70)

        slug = slugify(
            episode.title
        )

        audio_file = (
            AUDIO_DIR
            / f"{episode.index:03d}-{slug}.mp3"
        )

        output_file = (
            EPISODES_DIR
            / f"{episode.index:03d}-{slug}.diarization.json"
        )

        # ----------------------------------------------------
        # Check audio
        # ----------------------------------------------------

        if not audio_file.exists():

            print(
                f"  ERROR: Ljudfil saknas:"
                f"\n  {audio_file}"
            )

            print(
                "  Kör transcribe.py först "
                "för att ladda ner ljudet."
            )

            continue

        # ----------------------------------------------------
        # Already done?
        # ----------------------------------------------------

        if (
            output_file.exists()
            and not args.force
        ):

            print(
                "  ✓ Redan diariserat."
            )

            continue

        # ----------------------------------------------------
        # Diarize
        # ----------------------------------------------------

        segments = diarize(
            audio_file=audio_file,
            pipeline=pipeline,
            chunk_seconds=args.chunk,
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        write_diarization(
            output_file,
            segments,
        )

        print(
            f"  ✓ Skrev {output_file}"
        )

        print(
            f"  ✓ {len(segments)} "
            f"speakersegment"
        )

    print()
    print("=" * 70)
    print("KLART")
    print("=" * 70)


if __name__ == "__main__":
    main()