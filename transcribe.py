#!/usr/bin/env python3

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import feedparser
import requests
from bs4 import BeautifulSoup
from faster_whisper import WhisperModel

import os

os.add_dll_directory(
    r"C:\Users\axel\AppData\Local\Programs\Python\Python313\Lib\site-packages\nvidia\cublas\bin"
)
os.add_dll_directory(
    r"C:\Users\axel\AppData\Local\Programs\Python\Python313\Lib\site-packages\nvidia\cudnn\bin"
)

# ============================================================
# Configuration
# ============================================================

FEED_URL = "https://feed.podbean.com/maratonlabbet/feed.xml"

BASE_DIR = Path(__file__).resolve().parent

AUDIO_DIR = BASE_DIR / "audio"
EPISODES_DIR = BASE_DIR / "episodes"
STATE_FILE = BASE_DIR / "state.json"

DEFAULT_MODEL = "medium"

LANGUAGE = "sv"

# Number of transcript segments between automatically generated
# chapter headings.
CHAPTER_SEGMENTS = 25

# Don't create a chapter more frequently than this.
MIN_CHAPTER_MINUTES = 4

REQUEST_TIMEOUT = 120


# ============================================================
# Helpers
# ============================================================

def slugify(text: str) -> str:
    """
    Turn an episode title into a reasonably safe filename.
    """

    text = html.unescape(text)

    text = re.sub(r"<[^>]+>", "", text)

    # Swedish characters are fine on modern filesystems/GitHub,
    # but replacing them makes URLs and filenames simpler.
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
    """
    Convert an RSS HTML description into plain text.
    """

    if not text:
        return ""

    soup = BeautifulSoup(text, "html.parser")

    return soup.get_text(" ", strip=True)


def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to HH:MM:SS or MM:SS.
    """

    seconds = max(0, int(seconds))

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    return f"{minutes:02d}:{secs:02d}"


def normalize_text(text: str) -> str:
    """
    Light cleanup of Whisper output.
    """

    text = text.strip()

    # Collapse excessive whitespace.
    text = re.sub(r"\s+", " ", text)

    # Remove accidental spaces before punctuation.
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)

    return text


def load_state() -> dict[str, Any]:
    if not STATE_FILE.exists():
        return {
            "episodes": {}
        }

    try:
        return json.loads(
            STATE_FILE.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError:
        print(
            "WARNING: state.json verkar vara trasig. "
            "Börjar om utan tidigare state.",
            file=sys.stderr,
        )

        return {
            "episodes": {}
        }


def save_state(state: dict[str, Any]) -> None:
    temporary = STATE_FILE.with_suffix(".tmp")

    temporary.write_text(
        json.dumps(
            state,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    temporary.replace(STATE_FILE)


# ============================================================
# RSS
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
    #
    # Usually published_parsed is available, but keep the original
    # feed order as fallback.
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
# Download
# ============================================================

def download_file(
    url: str,
    destination: Path,
) -> None:

    if destination.exists():
        print(
            f"  Ljud finns redan: "
            f"{destination.name}"
        )
        return

    temporary = destination.with_suffix(
        destination.suffix + ".part"
    )

    print(f"  Laddar ner: {url}")

    headers = {
        "User-Agent": (
            "Maratonlabbet-transcriber/1.0 "
            "(personal archival project)"
        )
    }

    with requests.get(
        url,
        headers=headers,
        stream=True,
        timeout=REQUEST_TIMEOUT,
    ) as response:

        response.raise_for_status()

        total = int(
            response.headers.get(
                "content-length",
                0,
            )
        )

        downloaded = 0
        last_print = time.monotonic()

        with open(temporary, "wb") as f:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):

                if not chunk:
                    continue

                f.write(chunk)

                downloaded += len(chunk)

                now = time.monotonic()

                if (
                    total
                    and now - last_print > 2
                ):
                    percent = (
                        downloaded / total * 100
                    )

                    print(
                        f"    {percent:5.1f}%",
                        end="\r",
                        flush=True,
                    )

                    last_print = now

    print()

    temporary.replace(destination)

# ============================================================
# Transcription
# ============================================================

TRANSCRIBE_CHUNK_SECONDS = 5 * 60
TRANSCRIBE_OVERLAP_SECONDS = 5


def get_audio_duration(audio_file: Path) -> float:
    """Get audio duration using ffprobe."""

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


def transcribe(
    audio_file: Path,
    model: WhisperModel,
):
    print(
        f"  Transkriberar: "
        f"{audio_file.name}"
    )

    duration = get_audio_duration(audio_file)

    print(
        f"  Längd: {duration / 60:.1f} minuter"
    )

    print(
        f"  Chunkstorlek: "
        f"{TRANSCRIBE_CHUNK_SECONDS / 60:.0f} minuter"
    )

    all_segments = []
    chunk_start = 0.0
    first_chunk = True

    while chunk_start < duration:

        # Give every chunk a small overlap with the previous one.
        if first_chunk:
            decode_start = 0.0
        else:
            decode_start = max(
                0.0,
                chunk_start - TRANSCRIBE_OVERLAP_SECONDS,
            )

        chunk_end = min(
            chunk_start + TRANSCRIBE_CHUNK_SECONDS,
            duration,
        )

        print(
            f"  Transkriberar "
            f"{chunk_start / 60:.1f}–"
            f"{chunk_end / 60:.1f} min..."
        )

        segments, info = model.transcribe(
            str(audio_file),

            language=LANGUAGE,

            beam_size=5,

            vad_filter=True,

            word_timestamps=True,

            condition_on_previous_text=True,

            # Let Whisper only process this part of
            # the episode.
            clip_timestamps=(
                decode_start,
                chunk_end,
            ),
        )

        chunk_segments = list(segments)

        for segment in chunk_segments:

            # Ignore anything produced in the overlap
            # that belongs to the previous chunk.
            if (
                not first_chunk
                and segment.end <= (
                    chunk_start
                    + TRANSCRIBE_OVERLAP_SECONDS
                )
            ):
                continue

            all_segments.append(segment)

        first_chunk = False
        chunk_start = chunk_end

    return all_segments, info

# ============================================================
# Chapter generation
# ============================================================

def choose_chapter_indices(
    segments,
) -> set[int]:

    if not segments:
        return set()

    chapter_indices = {0}

    last_chapter_time = segments[0].start

    for i, segment in enumerate(segments):

        elapsed = (
            segment.start
            - last_chapter_time
        )

        if (
            i > 0
            and i % CHAPTER_SEGMENTS == 0
            and elapsed
            >= MIN_CHAPTER_MINUTES * 60
        ):
            chapter_indices.add(i)

            last_chapter_time = segment.start

    return chapter_indices


def guess_chapter_title(
    segment_text: str,
) -> str:
    """
    We deliberately do NOT use an LLM here.

    Instead, make a readable generic chapter title.
    This keeps the whole pipeline local and deterministic.
    """

    text = normalize_text(segment_text)

    # Keep it short enough for a heading.
    words = text.split()

    if len(words) > 10:
        text = " ".join(words[:10]) + "…"

    return text


# ============================================================
# Markdown
# ============================================================

def write_markdown(
    episode: Episode,
    segments,
    output_file: Path,
    model_name: str,
) -> None:

    segments = [
        s
        for s in segments
        if normalize_text(s.text)
    ]

    chapter_indices = choose_chapter_indices(
        segments
    )

    lines: list[str] = []

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    lines.append(
        f"# Avsnitt {episode.index}: "
        f"{episode.title}"
    )

    lines.append("")

    if episode.published:
        lines.append(
            f"**Publicerat:** {episode.published}"
        )

    if episode.link:
        lines.append(
            f"**Original:** {episode.link}"
        )

    lines.append("")

    # --------------------------------------------------------
    # Description
    # --------------------------------------------------------

    if episode.description:

        lines.extend(
            [
                "## Om avsnittet",
                "",
                episode.description,
                "",
            ]
        )

    # --------------------------------------------------------
    # Table of contents
    # --------------------------------------------------------

    if chapter_indices:

        lines.extend(
            [
                "## Kapitel",
                "",
            ]
        )

        for index in sorted(
            chapter_indices
        ):

            segment = segments[index]

            timestamp = format_timestamp(
                segment.start
            )

            title = guess_chapter_title(
                segment.text
            )

            anchor = slugify(title)

            lines.append(
                f"- [{timestamp} – {title}]"
                f"(#{anchor})"
            )

        lines.append("")

    # --------------------------------------------------------
    # Transcript
    # --------------------------------------------------------

    lines.extend(
        [
            "## Transkription",
            "",
        ]
    )

    for i, segment in enumerate(
        segments
    ):

        text = normalize_text(
            segment.text
        )

        if not text:
            continue

        timestamp = format_timestamp(
            segment.start
        )

        # Chapter heading.
        if i in chapter_indices:

            chapter_title = guess_chapter_title(
                text
            )

            lines.extend(
                [
                    f"### {chapter_title}",
                    "",
                ]
            )

        lines.append(
            f"**[{timestamp}]** {text}"
        )

        lines.append("")

    # --------------------------------------------------------
    # Footer
    # --------------------------------------------------------

    lines.extend(
        [
            "---",
            "",
            f"*Transkriberad lokalt med "
            f"`faster-whisper` ({model_name}).*",
            "",
        ]
    )

    output_file.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Ladda ner och transkribera "
            "Maratonlabbet."
        )
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Whisper-modell, t.ex. "
            "small, medium, large-v3"
        ),
    )

    parser.add_argument(
        "--from",
        dest="from_episode",
        type=int,
        default=None,
        help=(
            "Börja från detta avsnittsnummer."
        ),
    )

    parser.add_argument(
        "--only",
        type=int,
        default=None,
        help=(
            "Bearbeta endast detta avsnittsnummer."
        ),
    )

    parser.add_argument(
        "--redownload",
        action="store_true",
        help=(
            "Ladda ner ljudet igen även om "
            "det redan finns."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Transkribera om även om Markdown "
            "redan finns."
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

    if not episodes:
        return

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

    print(
        f"\nLaddar Whisper-modell: "
        f"{args.model}"
    )

    print(
        "Detta kan ta en stund första gången."
    )

    model = WhisperModel(
        args.model,
        #device="cpu", compute_type="int8",
        device="cuda", compute_type="float16"
    )

    # --------------------------------------------------------
    # State
    # --------------------------------------------------------

    state = load_state()

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
            / f"{episode.index:03d}-{slug}.md"
        )

        # ----------------------------------------------------
        # Already done?
        # ----------------------------------------------------

        if (
            output_file.exists()
            and not args.force
        ):

            print(
                "  ✓ Redan transkriberat."
            )

            # Make sure state knows about it.
            state["episodes"][
                episode.id
            ] = {
                "index": episode.index,
                "title": episode.title,
                "audio_url": episode.audio_url,
                "markdown": str(
                    output_file.relative_to(
                        BASE_DIR
                    )
                ),
            }

            save_state(state)

            continue

        # ----------------------------------------------------
        # Download
        # ----------------------------------------------------

        if (
            args.redownload
            and audio_file.exists()
        ):
            print(
                "  Tar bort befintlig ljudfil."
            )

            audio_file.unlink()

        download_file(
            episode.audio_url,
            audio_file,
        )

        # ----------------------------------------------------
        # Transcribe
        # ----------------------------------------------------

        segments, info = transcribe(
            audio_file,
            model,
        )

        print(
            f"  Språk: {info.language} "
            f"({info.language_probability:.1%})"
        )

        print(
            f"  Segment: {len(segments)}"
        )

        # ----------------------------------------------------
        # Markdown
        # ----------------------------------------------------

        write_markdown(
            episode=episode,
            segments=segments,
            output_file=output_file,
            model_name=args.model,
        )

        print(
            f"  ✓ Skrev {output_file}"
        )

        # ----------------------------------------------------
        # State
        # ----------------------------------------------------

        state["episodes"][
            episode.id
        ] = {
            "index": episode.index,
            "title": episode.title,
            "audio_url": episode.audio_url,
            "markdown": str(
                output_file.relative_to(
                    BASE_DIR
                )
            ),
        }

        save_state(state)

    print()
    print("=" * 70)
    print("KLART")
    print("=" * 70)


if __name__ == "__main__":
    main()
