import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import requests


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
EPISODES_DIR = BASE_DIR / "episodes"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:4b"

# Small batches are much more reliable with gemma3:4b.
CORRECTION_LINES_PER_CHUNK = 15

# Chapter analysis window.
CHAPTER_WINDOW_SECONDS = 8 * 60

# Don't allow chapters closer together than this.
MIN_CHAPTER_SECONDS = 4 * 60

REQUEST_TIMEOUT = 600

# Number of excerpts used for speaker identification.
SPEAKER_EXCERPTS_PER_SPEAKER = 10


# ============================================================
# Ollama
# ============================================================

def ask_ollama(
    prompt: str,
    temperature: float = 0.0,
) -> str:

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not communicate with Ollama: {exc}"
        ) from exc

    data = response.json()

    if "response" not in data:
        raise RuntimeError(
            f"Unexpected Ollama response: {data}"
        )

    return data["response"].strip()


def clean_json_response(text: str) -> str:
    """
    Remove markdown fences and surrounding text where possible.
    """

    text = text.strip()

    if "```json" in text:
        text = text.split("```json", 1)[1]

        if "```" in text:
            text = text.split("```", 1)[0]

    elif "```" in text:
        text = text.split("```", 1)[1]

        if "```" in text:
            text = text.split("```", 1)[0]

    text = text.strip()

    # Try to extract the outermost JSON object.
    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        text = text[start:end + 1]

    return text.strip()


# ============================================================
# Timestamp helpers
# ============================================================

def mmss_to_seconds(value: str) -> float:

    parts = [int(x) for x in value.split(":")]

    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds

    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds

    raise ValueError(f"Invalid timestamp: {value}")


def seconds_to_mmss(seconds: float) -> str:

    seconds = max(0, int(round(seconds)))

    minutes = seconds // 60
    secs = seconds % 60

    return f"{minutes:02d}:{secs:02d}"


# ============================================================
# Episode files
# ============================================================

def find_episode_markdown(
    episode_number: int,
) -> Path:

    matches = sorted(
        EPISODES_DIR.glob(
            f"{episode_number:03d}-*.md"
        )
    )

    matches = [
        p for p in matches
        if not p.name.endswith(".processed.md")
    ]

    if not matches:
        raise FileNotFoundError(
            f"No Markdown found for episode "
            f"{episode_number:03d}"
        )

    return matches[0]


def find_diarization_file(
    markdown_file: Path,
) -> Path:

    return markdown_file.with_suffix(
        ".diarization.json"
    )


# ============================================================
# Transcript parsing
# ============================================================

TIMESTAMP_RE = re.compile(
    r"^\*\*\[(\d{2}:\d{2}(?::\d{2})?)\]\*\*\s*(.*)$"
)


def parse_transcript(
    markdown: str,
) -> list[dict[str, Any]]:

    transcript = []

    in_transcript = False

    for line in markdown.splitlines():

        if line.strip() == "## Transkription":
            in_transcript = True
            continue

        if not in_transcript:
            continue

        match = TIMESTAMP_RE.match(
            line.strip()
        )

        if not match:
            continue

        timestamp = match.group(1)
        text = match.group(2).strip()

        transcript.append(
            {
                "timestamp": timestamp,
                "seconds": mmss_to_seconds(timestamp),
                "text": text,
            }
        )

    return transcript


# ============================================================
# Diarization
# ============================================================

def load_diarization(
    diarization_file: Path,
) -> list[dict[str, Any]]:

    if not diarization_file.exists():

        print(
            f"WARNING: no diarization file: "
            f"{diarization_file}",
            file=sys.stderr,
        )

        return []

    with diarization_file.open(
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    if isinstance(data, dict):

        segments = data.get(
            "segments",
            [],
        )

    elif isinstance(data, list):

        segments = data

    else:

        raise ValueError(
            "Unexpected diarization format"
        )

    return segments


def speaker_for_interval(
    start: float,
    end: float,
    diarization: list[dict[str, Any]],
) -> str | None:

    best_speaker = None
    best_overlap = 0.0

    for segment in diarization:

        seg_start = float(
            segment["start"]
        )

        seg_end = float(
            segment["end"]
        )

        overlap_start = max(
            start,
            seg_start,
        )

        overlap_end = min(
            end,
            seg_end,
        )

        overlap = max(
            0.0,
            overlap_end - overlap_start,
        )

        if overlap > best_overlap:

            best_overlap = overlap

            best_speaker = segment.get(
                "speaker"
            )

    return best_speaker


def assign_speakers(
    transcript: list[dict[str, Any]],
    diarization: list[dict[str, Any]],
) -> None:

    for i, line in enumerate(transcript):

        start = line["seconds"]

        if i + 1 < len(transcript):
            end = transcript[i + 1]["seconds"]
        else:
            end = start + 10

        speaker = speaker_for_interval(
            start,
            end,
            diarization,
        )

        line["local_speaker"] = (
            speaker or "UNKNOWN"
        )


# ============================================================
# Speaker identification
# ============================================================

def build_speaker_excerpts(
    transcript: list[dict[str, Any]],
) -> str:

    speakers: dict[str, list] = {}

    for line in transcript:

        speaker = line.get(
            "local_speaker",
            "UNKNOWN",
        )

        speakers.setdefault(
            speaker,
            [],
        ).append(line)

    output = []

    for speaker, lines in sorted(
        speakers.items()
    ):

        output.append(
            f"\n--- {speaker} ---"
        )

        if len(lines) <= SPEAKER_EXCERPTS_PER_SPEAKER:

            selected = lines

        else:

            step = (
                len(lines)
                / SPEAKER_EXCERPTS_PER_SPEAKER
            )

            selected = [
                lines[
                    min(
                        int(i * step),
                        len(lines) - 1,
                    )
                ]
                for i in range(
                    SPEAKER_EXCERPTS_PER_SPEAKER
                )
            ]

        for line in selected:

            output.append(
                f"[{line['timestamp']}] "
                f"{speaker}: "
                f"{line['text']}"
            )

    return "\n".join(output)


def identify_speakers(
    transcript: list[dict[str, Any]],
) -> dict[str, str]:

    excerpts = build_speaker_excerpts(
        transcript
    )

    prompt = f"""
You are identifying speakers in a Swedish podcast.

The regular hosts are:

Johan Forsstedt
Erik Olofsson

There may also be guests.

IMPORTANT:

The audio was diarised independently in short chunks.

Therefore SPEAKER_00, SPEAKER_01 etc. are LOCAL labels.
A label can change meaning in different parts of the episode.

Do NOT assume that SPEAKER_00 is always one person.

Use the actual dialogue to identify people.

Look especially for:
- introductions
- people saying their own names
- people addressing each other
- interviewer/question patterns
- guest introductions
- recurring conversational roles

Possible identities:

Johan Forsstedt
Erik Olofsson
Guest
Unknown

Only identify a person when there is reasonable evidence.

Return ONLY JSON.

Example:

{{
  "assignments": [
    {{
      "speaker": "SPEAKER_00",
      "identity": "Johan Forsstedt"
    }}
  ]
}}

Transcript excerpts:

{excerpts}
"""

    print(
        "Identifying speakers with Ollama..."
    )

    response = ask_ollama(
        prompt,
        temperature=0.0,
    )

    response = clean_json_response(
        response
    )

    try:

        data = json.loads(
            response
        )

    except json.JSONDecodeError:

        print(
            "WARNING: speaker identification "
            "did not return valid JSON.",
            file=sys.stderr,
        )

        print(
            response,
            file=sys.stderr,
        )

        return {}

    result = {}

    for item in data.get(
        "assignments",
        [],
    ):

        speaker = item.get(
            "speaker"
        )

        identity = item.get(
            "identity"
        )

        if speaker and identity:

            result[speaker] = identity

    return result


# ============================================================
# Transcript correction
# ============================================================

def correct_transcript_batch(
    lines: list[dict[str, Any]],
) -> list[str]:

    numbered = "\n".join(
        f"{i + 1}. {line['text']}"
        for i, line in enumerate(lines)
    )

    prompt = f"""
Correct this Swedish podcast transcription.

Fix ONLY obvious speech-to-text errors.

Useful things to correct:

- Swedish words
- names
- places
- numbers
- running terminology
- marathon terminology
- training terminology
- obvious punctuation
- obvious Whisper mistakes

Do NOT:

- summarize
- rewrite
- shorten
- combine lines
- split lines
- add information
- remove information
- change the speaker's style

If a line is already correct, return it unchanged.

There are exactly {len(lines)} input lines.

Return exactly {len(lines)} output lines.

Return ONLY the corrected text.

Do not include:
- line numbers
- timestamps
- speaker names
- explanations
- markdown

INPUT:

{numbered}
"""

    response = ask_ollama(
        prompt,
        temperature=0.0,
    )

    corrected = [
        line.strip()
        for line in response.splitlines()
        if line.strip()
    ]

    if len(corrected) != len(lines):

        print(
            f"WARNING: Ollama returned "
            f"{len(corrected)} lines for "
            f"{len(lines)} inputs. "
            f"Keeping originals.",
            file=sys.stderr,
        )

        return [
            line["text"]
            for line in lines
        ]

    return corrected


def correct_transcript(
    transcript: list[dict[str, Any]],
    progress_file: Path,
) -> None:

    total = len(transcript)

    completed = 0

    if progress_file.exists():

        try:

            data = json.loads(
                progress_file.read_text(
                    encoding="utf-8"
                )
            )

            completed = int(
                data.get(
                    "completed",
                    0,
                )
            )

            if completed > total:
                completed = 0

            print(
                f"Resuming transcript correction "
                f"from line {completed + 1}."
            )

        except Exception:

            completed = 0

    for start in range(
        completed,
        total,
        CORRECTION_LINES_PER_CHUNK,
    ):

        end = min(
            start
            + CORRECTION_LINES_PER_CHUNK,
            total,
        )

        batch = transcript[
            start:end
        ]

        print(
            f"Correcting transcript lines "
            f"{start + 1}-{end}/{total}..."
        )

        corrected = correct_transcript_batch(
            batch
        )

        for line, text in zip(
            batch,
            corrected,
        ):

            line["text"] = text

        progress_file.write_text(
            json.dumps(
                {
                    "completed": end,
                    "transcript": transcript,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


# ============================================================
# Chapter candidates
# ============================================================

def transcript_windows(
    transcript: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:

    if not transcript:
        return []

    windows = []

    current = []
    window_start = transcript[0]["seconds"]

    for line in transcript:

        if (
            current
            and line["seconds"]
            - window_start
            >= CHAPTER_WINDOW_SECONDS
        ):

            windows.append(current)

            current = []

            window_start = line["seconds"]

        current.append(line)

    if current:
        windows.append(current)

    return windows


def analyse_chapter_window(
    lines: list[dict[str, Any]],
) -> dict[str, Any] | None:

    text = "\n".join(
        f"[{line['timestamp']}] "
        f"{line['text']}"
        for line in lines
    )

    start_timestamp = lines[0]["timestamp"]

    prompt = f"""
You are analysing one section of a Swedish running podcast.

Identify the MAIN subject discussed in this section.

Do not summarize the whole section.

Find the most useful chapter title.

The title should be short, specific and descriptive.

Examples:

"Planering av maratonträningen"
"Intervallträning inför maraton"
"Eriks mål för säsongen"
"Träningsmängd och återhämtning"

If this section is mostly continuation of another topic,
still describe what is actually discussed.

Return ONLY one line in this exact format:

TITLE: your chapter title

Section starts at:

{start_timestamp}

Transcript:

{text}
"""

    response = ask_ollama(
        prompt,
        temperature=0.1,
    )

    for line in response.splitlines():

        if line.upper().startswith(
            "TITLE:"
        ):

            title = line.split(
                ":",
                1,
            )[1].strip()

            if title:
                return {
                    "timestamp": start_timestamp,
                    "seconds": lines[0]["seconds"],
                    "title": title,
                }

    return None


def generate_chapter_candidates(
    transcript: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    windows = transcript_windows(
        transcript
    )

    candidates = []

    print(
        f"Generating chapter candidates "
        f"from {len(windows)} sections..."
    )

    for i, window in enumerate(
        windows,
        start=1,
    ):

        print(
            f"  Analysing section "
            f"{i}/{len(windows)} "
            f"({window[0]['timestamp']})..."
        )

        candidate = analyse_chapter_window(
            window
        )

        if candidate:
            candidates.append(
                candidate
            )

    return candidates


# ============================================================
# Chapter selection
# ============================================================

def select_chapters(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    if not candidates:
        return []

    selected = []

    for candidate in candidates:

        if not selected:

            selected.append(candidate)

            continue

        gap = (
            candidate["seconds"]
            - selected[-1]["seconds"]
        )

        if gap < MIN_CHAPTER_SECONDS:

            # Keep the candidate that looks more
            # specific rather than blindly adding both.
            continue

        selected.append(candidate)

    return selected


# ============================================================
# Topics and keywords
# ============================================================

def generate_topics_keywords(
    transcript: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:

    # Use representative lines instead of the whole episode.
    #
    # This keeps the prompt small enough for gemma3:4b.

    if len(transcript) <= 100:

        selected = transcript

    else:

        count = 100

        step = (
            len(transcript)
            / count
        )

        selected = [
            transcript[
                min(
                    int(i * step),
                    len(transcript) - 1,
                )
            ]
            for i in range(count)
        ]

    text = "\n".join(
        f"[{line['timestamp']}] "
        f"{line['text']}"
        for line in selected
    )

    prompt = f"""
Identify the main topics and useful keywords in this Swedish
running podcast transcript.

Only use information actually present in the text.

Return exactly this format:

TOPICS: topic 1 | topic 2 | topic 3
KEYWORDS: keyword 1 | keyword 2 | keyword 3

Give 5-10 topics.

Give 10-20 useful keywords.

Do not explain anything else.

Transcript:

{text}
"""

    print(
        "Generating topics and keywords..."
    )

    response = ask_ollama(
        prompt,
        temperature=0.1,
    )

    topics = []
    keywords = []

    for line in response.splitlines():

        upper = line.upper()

        if upper.startswith(
            "TOPICS:"
        ):

            value = line.split(
                ":",
                1,
            )[1]

            topics = [
                x.strip()
                for x in value.split("|")
                if x.strip()
            ]

        elif upper.startswith(
            "KEYWORDS:"
        ):

            value = line.split(
                ":",
                1,
            )[1]

            keywords = [
                x.strip()
                for x in value.split("|")
                if x.strip()
            ]

    return topics, keywords


# ============================================================
# Markdown
# ============================================================

def markdown_anchor(
    text: str,
) -> str:

    text = text.lower()

    replacements = {
        "å": "a",
        "ä": "a",
        "ö": "o",
    }

    for old, new in replacements.items():
        text = text.replace(
            old,
            new,
        )

    text = re.sub(
        r"[^a-z0-9\s-]",
        "",
        text,
    )

    text = re.sub(
        r"\s+",
        "-",
        text.strip(),
    )

    return text


def build_markdown(
    original: str,
    transcript: list[dict[str, Any]],
    chapters: list[dict[str, Any]],
    topics: list[str],
    keywords: list[str],
) -> str:

    original_lines = original.splitlines()

    prefix = []

    for line in original_lines:

        if line.strip() == "## Kapitel":
            break

        prefix.append(line)

    output = list(prefix)

    # --------------------------------------------------------
    # Topics
    # --------------------------------------------------------

    if topics:

        output.append("")

        output.append(
            "**Ämnen:** "
            + ", ".join(topics)
        )

    if keywords:

        output.append(
            "**Nyckelord:** "
            + ", ".join(keywords)
        )

    # --------------------------------------------------------
    # Chapters
    # --------------------------------------------------------

    output.append("")
    output.append("## Kapitel")
    output.append("")

    chapter_lookup = {
        c["timestamp"]: c
        for c in chapters
    }

    for chapter in chapters:

        timestamp = chapter[
            "timestamp"
        ]

        title = chapter[
            "title"
        ]

        output.append(
            f"- [{timestamp} – {title}]"
            f"(#{markdown_anchor(title)})"
        )

    # --------------------------------------------------------
    # Transcript
    # --------------------------------------------------------

    output.append("")
    output.append("## Transkription")
    output.append("")

    for line in transcript:

        timestamp = line[
            "timestamp"
        ]

        chapter = chapter_lookup.get(
            timestamp
        )

        if chapter:

            output.append("")

            output.append(
                f"### {chapter['title']}"
            )

            output.append("")

        identity = line.get(
            "identity"
        )

        if not identity:

            identity = "Unknown"

        output.append(
            f"**[{timestamp}]** "
            f"{identity}: "
            f"{line['text']}"
        )

    output.append("")

    return "\n".join(output)


# ============================================================
# Episode processing
# ============================================================

def process_episode(
    episode_number: int,
    force: bool = False,
) -> None:

    markdown_file = find_episode_markdown(
        episode_number
    )

    diarization_file = find_diarization_file(
        markdown_file
    )

    output_file = markdown_file.with_name(
        markdown_file.stem
        + ".processed.md"
    )

    metadata_file = markdown_file.with_name(
        markdown_file.stem
        + ".metadata.json"
    )

    progress_file = markdown_file.with_name(
        markdown_file.stem
        + ".postprocess-progress.json"
    )

    if output_file.exists() and not force:

        print(
            f"Skipping episode {episode_number}: "
            f"{output_file.name} already exists."
        )

        return

    print()
    print("=" * 70)
    print(
        f"Episode {episode_number}: "
        f"{markdown_file.name}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load transcript
    # --------------------------------------------------------

    markdown = markdown_file.read_text(
        encoding="utf-8"
    )

    transcript = parse_transcript(
        markdown
    )

    if not transcript:

        raise RuntimeError(
            "No transcript lines found."
        )

    print(
        f"Loaded {len(transcript)} "
        f"transcript lines."
    )

    # --------------------------------------------------------
    # Diarization
    # --------------------------------------------------------

    diarization = load_diarization(
        diarization_file
    )

    print(
        f"Loaded {len(diarization)} "
        f"diarization segments."
    )

    assign_speakers(
        transcript,
        diarization,
    )

    # --------------------------------------------------------
    # Speaker identification
    # --------------------------------------------------------

    assignments = identify_speakers(
        transcript
    )

    print()
    print("Speaker assignments:")

    for speaker, identity in sorted(
        assignments.items()
    ):

        print(
            f"  {speaker} -> {identity}"
        )

    for line in transcript:

        local = line.get(
            "local_speaker",
            "UNKNOWN",
        )

        line["identity"] = assignments.get(
            local,
            "Unknown",
        )

    # --------------------------------------------------------
    # Correction
    # --------------------------------------------------------

    print()

    correct_transcript(
        transcript,
        progress_file,
    )

    # --------------------------------------------------------
    # Chapters
    # --------------------------------------------------------

    candidates = generate_chapter_candidates(
        transcript
    )

    chapters = select_chapters(
        candidates
    )

    # --------------------------------------------------------
    # Topics / keywords
    # --------------------------------------------------------

    topics, keywords = (
        generate_topics_keywords(
            transcript
        )
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    processed = build_markdown(
        markdown,
        transcript,
        chapters,
        topics,
        keywords,
    )

    output_file.write_text(
        processed,
        encoding="utf-8",
    )

    metadata = {
        "episode": episode_number,
        "ollama_model": OLLAMA_MODEL,
        "speaker_assignments": assignments,
        "chapters": chapters,
        "topics": topics,
        "keywords": keywords,
    }

    metadata_file.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Correction progress is no longer needed.
    if progress_file.exists():
        progress_file.unlink()

    print()
    print(
        f"Written: {output_file.name}"
    )

    print(
        f"Written: {metadata_file.name}"
    )


# ============================================================
# CLI
# ============================================================

def main() -> None:

    global OLLAMA_MODEL

    parser = argparse.ArgumentParser(
        description=(
            "Post-process Maratonlabbet "
            "transcripts using Ollama."
        )
    )

    parser.add_argument(
        "--only",
        type=int,
        help="Process only one episode.",
    )

    parser.add_argument(
        "--from",
        dest="from_episode",
        type=int,
        help="Process from this episode onward.",
    )

    parser.add_argument(
        "--model",
        default=OLLAMA_MODEL,
        help="Ollama model to use.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output.",
    )

    args = parser.parse_args()

    OLLAMA_MODEL = args.model

    # --------------------------------------------------------
    # Episodes
    # --------------------------------------------------------

    if args.only is not None:

        episodes = [
            args.only
        ]

    else:

        files = sorted(
            EPISODES_DIR.glob("*.md")
        )

        episodes = []

        for file in files:

            if file.name.endswith(
                ".processed.md"
            ):
                continue

            match = re.match(
                r"^(\d+)-",
                file.name,
            )

            if not match:
                continue

            number = int(
                match.group(1)
            )

            if (
                args.from_episode is None
                or number >= args.from_episode
            ):

                episodes.append(
                    number
                )

    if not episodes:

        print(
            "No episodes found."
        )

        return

    print(
        f"Using Ollama model: "
        f"{OLLAMA_MODEL}"
    )

    for episode_number in episodes:

        try:

            process_episode(
                episode_number,
                force=args.force,
            )

        except KeyboardInterrupt:

            print(
                "\nInterrupted."
            )

            sys.exit(1)

        except Exception as exc:

            print(
                f"\nERROR processing episode "
                f"{episode_number}: {exc}",
                file=sys.stderr,
            )

            # Continue to next episode.
            continue


if __name__ == "__main__":
    main()

