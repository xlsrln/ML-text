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

REQUEST_TIMEOUT = 600

# Whisper correction.
# Smaller = safer for a 4B model.
CORRECTION_LINES_PER_CHUNK = 15

# Speaker identification.
# We want enough context to understand a conversation,
# but not so much that Gemma gets confused.
SPEAKER_LINES_PER_CHUNK = 30

# Chapter analysis.
TOPIC_WINDOW_SECONDS = 4 * 60
MIN_CHAPTER_SECONDS = 4 * 60

# Maximum amount of transcript sent for episode-level
# topic/keyword extraction.
TOPIC_SAMPLE_LINES = 150


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
            f"Kunde inte kommunicera med Ollama: {exc}"
        ) from exc

    data = response.json()

    if "response" not in data:
        raise RuntimeError(
            f"Oväntat svar från Ollama: {data}"
        )

    return data["response"].strip()


def clean_json_response(text: str) -> str:

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

    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        text = text[start:end + 1]

    return text.strip()


# ============================================================
# Time
# ============================================================

def mmss_to_seconds(value: str) -> float:

    parts = [
        int(x)
        for x in value.split(":")
    ]

    if len(parts) == 2:

        minutes, seconds = parts

        return (
            minutes * 60
            + seconds
        )

    if len(parts) == 3:

        hours, minutes, seconds = parts

        return (
            hours * 3600
            + minutes * 60
            + seconds
        )

    raise ValueError(
        f"Ogiltig tidsstämpel: {value}"
    )


def seconds_to_mmss(seconds: float) -> str:

    seconds = max(
        0,
        int(round(seconds)),
    )

    minutes = seconds // 60
    secs = seconds % 60

    return (
        f"{minutes:02d}:"
        f"{secs:02d}"
    )


# ============================================================
# Files
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
        p
        for p in matches
        if not p.name.endswith(
            ".processed.md"
        )
    ]

    if not matches:
        raise FileNotFoundError(
            f"Hittade ingen Markdown-fil för "
            f"avsnitt {episode_number:03d}"
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
                "seconds": mmss_to_seconds(
                    timestamp
                ),
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
            f"VARNING: ingen diariseringsfil hittades: "
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
            "Okänt format på diariseringsfilen."
        )

    return segments


def speaker_for_interval(
    start: float,
    end: float,
    diarization: list[dict[str, Any]],
) -> str | None:

    overlaps: dict[str, float] = {}

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

        if overlap <= 0:
            continue

        speaker = segment.get(
            "speaker"
        )

        if not speaker:
            continue

        overlaps[speaker] = (
            overlaps.get(
                speaker,
                0.0,
            )
            + overlap
        )

    if not overlaps:
        return None

    return max(
        overlaps,
        key=overlaps.get,
    )


def assign_local_speakers(
    transcript: list[dict[str, Any]],
    diarization: list[dict[str, Any]],
) -> None:

    for i, line in enumerate(
        transcript
    ):

        start = line["seconds"]

        if i + 1 < len(transcript):
            end = transcript[
                i + 1
            ]["seconds"]
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

def speaker_batch_text(
    lines: list[dict[str, Any]],
) -> str:

    output = []

    for i, line in enumerate(
        lines,
        start=1,
    ):

        speaker = line.get(
            "local_speaker",
            "UNKNOWN",
        )

        output.append(
            f"{i}. "
            f"[{line['timestamp']}] "
            f"{speaker}: "
            f"{line['text']}"
        )

    return "\n".join(output)


def parse_speaker_assignments(
    response: str,
    count: int,
) -> list[str] | None:

    response = clean_json_response(
        response
    )

    try:

        data = json.loads(
            response
        )

    except json.JSONDecodeError:

        return None

    assignments = data.get(
        "assignments"
    )

    if not isinstance(
        assignments,
        list,
    ):
        return None

    result = []

    for item in assignments:

        identity = item.get(
            "identity"
        )

        if identity not in {
            "Johan Forsstedt",
            "Erik Olofsson",
            "Guest",
            "Unknown",
        }:

            identity = "Unknown"

        result.append(identity)

    if len(result) != count:
        return None

    return result


def identify_speakers_batch(
    lines: list[dict[str, Any]],
) -> list[str]:

    text = speaker_batch_text(
        lines
    )

    prompt = f"""
Du ska identifiera vem som talar i en svensk podcast.

De två ordinarie programledarna är:

- Johan Forsstedt
- Erik Olofsson

Det kan också förekomma gäster.

Det är mycket viktigt att förstå följande:

Etiketterna SPEAKER_00, SPEAKER_01 osv kommer från
automatisk talardiarisering och är ENDAST lokala etiketter.

De är INTE globala identiteter.

SPEAKER_00 kan alltså vara Johan i en del av avsnittet
och Erik i en annan del.

Du får därför INTE göra en regel som exempelvis
"SPEAKER_00 = Johan".

Identifiera i stället vem som sannolikt talar på VARJE rad
utifrån dialogens innehåll och sammanhang.

Använd särskilt dessa ledtrådar:

- Om någon säger "Ja, Erik..." är personen som tilltalas Erik.
- Om någon säger "Ja, Johan..." är personen som tilltalas Johan.
- Om en person ställer en fråga till Erik är nästa svar ofta Erik.
- Om en person ställer en fråga till Johan är nästa svar ofta Johan.
- Om någon presenterar sig själv kan det identifiera personen.
- Använd kontinuitet i samtalet.
- Använd vem som brukar ställa frågor och vem som svarar.
- Använd grammatiken och sammanhanget.
- Om en rad är osäker ska du hellre välja Unknown än att hitta på.

VIKTIGT:

Det kan finnas fel i transkriberingen.
Försök förstå den avsedda betydelsen trots mindre Whisper-fel.

Möjliga identiteter är exakt:

"Johan Forsstedt"
"Erik Olofsson"
"Guest"
"Unknown"

Du ska returnera EN identitet per rad.

Returnera ENDAST giltig JSON i exakt detta format:

{{
  "assignments": [
    {{
      "identity": "Erik Olofsson"
    }},
    {{
      "identity": "Johan Forsstedt"
    }}
  ]
}}

Antalet assignments måste vara exakt {len(lines)}.

Ingen markdown.
Ingen förklaring.
Inga radnummer i JSON.

Här är dialogen:

{text}
"""

    response = ask_ollama(
        prompt,
        temperature=0.0,
    )

    assignments = parse_speaker_assignments(
        response,
        len(lines),
    )

    if assignments is None:

        print(
            "VARNING: kunde inte tolka "
            "talarsvaret. Använder Unknown "
            "för detta block.",
            file=sys.stderr,
        )

        return [
            "Unknown"
            for _ in lines
        ]

    return assignments


def identify_all_speakers(
    transcript: list[dict[str, Any]],
) -> None:

    total = len(transcript)

    for start in range(
        0,
        total,
        SPEAKER_LINES_PER_CHUNK,
    ):

        end = min(
            start
            + SPEAKER_LINES_PER_CHUNK,
            total,
        )

        batch = transcript[
            start:end
        ]

        print(
            f"Identifierar talare "
            f"{start + 1}-{end}/{total}..."
        )

        identities = identify_speakers_batch(
            batch
        )

        for line, identity in zip(
            batch,
            identities,
        ):

            line["identity"] = identity


# ============================================================
# Transcript correction
# ============================================================

def strip_unwanted_prefix(
    text: str,
) -> str:

    text = text.strip()

    # Ollama kan trots instruktionerna lägga till:
    # 1. text
    # 12. text
    text = re.sub(
        r"^\d+\.\s*",
        "",
        text,
    )

    # Ibland kan modellen sätta citattecken runt hela raden.
    if (
        len(text) >= 2
        and text[0] == '"'
        and text[-1] == '"'
    ):

        text = text[1:-1].strip()

    return text


def correct_transcript_batch(
    lines: list[dict[str, Any]],
) -> list[str]:

    input_text = "\n".join(
        line["text"]
        for line in lines
    )

    prompt = f"""
Du korrekturläser en svensk automatisk transkribering från
en podcast om löpning och maraton.

Din uppgift är mycket begränsad:

Rätta ENDAST uppenbara fel som sannolikt kommer från
tal-till-text-transkriberingen.

Du kan exempelvis rätta:

- uppenbart felhörda svenska ord
- uppenbart felstavade namn
- löpartermer
- maratintermer
- träningsbegrepp
- ortsnamn
- siffror när sammanhanget tydligt visar vad som avses
- uppenbara grammatiska fel som uppstår genom felskrivning

Var konservativ.

Om texten verkar rimlig ska du lämna den oförändrad.

Ändra INTE:

- talarens personliga språk
- ordval som bara låter lite talspråkliga
- innehållet
- betydelsen
- längden
- ordningen

Du får INTE:

- sammanfatta
- skriva om
- förbättra stilen
- lägga till information
- ta bort information
- slå ihop rader
- dela upp rader
- kommentera ändringarna

Det finns exakt {len(lines)} rader i indata.

Du måste returnera exakt {len(lines)} rader
i exakt samma ordning.

Returnera ENDAST de korrigerade raderna.

Skriv INTE:

- radnummer
- tidsstämplar
- talarnamn
- punktlistor
- kommentarer
- förklaringar
- markdown

INDATA:

{input_text}
"""

    response = ask_ollama(
        prompt,
        temperature=0.0,
    )

    corrected = [
        strip_unwanted_prefix(line)
        for line in response.splitlines()
        if line.strip()
    ]

    if len(corrected) != len(lines):

        print(
            f"VARNING: Ollama returnerade "
            f"{len(corrected)} rader för "
            f"{len(lines)} indata-rader. "
            f"Behåller originalet för blocket.",
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
    force: bool,
) -> None:

    total = len(transcript)

    completed = 0

    if force and progress_file.exists():

        progress_file.unlink()

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

            saved_transcript = data.get(
                "transcript"
            )

            if (
                isinstance(
                    saved_transcript,
                    list,
                )
                and len(saved_transcript)
                == total
            ):

                for i in range(
                    completed
                ):

                    if i < len(
                        saved_transcript
                    ):

                        transcript[i][
                            "text"
                        ] = saved_transcript[i].get(
                            "text",
                            transcript[i]["text"],
                        )

                print(
                    f"Fortsätter korrektur från "
                    f"rad {completed + 1}."
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
            f"Korrigerar transkript "
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
# Topic windows
# ============================================================

def make_topic_windows(
    transcript: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:

    if not transcript:
        return []

    windows = []

    current = []

    window_start = transcript[
        0
    ]["seconds"]

    for line in transcript:

        if (
            current
            and line["seconds"]
            - window_start
            >= TOPIC_WINDOW_SECONDS
        ):

            windows.append(current)

            current = []

            window_start = line[
                "seconds"
            ]

        current.append(line)

    if current:
        windows.append(current)

    return windows


# ============================================================
# Topic change detection
# ============================================================

def analyse_topic_window(
    lines: list[dict[str, Any]],
) -> dict[str, Any] | None:

    text = "\n".join(
        f"[{line['timestamp']}] "
        f"{line['text']}"
        for line in lines
    )

    prompt = f"""
Du analyserar en del av en svensk podcast om löpning,
maraton och träning.

Bestäm vad som är HUVUDÄMNET i den här delen.

Det viktiga är inte att hitta på en fin rubrik.
Det viktiga är att korrekt beskriva vad personerna faktiskt
pratar om.

Ge:

1. Ett kort ämne.
2. En kort beskrivning av ämnet.

Returnera ENDAST giltig JSON:

{{
  "topic": "kort ämne",
  "description": "kort beskrivning"
}}

Hitta inte på information som inte finns i texten.

Transkript:

{text}
"""

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

        return None

    topic = str(
        data.get(
            "topic",
            "",
        )
    ).strip()

    description = str(
        data.get(
            "description",
            "",
        )
    ).strip()

    if not topic:
        return None

    return {
        "timestamp": lines[0][
            "timestamp"
        ],
        "seconds": lines[0][
            "seconds"
        ],
        "topic": topic,
        "description": description,
    }


def analyse_all_topic_windows(
    transcript: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    windows = make_topic_windows(
        transcript
    )

    results = []

    print(
        f"Analyserar {len(windows)} "
        f"ämnesdelar..."
    )

    for i, window in enumerate(
        windows,
        start=1,
    ):

        print(
            f"  Ämnesdel "
            f"{i}/{len(windows)} "
            f"({window[0]['timestamp']})..."
        )

        result = analyse_topic_window(
            window
        )

        if result:
            results.append(
                result
            )

    return results


# ============================================================
# Merge adjacent topics
# ============================================================

def merge_topics(
    topics: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    if not topics:
        return []

    merged = [
        dict(topics[0])
    ]

    for current in topics[1:]:

        previous = merged[-1]

        # Låt Ollama avgöra om två intilliggande
        # delar egentligen handlar om samma sak.
        prompt = f"""
Du avgör om två intilliggande delar av en svensk podcast
handlar om samma huvudämne.

Del 1:
Ämne: {previous['topic']}
Beskrivning: {previous['description']}

Del 2:
Ämne: {current['topic']}
Beskrivning: {current['description']}

Svara ENDAST med:

SAMMA

eller

NYTT

Använd SAMMA om delarna huvudsakligen handlar om samma
sak även om de använder lite olika formuleringar.

Använd NYTT om samtalet tydligt har gått över till ett nytt
huvudämne.
"""

        response = ask_ollama(
            prompt,
            temperature=0.0,
        ).strip().upper()

        if response.startswith(
            "SAMMA"
        ):

            # Behåll starttiden från det första ämnet
            # och slå ihop beskrivningarna.
            previous["description"] = (
                previous["description"]
                + " "
                + current["description"]
            )

        else:

            merged.append(
                dict(current)
            )

    return merged


# ============================================================
# Final chapter titles
# ============================================================

def make_chapter_title(
    topic: dict[str, Any],
) -> str:

    prompt = f"""
Skapa en kort och naturlig svensk kapitelrubrik för en
podcast.

Ämne:
{topic['topic']}

Beskrivning:
{topic['description']}

Rubriken ska:

- vara kort
- vara konkret
- beskriva vad som faktiskt diskuteras
- fungera som en Markdown-rubrik
- inte vara en hel mening om det inte behövs

Exempel:

"Eriks träningsstart"
"30-kilometerspasset"
"Träningsmängd och intensitet"
"Intervallträning inför maraton"
"Grundträning och MAF-test"

Returnera ENDAST rubriken.

Ingen punkt.
Inga citattecken.
Ingen förklaring.
"""

    title = ask_ollama(
        prompt,
        temperature=0.1,
    ).strip()

    title = title.strip(
        '"'
    ).strip()

    if title.startswith(
        "Rubrik:"
    ):

        title = title.split(
            ":",
            1,
        )[1].strip()

    return title


def select_chapters(
    topics: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    if not topics:
        return []

    selected = []

    for topic in topics:

        if not selected:

            selected.append(
                topic
            )

            continue

        gap = (
            topic["seconds"]
            - selected[-1]["seconds"]
        )

        if gap < MIN_CHAPTER_SECONDS:

            # Om ämnesbytet sker för snabbt efter
            # föregående kapitel är det oftast bättre
            # att låta föregående kapitel fortsätta.
            continue

        selected.append(
            topic
        )

    chapters = []

    for topic in selected:

        title = make_chapter_title(
            topic
        )

        if not title:

            title = topic["topic"]

        chapters.append(
            {
                "timestamp": topic[
                    "timestamp"
                ],
                "seconds": topic[
                    "seconds"
                ],
                "title": title,
            }
        )

    return chapters


# ============================================================
# Episode topics / keywords
# ============================================================

def sample_transcript(
    transcript: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    if len(transcript) <= TOPIC_SAMPLE_LINES:
        return transcript

    result = []

    step = (
        len(transcript)
        / TOPIC_SAMPLE_LINES
    )

    for i in range(
        TOPIC_SAMPLE_LINES
    ):

        index = min(
            int(i * step),
            len(transcript) - 1,
        )

        result.append(
            transcript[index]
        )

    return result


def generate_topics_keywords(
    transcript: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:

    selected = sample_transcript(
        transcript
    )

    text = "\n".join(
        f"[{line['timestamp']}] "
        f"{line['text']}"
        for line in selected
    )

    prompt = f"""
Identifiera de viktigaste ämnena och nyckelorden i denna
svenska podcast.

Podcasten handlar huvudsakligen om löpning, maraton och
träning.

Använd endast information som faktiskt finns i texten.

ÄMNEN ska vara övergripande ämnen som är relevanta för
hela avsnittet.

NYCKELORD ska vara konkreta personer, träningsformer,
begrepp, lopp, distanser eller andra viktiga saker som
faktiskt nämns.

Undvik generiska ord som:

"podcast"
"träning"
"löpning"

om de inte tillför något.

Returnera ENDAST giltig JSON i detta format:

{{
  "topics": [
    "ämne 1",
    "ämne 2",
    "ämne 3"
  ],
  "keywords": [
    "nyckelord 1",
    "nyckelord 2",
    "nyckelord 3"
  ]
}}

Ge ungefär 5-8 ämnen och 8-15 nyckelord.

Transkript:

{text}
"""

    print(
        "Skapar ämnen och nyckelord..."
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
            "VARNING: kunde inte tolka "
            "ämnen/nyckelord.",
            file=sys.stderr,
        )

        return [], []

    topics = [
        str(x).strip()
        for x in data.get(
            "topics",
            [],
        )
        if str(x).strip()
    ]

    keywords = [
        str(x).strip()
        for x in data.get(
            "keywords",
            [],
        )
        if str(x).strip()
    ]

    return topics, keywords


# ============================================================
# Markdown helpers
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


def find_chapter_before(
    seconds: float,
    chapters: list[dict[str, Any]],
) -> dict[str, Any] | None:

    result = None

    for chapter in chapters:

        if chapter["seconds"] <= seconds:

            result = chapter

        else:

            break

    return result


# ============================================================
# Markdown output
# ============================================================

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

        if line.strip() in {
            "## Kapitel",
            "## Transkription",
        }:

            break

        prefix.append(line)

    output = list(prefix)

    # --------------------------------------------------------
    # Topics
    # --------------------------------------------------------

    output.append("")

    if topics:

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

    for chapter in chapters:

        timestamp = chapter[
            "timestamp"
        ]

        title = chapter[
            "title"
        ]

        output.append(
            f"- [{timestamp}] "
            f"{title}"
        )

    # --------------------------------------------------------
    # Transcript
    # --------------------------------------------------------

    output.append("")
    output.append("## Transkription")
    output.append("")

    previous_chapter = None

    for line in transcript:

        chapter = find_chapter_before(
            line["seconds"],
            chapters,
        )

        if (
            chapter is not None
            and chapter
            is not previous_chapter
        ):

            output.append("")

            output.append(
                f"### {chapter['title']}"
            )

            output.append("")

            previous_chapter = chapter

        identity = line.get(
            "identity",
            "Unknown",
        )

        if identity not in {
            "Johan Forsstedt",
            "Erik Olofsson",
            "Guest",
            "Unknown",
        }:

            identity = "Unknown"

        output.append(
            f"**[{line['timestamp']}]** "
            f"{identity}: "
            f"{line['text']}"
        )

    output.append("")

    return "\n".join(output)


# ============================================================
# Process one episode
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

    if (
        output_file.exists()
        and not force
    ):

        print(
            f"Hoppar över avsnitt "
            f"{episode_number}: "
            f"{output_file.name} finns redan."
        )

        return

    print()
    print("=" * 70)
    print(
        f"Avsnitt {episode_number}: "
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
            "Hittade inga transkriptrader."
        )

    print(
        f"Laddade {len(transcript)} "
        f"transkriptrader."
    )

    # --------------------------------------------------------
    # Diarization
    # --------------------------------------------------------

    diarization = load_diarization(
        diarization_file
    )

    print(
        f"Laddade {len(diarization)} "
        f"diariseringssegment."
    )

    assign_local_speakers(
        transcript,
        diarization,
    )

    # --------------------------------------------------------
    # Speaker identification
    # --------------------------------------------------------

    print()
    print(
        "Identifierar talare lokalt "
        "i dialogblock..."
    )

    identify_all_speakers(
        transcript
    )

    # --------------------------------------------------------
    # Transcript correction
    # --------------------------------------------------------

    print()
    print(
        "Korrigerar Whisper-transkript..."
    )

    correct_transcript(
        transcript,
        progress_file,
        force,
    )

    # --------------------------------------------------------
    # Topic analysis
    # --------------------------------------------------------

    print()

    topic_windows = (
        analyse_all_topic_windows(
            transcript
        )
    )

    merged_topics = merge_topics(
        topic_windows
    )

    chapters = select_chapters(
        merged_topics
    )

    # --------------------------------------------------------
    # Episode topics / keywords
    # --------------------------------------------------------

    topics, keywords = (
        generate_topics_keywords(
            transcript
        )
    )

    # --------------------------------------------------------
    # Build output
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

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "episode": episode_number,
        "ollama_model": OLLAMA_MODEL,
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

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    if progress_file.exists():

        progress_file.unlink()

    print()
    print(
        f"Skrev: {output_file.name}"
    )

    print(
        f"Skrev: {metadata_file.name}"
    )


# ============================================================
# CLI
# ============================================================

def main() -> None:

    global OLLAMA_MODEL

    parser = argparse.ArgumentParser(
        description=(
            "Efterbehandla Maratonlabbet-"
            "transkript med Ollama."
        )
    )

    parser.add_argument(
        "--only",
        type=int,
        help="Bearbeta endast ett avsnitt.",
    )

    parser.add_argument(
        "--from",
        dest="from_episode",
        type=int,
        help="Bearbeta från och med detta avsnitt.",
    )

    parser.add_argument(
        "--model",
        default=OLLAMA_MODEL,
        help="Ollama-modell.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Bearbeta om även om output redan finns.",
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
            "Hittade inga avsnitt."
        )

        return

    print(
        f"Använder Ollama-modell: "
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
                "\nAvbrutet."
            )

            sys.exit(1)

        except Exception as exc:

            print(
                f"\nFEL vid bearbetning av "
                f"avsnitt {episode_number}: "
                f"{exc}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
