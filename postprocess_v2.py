import argparse
import json
import re
import sys
from pathlib import Path

import requests


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
EPISODES_DIR = BASE_DIR / "episodes"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:8b"

# Qwen gets this many transcript lines per analysis call.
LINES_PER_CHUNK = 80

# Ollama timeout per request.
OLLAMA_TIMEOUT = 600


# ============================================================
# Final output schema
# ============================================================

FINAL_SCHEMA = {
    "type": "object",
    "properties": {
        "episode_summary": {
            "type": "string"
        },
        "chapters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "timestamp": {
                        "type": "string"
                    },
                    "title": {
                        "type": "string"
                    },
                    "summary": {
                        "type": "string"
                    }
                },
                "required": [
                    "timestamp",
                    "title",
                    "summary"
                ]
            }
        },
        "keywords": {
            "type": "array",
            "items": {
                "type": "string"
            }
        }
    },
    "required": [
        "episode_summary",
        "chapters",
        "keywords"
    ]
}


# ============================================================
# Chunk output schema
# ============================================================

CHUNK_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string"
        },
        "topics": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },
        "chapter_candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "timestamp": {
                        "type": "string"
                    },
                    "title": {
                        "type": "string"
                    },
                    "summary": {
                        "type": "string"
                    }
                },
                "required": [
                    "timestamp",
                    "title",
                    "summary"
                ]
            }
        },
        "keywords": {
            "type": "array",
            "items": {
                "type": "string"
            }
        }
    },
    "required": [
        "summary",
        "topics",
        "chapter_candidates",
        "keywords"
    ]
}


# ============================================================
# Timestamp / transcript parsing
# ============================================================

TIMESTAMP_RE = re.compile(
    r"^\*\*\[(\d{1,2}:\d{2}(?::\d{2})?)\]\*\*\s*(.*)$"
)


def parse_transcript(markdown: str):
    """
    Extract transcript lines from the Markdown file.
    """

    lines = []
    in_transcript = False

    for line in markdown.splitlines():

        if line.strip() == "## Transkription":
            in_transcript = True
            continue

        if not in_transcript:
            continue

        match = TIMESTAMP_RE.match(line.strip())

        if not match:
            continue

        timestamp = match.group(1)
        text = match.group(2).strip()

        if text:
            lines.append({
                "timestamp": timestamp,
                "text": text,
            })

    return lines


def extract_title(markdown: str) -> str:

    match = re.search(
        r"^#\s+Avsnitt\s+\d+:\s*(.+)$",
        markdown,
        re.MULTILINE,
    )

    if match:
        return match.group(1).strip()

    return "Okänt avsnitt"


# ============================================================
# Chunking
# ============================================================

def make_chunks(lines, chunk_size=LINES_PER_CHUNK):

    return [
        lines[i:i + chunk_size]
        for i in range(
            0,
            len(lines),
            chunk_size,
        )
    ]


# ============================================================
# Ollama
# ============================================================

def call_ollama(
    prompt: str,
    schema: dict,
) -> dict:

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "format": schema,
        "options": {
            "temperature": 0,
        },
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Kunde inte kontakta Ollama: {exc}"
        ) from exc

    if response.status_code != 200:
        raise RuntimeError(
            f"Ollama returnerade HTTP "
            f"{response.status_code}:\n"
            f"{response.text[:3000]}"
        )

    try:
        outer = response.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Ollama returnerade inte giltig JSON:\n"
            + response.text[:3000]
        ) from exc

    raw = outer.get("response", "")

    if not raw:
        print("\nOllama-svar:")
        print(
            json.dumps(
                outer,
                ensure_ascii=False,
                indent=2,
            )[:5000]
        )

        raise RuntimeError(
            "Ollama returnerade ett tomt response-fält."
        )

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:

        print("\n--- Ogiltigt JSON från Ollama ---")
        print(raw[:5000])
        print("--- slut ---")

        raise RuntimeError(
            f"Kunde inte parsa Ollamas JSON: {exc}"
        ) from exc

    return data


# ============================================================
# Chunk prompt
# ============================================================

def make_chunk_prompt(
    title: str,
    chunk: list[dict],
    chunk_number: int,
    total_chunks: int,
) -> str:

    transcript = "\n".join(
        f"[{item['timestamp']}] {item['text']}"
        for item in chunk
    )

    return f"""
Du analyserar en svensk podcast om löpning och maratonträning.

Podcast:
{title}

Detta är del {chunk_number} av {total_chunks}.

Analysera ENDAST innehållet i detta utdrag.

Ta fram:

1. En kort sammanfattning av vad som faktiskt sägs.
2. De viktigaste ämnena.
3. Kandidater till kapitel när ett tydligt ämnesbyte sker.
4. Relevanta keywords.

VIKTIGT:

- Återge inte hela transkriptionen.
- Skapa inte ett "transcript"-fält.
- Hitta inte på information som inte finns i texten.
- Kapitel ska bara föreslås vid verkliga ämnesbyten.
- Använd tidsstämplarna från texten.
- Skriv på svenska.
- Var särskilt uppmärksam på löpning, träning, maraton,
  träningsfysiologi och namn.

Returnera endast JSON enligt det schema du fått.

TRANSKRIPTION:

{transcript}
"""


# ============================================================
# Final synthesis prompt
# ============================================================

def make_final_prompt(
    title: str,
    chunk_results: list[dict],
) -> str:

    parts = []

    for i, result in enumerate(chunk_results, start=1):

        parts.append(
            f"""
--- DEL {i} ---

Sammanfattning:
{result.get("summary", "")}

Ämnen:
{", ".join(result.get("topics", []))}

Kapitelkandidater:
{json.dumps(
    result.get("chapter_candidates", []),
    ensure_ascii=False,
    indent=2,
)}

Keywords:
{", ".join(result.get("keywords", []))}
"""
        )

    combined = "\n".join(parts)

    return f"""
Du är slutredaktör för en svensk podcast om löpning och
maratonträning.

Podcastavsnitt:
{title}

Du har fått analyser från flera delar av samma podcastavsnitt.

Din uppgift är att skapa den slutliga strukturen för hela avsnittet.

Gör följande:

1. Skriv en bra sammanfattning av hela avsnittet.
2. Skapa en sammanhängande lista med kapitel.
3. Slå ihop kapitelkandidater som handlar om samma ämne.
4. Ta bort kapitel som är för små eller oviktiga.
5. Behåll tidsstämplar från kapitelkandidaterna.
6. Ge kapitlen korta och beskrivande svenska titlar.
7. Ge varje kapitel en kort sammanfattning.
8. Skapa 5–15 relevanta keywords.

Kapitel ska representera verkliga ämnesblock, inte varje liten
fråga eller kommentar.

VIKTIGT:

- Hitta inte på information.
- Återge inte transkriptionen.
- Skapa inte ett "transcript"-fält.
- Skriv på svenska.
- Returnera endast JSON enligt det schema du fått.

Här är delanalyserna:

{combined}
"""


# ============================================================
# Validation
# ============================================================

def validate_chunk_result(data: dict) -> bool:

    required = [
        "summary",
        "topics",
        "chapter_candidates",
        "keywords",
    ]

    missing = [
        key
        for key in required
        if key not in data
    ]

    if missing:
        print(
            "  FEL: saknade fält:",
            ", ".join(missing),
        )
        return False

    return True


def validate_final_result(data: dict) -> bool:

    required = [
        "episode_summary",
        "chapters",
        "keywords",
    ]

    missing = [
        key
        for key in required
        if key not in data
    ]

    if missing:
        print(
            "\nFEL: slutresultatet saknar:",
            ", ".join(missing),
        )

        print(
            json.dumps(
                data,
                ensure_ascii=False,
                indent=2,
            )[:5000]
        )

        return False

    return True


# ============================================================
# Markdown builder
# ============================================================

def build_markdown(
    original_markdown: str,
    data: dict,
) -> str:

    summary = data["episode_summary"]
    chapters = data["chapters"]
    keywords = data["keywords"]

    # Remove previously generated section.
    text = re.split(
        r"\n## AI-efterbearbetning\s*\n",
        original_markdown,
        maxsplit=1,
    )[0].rstrip()

    output = [
        text,
        "",
        "## AI-efterbearbetning",
        "",
        "### Sammanfattning",
        "",
        summary.strip(),
    ]

    if chapters:

        output.extend([
            "",
            "### Kapitel",
            "",
        ])

        for chapter in chapters:

            timestamp = str(
                chapter.get("timestamp", "")
            ).strip()

            title = str(
                chapter.get("title", "")
            ).strip()

            chapter_summary = str(
                chapter.get("summary", "")
            ).strip()

            if not title:
                continue

            if timestamp:
                output.append(
                    f"- **[{timestamp}] {title}**"
                )
            else:
                output.append(
                    f"- **{title}**"
                )

            if chapter_summary:
                output.append(
                    f"  {chapter_summary}"
                )

    if keywords:

        clean_keywords = []

        for keyword in keywords:

            keyword = str(keyword).strip()

            if (
                keyword
                and keyword not in clean_keywords
            ):
                clean_keywords.append(keyword)

        if clean_keywords:

            output.extend([
                "",
                "### Ämnesord",
                "",
                ", ".join(clean_keywords),
            ])

    output.append("")

    return "\n".join(output)


# ============================================================
# Process episode
# ============================================================

def process_episode(path: Path):

    print()
    print("=" * 70)
    print(f"Bearbetar {path.name}")
    print("=" * 70)

    markdown = path.read_text(
        encoding="utf-8"
    )

    transcript_lines = parse_transcript(
        markdown
    )

    if not transcript_lines:
        print("Hittade ingen transkription.")
        return

    print(
        f"Hittade {len(transcript_lines)} transkriptrader."
    )

    title = extract_title(markdown)

    print(f"Avsnitt: {title}")

    chunks = make_chunks(
        transcript_lines
    )

    print(
        f"Delar upp i {len(chunks)} Ollama-delar "
        f"({LINES_PER_CHUNK} rader/del)."
    )

    # --------------------------------------------------------
    # Stage 1: analyze chunks
    # --------------------------------------------------------

    chunk_results = []

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):

        start_time = chunk[0]["timestamp"]
        end_time = chunk[-1]["timestamp"]

        print()
        print(
            f"[{index}/{len(chunks)}] "
            f"{start_time} -> {end_time}"
        )

        prompt = make_chunk_prompt(
            title,
            chunk,
            index,
            len(chunks),
        )

        try:

            result = call_ollama(
                prompt,
                CHUNK_SCHEMA,
            )

        except Exception as exc:

            print(
                f"  FEL: {exc}"
            )

            # Continue with other chunks.
            continue

        if not validate_chunk_result(
            result
        ):
            continue

        chunk_results.append(result)

        print("  OK")

    if not chunk_results:

        raise RuntimeError(
            "Ingen chunk kunde analyseras."
        )

    print()
    print(
        f"Analyserade {len(chunk_results)} "
        f"av {len(chunks)} delar."
    )

    # --------------------------------------------------------
    # Stage 2: final synthesis
    # --------------------------------------------------------

    print()
    print("Skickar delanalyserna till Ollama")
    print("för slutlig sammanställning...")

    final_prompt = make_final_prompt(
        title,
        chunk_results,
    )

    final_data = call_ollama(
        final_prompt,
        FINAL_SCHEMA,
    )

    if not validate_final_result(
        final_data
    ):
        print(
            "\nSlutresultatet kunde inte valideras."
        )
        return

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata_path = path.with_suffix(
        ".metadata.json"
    )

    metadata = {
        "model": OLLAMA_MODEL,
        "chunks": len(chunks),
        "successful_chunks": len(
            chunk_results
        ),
        **final_data,
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Sparade metadata: "
        f"{metadata_path.name}"
    )

    # --------------------------------------------------------
    # Save processed Markdown
    # --------------------------------------------------------

    output = build_markdown(
        markdown,
        final_data,
    )

    output_path = path.with_suffix(
        ".processed.md"
    )

    output_path.write_text(
        output,
        encoding="utf-8",
    )

    print(
        f"Sparade Markdown: "
        f"{output_path.name}"
    )

    print()
    print("Klart.")


# ============================================================
# Find episode files
# ============================================================

def find_episode_files():

    files = sorted(
        EPISODES_DIR.glob("*.md")
    )

    return [
        path
        for path in files
        if not path.name.endswith(
            ".processed.md"
        )
        and not path.name.endswith(
            ".summary.md"
        )
    ]


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Efterbearbeta podcasttranskriptioner "
            "med Ollama."
        )
    )

    parser.add_argument(
        "--only",
        type=int,
        help=(
            "Bearbeta endast ett avsnitt, "
            "t.ex. --only 2"
        ),
    )

    parser.add_argument(
        "--from",
        dest="from_episode",
        type=int,
        help=(
            "Börja från ett visst avsnitt."
        ),
    )

    args = parser.parse_args()

    if not EPISODES_DIR.exists():

        print(
            f"Hittar inte katalogen: "
            f"{EPISODES_DIR}"
        )

        sys.exit(1)

    files = find_episode_files()

    if args.only is not None:

        prefix = f"{args.only:03d}-"

        files = [
            path
            for path in files
            if path.name.startswith(prefix)
        ]

    elif args.from_episode is not None:

        files = [
            path
            for path in files
            if path.name[:3].isdigit()
            and int(path.name[:3])
            >= args.from_episode
        ]

    if not files:

        print("Inga avsnitt hittades.")
        return

    print(
        f"Använder Ollama-modell: "
        f"{OLLAMA_MODEL}"
    )

    for path in files:

        try:

            process_episode(path)

        except KeyboardInterrupt:

            print("\nAvbrutet.")
            sys.exit(1)

        except Exception as exc:

            print()
            print(
                f"FEL vid bearbetning av "
                f"{path.name}:"
            )
            print(exc)
            print()

            continue


if __name__ == "__main__":
    main()
