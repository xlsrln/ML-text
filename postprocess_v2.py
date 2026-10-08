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

OLLAMA_TIMEOUT = 600

# Number of transcript lines sent to Ollama at once.
# Keeping this reasonably small helps Qwen stay focused.
LINES_PER_BATCH = 80


# ============================================================
# JSON schema for Ollama
# ============================================================

OUTPUT_SCHEMA = {
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
# Markdown parsing
# ============================================================

TIMESTAMP_RE = re.compile(
    r"^\*\*\[(\d{1,2}:\d{2}(?::\d{2})?)\]\*\*\s*(.*)$"
)


def parse_transcript(markdown: str):
    """
    Extract transcript lines:

        **[00:05]** Hello there

    Returns:
        [
            {
                "timestamp": "00:05",
                "text": "Hello there"
            },
            ...
        ]
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


# ============================================================
# Episode metadata
# ============================================================

def extract_title(markdown: str) -> str:
    match = re.search(
        r"^#\s+Avsnitt\s+\d+:\s*(.+)$",
        markdown,
        re.MULTILINE,
    )

    if match:
        return match.group(1).strip()

    return "Okänt avsnitt"


def extract_episode_number(path: Path) -> str:
    match = re.match(r"(\d+)-", path.name)

    if match:
        return match.group(1)

    return "?"


# ============================================================
# Ollama
# ============================================================

def ask_ollama(prompt: str) -> dict:

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,

        # Qwen3 supports this at the top level.
        "think": False,

        # IMPORTANT:
        # Give Ollama the actual JSON schema instead of just
        # asking for JSON in natural language.
        "format": OUTPUT_SCHEMA,

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
            f"Ollama returnerade HTTP {response.status_code}:\n"
            f"{response.text[:3000]}"
        )

    try:
        outer = response.json()
    except json.JSONDecodeError:
        raise RuntimeError(
            "Ollama returnerade inte giltig JSON:\n"
            + response.text[:3000]
        )

    raw = outer.get("response", "")

    if not raw:
        print("\nOllama-svar:")
        print(json.dumps(outer, ensure_ascii=False, indent=2)[:5000])

        raise RuntimeError(
            "Ollama returnerade ett tomt 'response'-fält."
        )

    print("\nSvar från Ollama:")
    print(raw[:1500])

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print("\n--- Ogiltigt JSON från Ollama ---")
        print(raw[:5000])
        print("--- slut ---")

        raise RuntimeError(
            f"Ollamas response kunde inte parsas som JSON: {exc}"
        ) from exc

    return data


# ============================================================
# Prompt
# ============================================================

def make_prompt(
    title: str,
    transcript_lines: list[dict],
    is_full_episode: bool = True,
) -> str:

    transcript = "\n".join(
        f"[{item['timestamp']}] {item['text']}"
        for item in transcript_lines
    )

    if is_full_episode:
        task = """
Analysera hela podcastavsnittet.

1. Skriv en kort men informativ sammanfattning av hela avsnittet.
2. Identifiera de viktigaste ämnesbytena och skapa kapitel.
3. Ge varje kapitel en kort, tydlig titel.
4. Ge varje kapitel en kort sammanfattning.
5. Ta fram 5–15 relevanta ämnesord/keywords.

Välj bara kapitel när samtalet faktiskt byter ämne.
Skapa inte ett kapitel för varje liten fråga eller kommentar.

Använd tidsstämplarna i transkriptionen för kapitlens starttid.
"""
    else:
        task = """
Analysera detta transkriptutdrag.

Identifiera:
- viktiga ämnen
- viktiga fakta
- möjliga kapitel
- relevanta keywords

Detta är ett delutdrag, så sammanfattningen behöver bara beskriva
vad som faktiskt förekommer i utdraget.
"""

    return f"""
Du arbetar med efterbearbetning av en svensk podcasttranskription.

Podcastavsnitt:
{title}

{task}

VIKTIGT:

Du ska INTE skriva om transkriptionen.

Du ska INTE återge transkriptionen.

Du ska INTE skapa ett fält som heter "transcript".

Du ska analysera innehållet och returnera endast JSON enligt det
schema som API:t har skickat till dig.

Var noggrann med svenska namn, löptermer och träningsbegrepp.

TRANSKRIPTION:

{transcript}
"""


# ============================================================
# Validation
# ============================================================

def validate_result(data: dict) -> bool:

    required = [
        "episode_summary",
        "chapters",
        "keywords",
    ]

    missing = [
        key for key in required
        if key not in data
    ]

    if missing:
        print("\nFEL: Ollama returnerade inte rätt struktur.")
        print("Saknade fält:", ", ".join(missing))
        print(
            "Returnerade fält:",
            ", ".join(data.keys()),
        )

        print("\nOllama returnerade:")
        print(
            json.dumps(
                data,
                ensure_ascii=False,
                indent=2,
            )[:5000]
        )

        return False

    if not isinstance(data["episode_summary"], str):
        print("FEL: episode_summary är inte en sträng.")
        return False

    if not isinstance(data["chapters"], list):
        print("FEL: chapters är inte en lista.")
        return False

    if not isinstance(data["keywords"], list):
        print("FEL: keywords är inte en lista.")
        return False

    return True


# ============================================================
# Markdown output
# ============================================================

def build_markdown(
    original_markdown: str,
    data: dict,
) -> str:

    summary = data["episode_summary"]
    chapters = data["chapters"]
    keywords = data["keywords"]

    # --------------------------------------------------------
    # Remove old generated sections if they exist.
    # --------------------------------------------------------

    text = original_markdown

    # Remove existing AI-generated section.
    text = re.split(
        r"\n## AI-efterbearbetning\s*\n",
        text,
        maxsplit=1,
    )[0].rstrip()

    output = []

    # Keep original content.
    output.append(text)

    output.append("")
    output.append("## AI-efterbearbetning")
    output.append("")
    output.append("### Sammanfattning")
    output.append("")
    output.append(summary.strip())

    # --------------------------------------------------------
    # Chapters
    # --------------------------------------------------------

    if chapters:
        output.append("")
        output.append("### Kapitel")
        output.append("")

        for chapter in chapters:

            timestamp = chapter.get("timestamp", "").strip()
            title = chapter.get("title", "").strip()
            chapter_summary = chapter.get(
                "summary",
                "",
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

    # --------------------------------------------------------
    # Keywords
    # --------------------------------------------------------

    if keywords:
        output.append("")
        output.append("### Ämnesord")
        output.append("")

        clean_keywords = []

        for keyword in keywords:
            keyword = str(keyword).strip()

            if keyword and keyword not in clean_keywords:
                clean_keywords.append(keyword)

        if clean_keywords:
            output.append(
                ", ".join(clean_keywords)
            )

    output.append("")

    return "\n".join(output)


# ============================================================
# Process one episode
# ============================================================

def process_episode(path: Path):

    print()
    print("=" * 70)
    print(f"Bearbetar {path.name}")
    print("=" * 70)

    markdown = path.read_text(
        encoding="utf-8"
    )

    transcript_lines = parse_transcript(markdown)

    if not transcript_lines:
        print("Hittade ingen transkription.")
        return

    print(
        f"Hittade {len(transcript_lines)} transkriptrader."
    )

    title = extract_title(markdown)

    print(f"Avsnitt: {title}")

    # --------------------------------------------------------
    # For now, send the complete episode.
    # --------------------------------------------------------

    prompt = make_prompt(
        title,
        transcript_lines,
        is_full_episode=True,
    )

    print("Skickar avsnittet till Ollama...")

    data = ask_ollama(prompt)

    if not validate_result(data):
        print("\nAvsnittet sparades inte.")
        return

    # --------------------------------------------------------
    # Save metadata separately.
    # --------------------------------------------------------

    metadata_path = path.with_suffix(
        ".metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Sparade metadata: {metadata_path.name}"
    )

    # --------------------------------------------------------
    # Build processed Markdown.
    # --------------------------------------------------------

    output = build_markdown(
        markdown,
        data,
    )

    output_path = path.with_suffix(
        ".processed.md"
    )

    output_path.write_text(
        output,
        encoding="utf-8",
    )

    print(
        f"Sparade Markdown: {output_path.name}"
    )

    print()
    print("Klart.")


# ============================================================
# Find episodes
# ============================================================

def find_episode_files():

    files = sorted(
        EPISODES_DIR.glob("*.md")
    )

    # Never process generated files as input.
    files = [
        path
        for path in files
        if not path.name.endswith(".processed.md")
        and not path.name.endswith(".summary.md")
    ]

    return files


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Efterbearbeta podcasttranskriptioner med Ollama."
    )

    parser.add_argument(
        "--only",
        type=int,
        help="Bearbeta endast ett avsnitt, t.ex. --only 2",
    )

    parser.add_argument(
        "--from",
        dest="from_episode",
        type=int,
        help="Börja från ett visst avsnitt",
    )

    args = parser.parse_args()

    if not EPISODES_DIR.exists():
        print(
            f"Hittar inte katalogen: {EPISODES_DIR}"
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
        f"Använder Ollama-modell: {OLLAMA_MODEL}"
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
                f"FEL vid bearbetning av {path.name}:"
            )
            print(exc)
            print()

            # Continue to next episode rather than
            # killing the entire batch.
            continue


if __name__ == "__main__":
    main()
