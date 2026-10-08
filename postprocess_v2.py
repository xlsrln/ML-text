import argparse
import json
import re
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent
EPISODES_DIR = BASE_DIR / "episodes"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:8b"

REQUEST_TIMEOUT = 600


# ==========================================================
# Ollama
# ==========================================================

def ask_ollama(prompt: str) -> str:

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "think": False,
        },
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    data = response.json()

    return data["response"]


def clean_json_response(text: str) -> str:

    text = text.strip()

    if "```json" in text:
        text = text.split("```json", 1)[1]

    if "```" in text:
        text = text.split("```", 1)[0]

    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        text = text[start:end + 1]

    return text


# ==========================================================
# Parsing
# ==========================================================

def load_episode(path: Path):

    text = path.read_text(encoding="utf-8")

    title_match = re.search(
        r"^#\s+(.*)$",
        text,
        flags=re.MULTILINE,
    )

    title = (
        title_match.group(1).strip()
        if title_match
        else path.stem
    )

    description = ""

    if "## Om avsnittet" in text:

        after = text.split(
            "## Om avsnittet",
            1,
        )[1]

        description = after.split(
            "##",
            1,
        )[0].strip()

    transcript = ""

    if "## Transkription" in text:

        transcript = text.split(
            "## Transkription",
            1,
        )[1]

    return {
        "title": title,
        "description": description,
        "transcript": transcript,
        "raw": text,
    }


# ==========================================================
# Prompt
# ==========================================================

def build_prompt(data):

    return f"""
Du analyserar ett avsnitt av podcasten Maratonlabbet.

Syftet är att skapa en långsiktig kunskapsbas om löpning,
maratonträning, Johan Forsstedt och Erik Olofsson.

Fokusera på:

- Johan Forsstedts träning
- Erik Olofssons träning
- Gäster
- Träningsråd
- Pass som nämns
- Träningsvolym
- Skador
- Tävlingar
- Böcker
- Coacher
- Mentala strategier

Returnera ENDAST giltig JSON.

{{
  "episode_summary": {{
    "main_topic": "",
    "key_takeaways": []
  }},

  "johan": {{
    "training": [],
    "goals": [],
    "injuries": [],
    "opinions": []
  }},

  "erik": {{
    "training": [],
    "goals": [],
    "injuries": [],
    "opinions": []
  }},

  "guests": [
    {{
      "name": "",
      "background": "",
      "advice": []
    }}
  ],

  "training_principles": [],

  "workouts": [],

  "races": [],

  "injuries": [],

  "books": [],

  "coaches": [],

  "keywords": []
}}

Titel:

{data["title"]}

Beskrivning:

{data["description"]}

Transkript:

{data["transcript"]}
"""


# ==========================================================
# Markdown output
# ==========================================================

def build_markdown(data):

    lines = []

    summary = data["episode_summary"]

    lines.append("# Sammanfattning")
    lines.append("")

    lines.append(
        f"**Huvudämne:** {summary['main_topic']}"
    )
    lines.append("")

    lines.append("## Viktigaste lärdomarna")
    lines.append("")

    for item in summary["key_takeaways"]:
        lines.append(f"- {item}")

    lines.append("")
    lines.append("## Johan")

    for section in [
        ("Träning", "training"),
        ("Mål", "goals"),
        ("Skador", "injuries"),
        ("Åsikter", "opinions"),
    ]:

        lines.append("")
        lines.append(f"### {section[0]}")

        for item in data["johan"][section[1]]:
            lines.append(f"- {item}")

    lines.append("")
    lines.append("## Erik")

    for section in [
        ("Träning", "training"),
        ("Mål", "goals"),
        ("Skador", "injuries"),
        ("Åsikter", "opinions"),
    ]:

        lines.append("")
        lines.append(f"### {section[0]}")

        for item in data["erik"][section[1]]:
            lines.append(f"- {item}")

    if data["guests"]:

        lines.append("")
        lines.append("## Gäster")

        for guest in data["guests"]:

            lines.append("")
            lines.append(
                f"### {guest['name']}"
            )

            if guest["background"]:
                lines.append("")
                lines.append(
                    guest["background"]
                )

            lines.append("")
            lines.append("Råd:")

            for advice in guest["advice"]:
                lines.append(
                    f"- {advice}"
                )

    sections = [
        ("Träningsprinciper", "training_principles"),
        ("Träningspass", "workouts"),
        ("Tävlingar", "races"),
        ("Skador", "injuries"),
        ("Böcker", "books"),
        ("Coacher", "coaches"),
        ("Nyckelord", "keywords"),
    ]

    for title, key in sections:

        values = data.get(key, [])

        if not values:
            continue

        lines.append("")
        lines.append(f"## {title}")

        for value in values:
            lines.append(
                f"- {value}"
            )

    return "\n".join(lines)


# ==========================================================
# Processing
# ==========================================================

def process_episode(path: Path):

    print(f"Bearbetar {path.name}")

    episode = load_episode(path)

    prompt = build_prompt(
        episode
    )

    response = ask_ollama(
        prompt
    )

    response = clean_json_response(
        response
    )

    data = json.loads(
        response
    )

    json_file = path.with_suffix(
        ".summary.json"
    )

    md_file = path.with_suffix(
        ".summary.md"
    )

    json_file.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    md_file.write_text(
        build_markdown(data),
        encoding="utf-8",
    )

    print(
        f"Skrev {json_file.name}"
    )

    print(
        f"Skrev {md_file.name}"
    )


# ==========================================================
# CLI
# ==========================================================

def main():

    global OLLAMA_MODEL

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--only",
        type=int,
    )

    parser.add_argument(
        "--model",
        default=OLLAMA_MODEL,
    )

    args = parser.parse_args()

    OLLAMA_MODEL = args.model

    files = sorted(
        EPISODES_DIR.glob("*.md")
    )

    if args.only is not None:

        prefix = f"{args.only:03d}-"

        files = [
            f
            for f in files
            if f.name.startswith(prefix)
        ]

    for file in files:

        if file.name.endswith(
            ".summary.md"
        ):
            continue

        process_episode(file)


if __name__ == "__main__":
    main()
