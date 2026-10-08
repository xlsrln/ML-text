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

        # Ask Ollama itself to constrain the output to JSON.
        "format": "json",

        "options": {
            "temperature": 0,
        },

        # Qwen3 supports thinking control.
        # We don't need reasoning text for this task.
        "think": False,
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    data = response.json()

    # Normal /api/generate response.
    result = data.get("response", "")

    if not result:
        print("\nOväntat svar från Ollama:")
        print(
            json.dumps(
                data,
                indent=2,
                ensure_ascii=False,
            )
        )

        raise RuntimeError(
            "Ollama returnerade inget 'response'-fält "
            "eller ett tomt response."
        )

    return result.strip()


def clean_json_response(text: str) -> str:

    text = text.strip()

    if not text:
        raise ValueError(
            "Ollama returnerade en tom sträng."
        )

    # Remove markdown code fences if the model
    # ignored the JSON format request.
    if "```json" in text:

        text = text.split(
            "```json",
            1,
        )[1]

        if "```" in text:
            text = text.split(
                "```",
                1,
            )[0]

    elif "```" in text:

        text = text.split(
            "```",
            1,
        )[1]

        if "```" in text:
            text = text.split(
                "```",
                1,
            )[0]

    text = text.strip()

    # Find the outer JSON object.
    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:

        text = text[
            start:end + 1
        ]

    else:

        raise ValueError(
            "Ollama returnerade inget JSON-objekt.\n\n"
            "Rått svar från Ollama:\n"
            + text
        )

    return text


# ==========================================================
# Parsing
# ==========================================================

def load_episode(path: Path):

    text = path.read_text(
        encoding="utf-8"
    )

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

Fokusera på information som faktiskt finns i avsnittet.

Identifiera särskilt:

- Johan Forsstedts träning
- Erik Olofssons träning
- gäster
- träningsråd
- träningspass
- träningsvolym
- skador
- tävlingar
- personliga rekord
- mål
- böcker
- coacher
- mentala strategier
- viktiga träningsprinciper

Var konservativ.

Hitta inte på information.

Om något inte nämns ska motsvarande lista vara tom.

Om du är osäker på en uppgift ska du hellre utelämna den
än att gissa.

Returnera ENDAST giltig JSON.

JSON-strukturen måste vara exakt:

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

Viktigt:

- Skriv alla värden på svenska.
- Använd korta, konkreta formuleringar.
- Upprepa inte samma information i flera kategorier om det
  inte behövs.
- "workouts" ska innehålla konkreta träningspass som nämns.
- "races" ska innehålla konkreta lopp/tävlingar som nämns.
- "training_principles" ska innehålla träningsidéer eller
  principer som faktiskt diskuteras.
- "keywords" ska vara användbara sökord, inte bara allmänna
  ord som "träning" och "löpning".

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
        f"**Huvudämne:** "
        f"{summary.get('main_topic', '')}"
    )

    lines.append("")

    lines.append(
        "## Viktigaste lärdomarna"
    )

    lines.append("")

    for item in summary.get(
        "key_takeaways",
        [],
    ):

        lines.append(
            f"- {item}"
        )

    lines.append("")
    lines.append("## Johan")

    for title, key in [
        ("Träning", "training"),
        ("Mål", "goals"),
        ("Skador", "injuries"),
        ("Åsikter", "opinions"),
    ]:

        values = data.get(
            "johan",
            {},
        ).get(
            key,
            [],
        )

        lines.append("")
        lines.append(
            f"### {title}"
        )

        for item in values:

            lines.append(
                f"- {item}"
            )

    lines.append("")
    lines.append("## Erik")

    for title, key in [
        ("Träning", "training"),
        ("Mål", "goals"),
        ("Skador", "injuries"),
        ("Åsikter", "opinions"),
    ]:

        values = data.get(
            "erik",
            {},
        ).get(
            key,
            [],
        )

        lines.append("")
        lines.append(
            f"### {title}"
        )

        for item in values:

            lines.append(
                f"- {item}"
            )

    guests = data.get(
        "guests",
        [],
    )

    if guests:

        lines.append("")
        lines.append("## Gäster")

        for guest in guests:

            name = guest.get(
                "name",
                "Okänd",
            )

            lines.append("")
            lines.append(
                f"### {name}"
            )

            background = guest.get(
                "background",
                "",
            )

            if background:

                lines.append("")
                lines.append(
                    background
                )

            advice = guest.get(
                "advice",
                [],
            )

            if advice:

                lines.append("")
                lines.append("Råd:")

                for item in advice:

                    lines.append(
                        f"- {item}"
                    )

    sections = [
        (
            "Träningsprinciper",
            "training_principles",
        ),
        (
            "Träningspass",
            "workouts",
        ),
        (
            "Tävlingar",
            "races",
        ),
        (
            "Skador",
            "injuries",
        ),
        (
            "Böcker",
            "books",
        ),
        (
            "Coacher",
            "coaches",
        ),
        (
            "Nyckelord",
            "keywords",
        ),
    ]

    for title, key in sections:

        values = data.get(
            key,
            [],
        )

        if not values:
            continue

        lines.append("")
        lines.append(
            f"## {title}"
        )

        for value in values:

            lines.append(
                f"- {value}"
            )

    return "\n".join(lines)


# ==========================================================
# Processing
# ==========================================================

def process_episode(path: Path):

    print()
    print("=" * 70)
    print(
        f"Bearbetar {path.name}"
    )
    print("=" * 70)

    episode = load_episode(
        path
    )

    prompt = build_prompt(
        episode
    )

    print(
        "Skickar avsnittet till Ollama..."
    )

    response = ask_ollama(
        prompt
    )

    print(
        "Svar från Ollama:"
    )

    print(
        response[:1000]
    )

    print()

    response = clean_json_response(
        response
    )

    try:

        data = json.loads(
            response
        )

    except json.JSONDecodeError as exc:

        print(
            "Kunde inte tolka Ollamas svar som JSON.",
        )

        print(
            "\nRensat svar:"
        )

        print(response)

        raise RuntimeError(
            f"JSON-fel: {exc}"
        ) from exc

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

        prefix = (
            f"{args.only:03d}-"
        )

        files = [
            f
            for f in files
            if f.name.startswith(
                prefix
            )
        ]

    for file in files:

        if file.name.endswith(
            ".summary.md"
        ):
            continue

        if file.name.endswith(
            ".processed.md"
        ):
            continue

        process_episode(file)


if __name__ == "__main__":
    main()
