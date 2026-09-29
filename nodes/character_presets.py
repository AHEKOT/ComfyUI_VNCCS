"""Curated Creator V2 presets and non-destructive species prompt expansion."""

import json
import os
import re


CHARACTER_PRESETS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "character_template", "character_presets_v2.json",
)
with open(CHARACTER_PRESETS_PATH, encoding="utf-8") as preset_file:
    CHARACTER_PRESETS = json.load(preset_file)


def preset_key(value):
    return " ".join(str(value or "").replace("_", " ").lower().split())


RACE_PRESETS = {}
for preset in CHARACTER_PRESETS["tags"]["races"]:
    for alias in (preset["tag"], preset["label"], *preset.get("synonyms", [])):
        RACE_PRESETS[preset_key(alias)] = preset


def race_features(value):
    """Explain known species once, without changing supplied character fields."""
    source = str(value or "").strip()
    descriptions = []
    seen = set()
    for token in source.split(","):
        token = token.strip().strip("()[]")
        token = re.sub(r":\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*$", "", token)
        preset = RACE_PRESETS.get(preset_key(token))
        if preset and preset["tag"] not in seen:
            seen.add(preset["tag"])
            descriptions.append(f'{preset["label"]}: {preset["prompt"]}')
    if not descriptions:
        return ""
    return (
        "Typical species features (explicit character traits override these defaults): "
        + " ".join(descriptions)
    )


def race_prompt(value):
    """Keep supplied species/custom traits verbatim and append their visual hints."""
    source = str(value or "").strip()
    features = race_features(source)
    return f"{source}; {features}" if features else source
