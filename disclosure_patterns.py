"""
Spanish AI-disclosure regex vocabulary, organized into 4 confidence buckets.

B1 = direct disclosure (high precision)
B2 = tool / model name (naming the generator implies disclosure)
B3 = soft cue (suggestive but ambiguous)
B4 = generic AI mention (high recall, low precision — captures accusations)

Compiled case-insensitive. Accents and gender variants are handled by character
classes (e.g. [eé], [oa]) rather than NFD normalization, so the input text is
matched verbatim.
"""

import re

B1_DIRECT = [
    r"generad[oa]s?\s+(?:con|por|usando|mediante|a\s+trav[eé]s\s+de)\s+(?:la\s+)?(?:IA|I\.A\.|inteligencia\s+artificial)",
    r"cread[oa]s?\s+(?:con|por|usando|mediante)\s+(?:la\s+)?(?:IA|I\.A\.|inteligencia\s+artificial)",
    r"hech[oa]s?\s+(?:con|por|usando|mediante)\s+(?:la\s+)?(?:IA|I\.A\.|inteligencia\s+artificial)",
    r"product[oa]s?\s+(?:con|por|usando)\s+(?:la\s+)?(?:IA|inteligencia\s+artificial)",
    r"realizad[oa]s?\s+(?:con|por|usando)\s+(?:la\s+)?(?:IA|inteligencia\s+artificial)",
    r"dise[ñn]ad[oa]s?\s+(?:con|por|usando)\s+(?:la\s+)?(?:IA|inteligencia\s+artificial)",
    r"imagen(?:es)?\s+(?:sint[eé]tica|artificial|generada(?:s)?\s+(?:por|con)\s+(?:la\s+)?IA)",
    r"foto(?:graf[ií]a)?(?:s)?\s+(?:sint[eé]tica|artificial|generada(?:s)?\s+(?:por|con)\s+(?:la\s+)?IA)",
    r"contenido\s+(?:sint[eé]tico|generado\s+(?:por|con)\s+(?:la\s+)?IA)",
    r"#IAGenerativa\b",
    r"#IAGen\b",
    r"#GeneradoConIA\b",
    r"#HechoConIA\b",
    r"#CreadoConIA\b",
    r"#ArteIA\b",
    r"#AIArt\b",
    r"#AIGenerated\b",
    r"#GenAI\b",
    r"AI[\s\-]?generated",
    r"made\s+with\s+AI",
    r"generated\s+(?:by|with|using)\s+AI",
    r"created\s+(?:by|with|using)\s+AI",
    r"synthetic\s+(?:image|content|media|photo)",
]

B2_TOOL = [
    r"\bMidjourney\b",
    r"\bMJ\b(?=.{0,40}(?:imagen|prompt|render|generad))",
    r"\bDALL[\s\-]?[E2-3]\b",
    r"\bStable\s+Diffusion\b",
    r"\bSDXL\b",
    r"\bFLUX(?:\.1)?\b",
    r"\bSora\b(?=.{0,80}(?:OpenAI|video|AI|IA|generad|prompt))",
    r"\b(?:Adobe\s+)?Firefly\b",
    r"\bLeonardo(?:\.ai|\s+AI)?\b",
    r"\bRunway(?:ML)?\b",
    r"\bIdeogram\b",
    r"\bGoogle\s+Imagen\b",
    r"\bGemini\s*(?:Pro|Nano|Flash|[12](?:\.[05])?)?\b",
    r"\bGrok(?:\s+(?:Imagine|2|3))?\b",
    r"\bChatGPT\b",
    r"\bGPT[\s\-]?(?:[34](?:o|\-mini|\.5)?|4\.1|5)\b",
    r"\bClaude\s*(?:Sonnet|Opus|Haiku)?\b",
    r"\bPika\s+(?:Labs|1\.0|art)?\b",
    r"\bKling(?:\s+AI)?\b",
    r"\bVeo\s*[23]?\b(?=.{0,80}(?:video|Google|OpenAI|AI|IA|generad|prompt))",
    r"\bNano\s+Banana\b",
    r"\bRecraft\b",
    r"\bKrea(?:\.ai)?\b",
    r"\bPlayground\s+AI\b",
    r"\bNightCafe\b",
    r"\bArtBreeder\b",
    r"\bBing\s+Image\s+Creator\b",
    r"\bCopilot\s+(?:Designer|Image)\b",
    r"\bCanva\s+(?:AI|Magic)\b",
    r"\bLumalabs\b",
    r"\bHaiper\b",
    r"\bHeyGen\b",
    r"\bHailuo\b",
    r"\bMiniMax\b",
    r"\bWan(?:\s+(?:2\.[12]|video))\b",
    r"\bElevenLabs\b",
]

B3_SOFT = [
    r"ilustraci[oó]n\s+digital",
    r"arte\s+digital",
    r"\brender(?:\s+3D)?\b",
    r"\brenderizad[oa]\b",
    r"arte\s+conceptual",
    r"concept\s+art",
    r"generad[oa]\s+digitalmente",
    r"creaci[oó]n\s+digital",
    r"manipulad[oa]\s+(?:con|por|usando|digitalmente)",
    r"editad[oa]\s+(?:con|por|usando)\s+(?:la\s+)?(?:IA|inteligencia\s+artificial)",
    r"retocad[oa]\s+(?:con|por|usando)\s+(?:la\s+)?(?:IA|inteligencia\s+artificial)",
    r"filtro\s+de\s+(?:la\s+)?IA",
    r"deepfake(?:s)?",
    r"\bprompt(?:s|ing|eo)?\b",
    r"AI\s+filter",
    r"AI\s+(?:image|art|photo|picture)",
    r"generative\s+AI",
    r"text[\s\-]?to[\s\-]?image",
    r"image[\s\-]?to[\s\-]?image",
    r"img2img",
    r"txt2img",
    r"upscal(?:ed|er|ing)",
    r"inpainting",
    r"outpainting",
]

B4_GENERIC = [
    r"\binteligencia\s+artificial\b",
    r"\bIA\s+generativa\b",
    r"(?<![A-Za-z0-9ÁÉÍÓÚáéíóúÑñ])IA(?![A-Za-z0-9ÁÉÍÓÚáéíóúÑñ])",
    r"\bAI\b",
    r"\balgoritmo\b",
    r"\bchatbot\b",
    r"red(?:es)?\s+neuronal(?:es)?",
    r"\bmodelo\s+de\s+lenguaje\b",
    r"large\s+language\s+model",
    r"\bLLM\b",
    r"machine\s+learning",
    r"aprendizaje\s+autom[aá]tico",
    r"aprendizaje\s+profundo",
    r"deep\s+learning",
    r"\bGenAI\b",
]

BUCKETS = {
    "B1_direct": B1_DIRECT,
    "B2_tool": B2_TOOL,
    "B3_soft": B3_SOFT,
    "B4_generic": B4_GENERIC,
}

# Compile once. Per-pattern regex (not unioned) so the matched_terms field
# can list every individual pattern that fired for a given text.
COMPILED = {
    bucket: [(pat, re.compile(pat, re.IGNORECASE)) for pat in pats]
    for bucket, pats in BUCKETS.items()
}


def scan(text: str):
    """Return dict of bucket -> list of (pattern, matched_substring)."""
    if not text or not isinstance(text, str):
        return {}
    out = {}
    for bucket, compiled in COMPILED.items():
        hits = []
        for pat, rx in compiled:
            m = rx.search(text)
            if m:
                hits.append((pat, m.group(0)))
        if hits:
            out[bucket] = hits
    return out


def snippet(text: str, span_start: int, span_end: int, radius: int = 100) -> str:
    """200-char window (radius on each side) around a match span."""
    if not text:
        return ""
    lo = max(0, span_start - radius)
    hi = min(len(text), span_end + radius)
    return text[lo:hi].replace("\n", " ").strip()
