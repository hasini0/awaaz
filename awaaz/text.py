"""Script-aware language detection and normalisation."""
from __future__ import annotations

import re
import unicodedata

LANG_NAMES = {
    "te": "Telugu",
    "hi": "Hindi",
    "ur": "Urdu",
    "en": "English",
    "tenglish": "Telugu (romanised)",
    "hinglish": "Hindi/Urdu (romanised)",
}

_ZW = dict.fromkeys(map(ord, "\u200c\u200d\u200b\ufeff"), None)  # zero-width (non-)joiners, space, BOM

# Common words that give away romanised Telugu vs romanised Hindi. Words that are also English
# ("sir", "current") are deliberately left out.
_TENGLISH = {"lo", "ledu", "undi", "unnayi", "vasthondi", "vasthunnayi", "cheyyatledu", "maa", "nenu", "chala",
             "raavatledu", "ayindi", "pedda", "neellu", "kukka", "kukkalu", "chetta", "roju", "raathri", "veedhi",
             "dayachesi", "evaru", "ikkada", "meeda", "nundi", "padi", "teega", "andi", "garu",
             "emi", "kaani", "inka", "ippudu", "vachindi", "poyindi", "untundi", "cheyandi", "cheyyandi",
             "daggara", "rojuluga", "nela", "vaaram", "raaledu", "padindi", "lekunda", "therichi", "antha",
             "kooda", "pagalu", "thakkuva", "tharvatha", "intlo", "ee", "loki", "tho", "kante", "pillalaki",
             "domalu", "alaage", "ila", "ilaage", "motham", "oka"}
_TE_SUFFIX = re.compile(r"(ledu|undi|indi|unnayi|thondi|thunnayi|thunnaru|ayi|aaru|uga|lo)$")
_HINGLISH = {"hai", "hain", "nahi", "mein", "ka", "ki", "ke", "raha", "rahi", "rahe", "bahut", "paani", "bijli",
             "kutte", "kachra", "gali", "hamari", "hamare", "kar", "se", "ho", "gaya", "gayi", "koi", "kuch", "yahan",
             "aur", "bhi", "pe", "par", "kripya", "jaldi", "lijiye", "karo", "karwaiye", "bilkul"}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "").translate(_ZW)
    return re.sub(r"\s+", " ", text).strip()


def script_of(text: str) -> str:
    counts = {"te": 0, "hi": 0, "ur": 0, "latin": 0}
    for ch in text:
        o = ord(ch)
        if 0x0C00 <= o <= 0x0C7F:
            counts["te"] += 1
        elif 0x0900 <= o <= 0x097F:
            counts["hi"] += 1
        elif 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F:
            counts["ur"] += 1
        elif ch.isascii() and ch.isalpha():
            counts["latin"] += 1
    return max(counts, key=counts.get) if any(counts.values()) else "latin"


def detect_language(text: str) -> str:
    """Returns one of te, hi, ur, en, tenglish, hinglish."""
    s = script_of(text)
    if s != "latin":
        return s
    words = re.findall(r"[a-z]+", text.lower())
    te = sum(w in _TENGLISH for w in words) + 0.5 * sum(bool(_TE_SUFFIX.search(w)) for w in words if len(w) > 4)
    hi = sum(w in _HINGLISH for w in words)
    if max(te, hi) < 2:
        return "en"
    return "tenglish" if te > hi else "hinglish"


# Sound-alike consonants that speech recognisers (and people) confuse in Indian place names:
# retroflex vs dental (ट/त), aspirated vs plain (ख/क), sibilants (श/ष/स).
_PHONETIC = str.maketrans({
    # Devanagari
    "ट": "त", "ठ": "त", "थ": "त", "ड": "द", "ढ": "द", "ध": "द", "ण": "न", "ख": "क", "घ": "ग", "छ": "च",
    "झ": "ज", "फ": "प", "भ": "ब", "श": "स", "ष": "स", "ङ": "न", "ञ": "न", "ळ": "ल",
    # Telugu
    "ట": "త", "ఠ": "త", "థ": "త", "డ": "ద", "ఢ": "ద", "ధ": "ద", "ణ": "న", "ఖ": "క", "ఘ": "గ", "ఛ": "చ",
    "ఝ": "జ", "ఫ": "ప", "భ": "బ", "శ": "స", "ష": "స", "ళ": "ల",
})
_VOWEL_SIGNS = re.compile(r"[\u093A-\u094F\u0900-\u0903\u0C3E-\u0C4D\u0C00-\u0C03\u0C55\u0C56]")


def phonetic_skeleton(word: str) -> str:
    """Consonant skeleton of a Devanagari/Telugu word: vowel signs dropped, sound-alike consonants merged,
    repeated letters collapsed. "कुकतपली" (ASR output) and "कूकटपल्ली" (Kukatpally) both become "कतपल"."""
    s = _VOWEL_SIGNS.sub("", word).translate(_PHONETIC)
    return re.sub(r"(.)\1+", r"\1", s)
