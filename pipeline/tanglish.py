"""Tamil script -> Tanglish (Tamil written the way people type it in English letters).

Three methods, from simplest to best:
  plain     aksharamukha "RomanReadable" (letter by letter, apostrophes removed)
  rules     our converter: same idea, but uses Tamil sound rules - a letter is
            pronounced differently depending on position (ட between vowels is
            "d", க after ங is "g", word-initial ச is "s", ...)
  rules+en  rules, plus a list mapping English words the speaker used back to
            English spelling (சார் -> sir, வீடியோ -> video, கரெக்டா -> correct-a)

  python tanglish.py "நாளைக்கு கிளாஸ் இருக்கு"
"""
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")

VOWELS = {"அ": "a", "ஆ": "aa", "இ": "i", "ஈ": "ee", "உ": "u", "ஊ": "oo", "எ": "e", "ஏ": "e",
          "ஐ": "ai", "ஒ": "o", "ஓ": "o", "ஔ": "au"}
SIGNS = {"ா": "aa", "ி": "i", "ீ": "ee", "ு": "u", "ூ": "oo", "ெ": "e",
         "ே": "e", "ை": "ai", "ொ": "o", "ோ": "o", "ௌ": "au"}
PULLI = "்"
# base sounds; context rules below override k/ch/t/th/p/R
CONS = {"க": "k", "ங": "ng", "ச": "s", "ஞ": "nj", "ட": "t", "ண": "n", "த": "th", "ந": "n", "ப": "p",
        "ம": "m", "ய": "y", "ர": "r", "ல": "l", "வ": "v", "ழ": "zh", "ள": "l", "ற": "r", "ன": "n",
        "ஜ": "j", "ஷ": "sh", "ஸ": "s", "ஹ": "h", "ஃ": "h"}

# English words as they appear in Tamil script (prefix match; the Tamil suffix
# after them is transliterated and attached with a hyphen: மைண்ட்ல -> mind-la)
LOANS = {
    "தேங்க்யூ": "thank you", "மார்க்": "mark", "ட்ராக்": "track", "டிராக்": "track", "அப்ளை": "apply", "டவுட்": "doubt",
    "டிகிரி": "degree", "நெக்ஸ்ட்": "next", "லாஜிக்": "logic", "ரைட்": "right", "பிரேக்": "break", "மியூட்": "mute",
    "வாய்ஸ்": "voice", "பாயிண்ட்": "point", "எஸ்": "yes", "ப்ளீஸ்": "please", "கன்ப்யூஸ்": "confuse", "ட்ராவல்": "travel",
    "டூர்": "tour", "ட்ரிப்": "trip", "கண்டக்டர்": "conductor", "டிரைவர்": "driver", "பைலட்": "pilot", "ஸ்பீடு": "speed",
    "எக்ஸ்ட்ரா": "extra", "மைனஸ்": "minus", "மெண்டல்": "mental", "சென்னை": "Chennai", "கீழ்ப்பாக்க": "Kilpauk",
    "டி நகர்": "T Nagar", "ஜெயில்": "jail", "போலீஸ்": "police", "மிலிட்டரி": "military", "சைலென்ட்": "silent", "தேங்க் யூ": "thank you", "நைட்": "night", "ஓகே": "ok", "டென்ஷன்": "tension", "பிரச்சனை": "prachanai", "சார்": "sir", "மேடம்": "madam", "போட்டோ": "photo", "ஃபோட்டோ": "photo",
    "வீடியோ": "video", "ஆடியோ": "audio", "கரெக்ட்": "correct", "கரெக்ட": "correct", "மைண்ட்": "mind",
    "ஸ்பீட்": "speed", "ஸ்பீடா": "speed-a", "ஸ்லோ": "slow", "தேங்க்": "thank", "யூ": "you", "டிவி": "TV",
    "ஃபோன்": "phone", "போன்": "phone", "கிளாஸ்": "class", "கிளாஸ": "class", "சூப்பர்": "super",
    "வெரி": "very", "குட்": "good", "நோட்": "note", "புக்": "book", "பாயிண்ட்": "point", "டைம்": "time",
    "லைன்": "line", "சார்ட்": "chart", "ஹவுஸ்": "house", "எக்ஸாம்பிள்": "example", "கேஸ்": "case",
    "கிளியர்": "clear", "டவுட்": "doubt", "மேட்டர்": "matter", "சிம்பிள்": "simple", "நியூஸ்": "news",
    "அசிஸ்டண்ட்": "assistant", "அஜிஸ்டண்ட்": "assistant", "அஜிஸ்டன்ட்": "assistant", "மேனேஜர்": "manager",
    "ஹாஸ்பிட்டல்": "hospital", "டாக்டர்": "doctor", "ஷுகர்": "sugar", "சுகர்": "sugar", "பவர்": "power",
    "பிராக்டிகல்": "practical", "பிளஸ்": "plus", "மைனஸ்": "minus", "லிஸ்ட்": "list", "டீச்சர்": "teacher",
    "ஸ்டூடண்ட்": "student", "கிளாஸ்ல": "class-la", "ஆக்சுவலா": "actually", "பேசிக்": "basic",
    "இம்பார்டன்ட்": "important", "ஈசி": "easy", "ஹெல்ப்": "help", "சேஞ்ச்": "change", "ஸ்டார்ட்": "start",
    "ஃபர்ஸ்ட்": "first", "செகண்ட்": "second", "லாஸ்ட்": "last", "நெக்ஸ்ட்": "next", "வெயிட்": "wait",
    "சாரி": "sorry", "ப்ளீஸ்": "please", "ஓவர்": "over", "டோட்டல்": "total", "ரிசல்ட்": "result",
}
# Astrology and Sanskrit-origin names that sound rules can't get right.
ASTRO = {
    "குரு": "guru", "ராகு": "raahu", "கேது": "kethu", "சனி": "sani", "சந்திரன்": "chandran", "சந்திர": "chandra", "சூரிய": "sooriya",
    "லக்னம்": "lagnam", "லக்கினம்": "lagnam", "ஜாதகம்": "jaathagam", "ராசி": "raasi", "நட்சத்திர": "natchathira",
    "விருச்சிக": "viruchiga", "மேஷ": "mesha", "ரிஷப": "rishaba", "மிதுன": "mithuna", "கடக": "kadaga",
    "சிம்ம": "simma", "கன்னி": "kanni", "துலா": "thulaa", "தனுசு": "dhanusu", "மகர": "magara", "கும்ப": "kumbha",
    "மீன": "meena", "உச்ச": "ucha", "நீச்ச": "neecha", "ஆட்சி": "aatchi", "தசை": "dhasai", "புத்தி": "budhdhi",
    "பாவ": "bhaava", "கிரக": "graha", "யோக": "yoga", "தோஷ": "dhosha", "சுக்கிர": "sukkira", "செவ்வா": "sevvaa",
    "புத": "budha",
}
LOANS.update(ASTRO)
_LOAN_KEYS = sorted(LOANS, key=len, reverse=True)


def _units(word: str):
    """Split a Tamil word into (consonant|None, vowel-sound|'' for pulli) units."""
    out, i = [], 0
    while i < len(word):
        ch = word[i]
        if ch in VOWELS:
            out.append((None, VOWELS[ch])); i += 1
        elif ch in CONS:
            nxt = word[i + 1] if i + 1 < len(word) else ""
            if nxt == PULLI:
                out.append((ch, "")); i += 2
            elif nxt in SIGNS:
                out.append((ch, SIGNS[nxt])); i += 2
            else:
                out.append((ch, "a")); i += 1
        else:
            out.append((ch, None)); i += 1  # digits, Latin, etc. pass through
    return out


def _sound(units, k):
    c, v = units[k]
    if v == "":  # consonant with pulli (no vowel): stays unvoiced (க்ட -> "kt")
        return {"க": "k", "ட": "t", "த": "th", "ப": "p", "ச": "ch", "ற": "r"}.get(c, CONS[c])
    prev = units[k - 1] if k > 0 else None
    initial = prev is None
    after_vowel = prev is not None and prev[1] not in ("", None)
    prev_c = prev[0] if prev is not None and prev[1] == "" else None  # consonant with pulli
    doubled = prev_c == c
    if c == "க":
        s = "k" if (initial or doubled) else "g" if prev_c in ("ங", "ன", "ண", "ம") or after_vowel else "k"
    elif c == "ச":
        s = "s" if initial else "ch" if doubled else "j" if prev_c in ("ஞ", "ன", "ண") else "s"
    elif c == "ட":
        s = "t" if doubled else "d" if (after_vowel or prev_c in ("ண", "ன")) else "t"
    elif c == "த":
        s = "th"  # people type த as "th" almost everywhere (Dakshina: marunthu, intha)
    elif c == "ப":
        s = "p" if (initial or doubled) else "b" if prev_c in ("ம", "ன") else "p"
    elif c == "ற" and prev_c in ("ன", "ந"):
        s = "dr"  # ன்ற -> "ndr" (endru)
    elif c == "ற":
        nxt = units[k + 1][0] if k + 1 < len(units) else None
        s = "r"
    elif c == "ஞ" and initial:
        s = "gny"  # ஞாபகம் -> gnyabagam
    else:
        s = CONS[c]
    if doubled and c in ("க", "ட", "ப", "த") and s in ("k", "t", "p", "th"):
        # write doubled stops once more: க்க -> kk, த்த -> th (people type 'pathu' for பத்து)
        s = "" if c == "த" else s
    return s


def rules(word: str) -> str:
    u = _units(word)
    out = []
    for k, (c, v) in enumerate(u):
        if v is None:
            out.append(c); continue
        if c is None:
            out.append(v); continue
        nxt = u[k + 1] if k + 1 < len(u) else None
        if c == "த" and v == "" and nxt and nxt[0] == "த":
            out.append("th")  # first of த்த
            continue
        if v == "" and nxt and nxt[0] == c and c in ("க", "ட", "ப"):
            out.append(CONS[c]); continue
        if c == "ற" and v == "" and nxt and nxt[0] == "ற":
            out.append("tt"); continue  # ற்ற -> "ttr" (mattrum)
        if c == "ச" and v == "" and nxt and nxt[0] == "ச":
            continue  # ச்ச -> "ch" (written once by the second ச)
        if c in ("ங", "ஞ") and v == "" and nxt and nxt[0] in ("க", "ச", "ஜ"):
            out.append("n"); continue  # ங்க -> nga, ஞ்ச -> nja
        out.append(_sound(u, k) + v)
    return "".join(out)


ENDINGS = {"", "ு", "ா", "ை", "ல", "ுல", "ில", "ில்", "லே", "ுக்கு", "க்கு", "ுங்க", "ங்க", "ுங்கய்யா", "ங்கய்யா",
           "ா", "ாங்க", "ாங்கய்யா", "ும்", "ுக்", "ோட", "ன்னு", "னு", "ாக", "ான்னு", "ுதான்", "ே", "ுமே", "ையும்",
           "த்துக்கு", "த்தை", "த்துல", "த்துலேயே", "த்தில்", "ம்", "மும்", "ங்கள்", "ங்களா", "க", "கள்", "ப்பா"}


def _ending_ok(rest: str) -> bool:
    return rest in ENDINGS or (rest[:1] in SIGNS and rest[1:] in ENDINGS)


def word_rules_en(word: str) -> str:
    for key in _LOAN_KEYS:
        if word.startswith(key) and (key in ASTRO or _ending_ok(word[len(key):])):
            rest = word[len(key):]
            # a vowel sign left over belongs to the key's last consonant (கரெக்டா -> correct + aa)
            suffix = (SIGNS[rest[0]] + rules(rest[1:])) if rest and rest[0] in SIGNS else rules(rest)
            if suffix == "aa":
                suffix = "a"  # people type "correct-a", "speed-a"
            if key in ASTRO:  # Tamil word: no hyphen before its suffix
                return LOANS[key] + suffix
            return LOANS[key] + ("-" + suffix if suffix else "")
    return rules(word)


def word_auto(word: str) -> str:
    """List first; otherwise the automatic English detector; otherwise sound rules."""
    for key in _LOAN_KEYS:
        if word.startswith(key) and (key in ASTRO or _ending_ok(word[len(key):])):
            return word_rules_en(word)
    from .english_detect import detect

    hit = detect(word)
    if hit:
        eng, suf = hit
        sfx = rules(suf) if suf and suf[0] not in SIGNS else (SIGNS[suf[0]] + rules(suf[1:]) if suf else "")
        return eng + ("-" + ("a" if sfx == "aa" else sfx) if sfx else "")
    return rules(word)


def plain(text: str) -> str:
    from aksharamukha import transliterate as T

    return T.process("Tamil", "RomanReadable", text).replace("'", "").replace("_", "")


def convert(text: str, method: str = "rules+en") -> str:
    if method == "plain":
        return plain(text)
    f = {"rules+en": word_rules_en, "auto": word_auto}.get(method, rules)
    return " ".join(f(w) for w in text.split())


if __name__ == "__main__":
    s = " ".join(sys.argv[1:]) or "வணக்கம் நாளைக்கு கிளாஸ் இருக்கு சார் தேங்க் யூ ரெண்டாவது எடுக்கணும் சந்திரன் பத்து வருஷம் மைண்ட்ல"
    for m in ["plain", "rules", "rules+en"]:
        print(f"{m:9s}: {convert(s, m)}")
