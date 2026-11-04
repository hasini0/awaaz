"""The citizen chatbot: the same engine, as a conversation that replies in the citizen's own language.

It asks only for what is missing: the location when the complaint names none, and the department when
the classifier is unsure (offering its top two guesses). Ticket numbers get a status reply.

UI-independent on purpose: the web chat page, the REST API (POST /chat) and a WhatsApp or IVR adapter
all drive the same `ChatBot.respond(conversation, message)`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from . import config
from .pipeline import Awaaz
from .text import detect_language, normalize, script_of

TICKET_ID = re.compile(r"\bAWZ-\d{4}-[0-9A-F]{6}\b", re.IGNORECASE)
GREETINGS = {  # greeting word -> language to answer in
    "hi": "en", "hello": "en", "hey": "en", "help": "en", "start": "en",
    "namaskaram": "tenglish", "namaste": "hinglish", "namaskar": "hinglish", "salaam": "hinglish",
    "salam": "hinglish", "adaab": "hinglish",
    "నమస్కారం": "te", "నమస్తే": "te", "नमस्ते": "hi", "नमस्कार": "hi", "آداب": "ur", "سلام": "ur",
    "السلام": "ur", "علیکم": "ur",
}
CHOICES = {"1": 0, "2": 1, "one": 0, "two": 1, "౧": 0, "౨": 1, "१": 0, "२": 1, "۱": 0, "۲": 1}
MAX_LOCATION_ASKS = 2  # after this many questions the complaint is filed without a location, for an officer to check


@dataclass
class Draft:
    """A complaint being collected over several messages."""
    text: str
    locality_id: str | None = None
    department: str | None = None
    options: list[str] = field(default_factory=list)  # departments offered when the model is unsure
    location_asks: int = 0  # how many times the bot has asked where the problem is


@dataclass
class Conversation:
    lang: str = "en"
    draft: Draft | None = None


@dataclass
class Reply:
    text: str
    ticket: dict | None = None  # the filed ticket, for UIs that want to show details


def say(key: str, lang: str, **values) -> str:
    """Message `key` in `lang`, falling back (e.g. romanised Telugu -> English) when not translated."""
    msgs = config.messages()
    for code in (lang, msgs["fallback"].get(lang), "en"):
        if code and key in msgs.get(code, {}):
            return msgs[code][key].format(**values)
    raise KeyError(key)


def _word(group: str, key: str, lang: str) -> str:
    msgs = config.messages()
    for code in (lang, msgs["fallback"].get(lang), "en"):
        if code and key in msgs.get(code, {}).get(group, {}):
            return msgs[code][group][key]
    return key


def _words(text: str) -> list[str]:
    return re.findall(r"[^\s.,!?।۔:;]+", text.lower())


class ChatBot:
    def __init__(self, engine: Awaaz):
        self.engine = engine

    def respond(self, conv: Conversation, message: str, channel: str = "chat") -> list[Reply]:
        """One citizen message in, the bot's replies out. Updates `conv`."""
        text = normalize(message)
        if not text:
            return [Reply(say("unheard", conv.lang))]
        self._follow_language(conv, text)
        words = _words(text)

        if m := TICKET_ID.search(text):
            return [self._status(m.group(0).upper(), conv.lang)]
        if words and all(w in GREETINGS for w in words):
            if conv.draft is None:
                conv.lang = GREETINGS[words[0]]
            return [Reply(say("greet", conv.lang))]
        if conv.draft and conv.draft.options:
            return self._on_department_answer(conv, text, words, channel)
        if conv.draft:
            return self._on_location_answer(conv, text, words, channel)
        if (len(words) >= 3 or self.engine.locator.find(text)) and not self._small_talk(text, words):
            conv.draft = Draft(text)
            return self._advance(conv, channel)
        return [Reply(say("greet", conv.lang))]  # "ok", "hi, what's up?"…: not a complaint, so explain the bot

    def _small_talk(self, text: str, words: list[str]) -> bool:
        """A short message that opens with a greeting, names no place and gives the models nothing to agree on
        ("Hi, how are you?") is conversation, not a complaint. "Hello, no water in Uppal" still counts."""
        return (len(words) <= 6 and words[0] in GREETINGS and not self.engine.locator.find(text)
                and self.engine.analyze(text).routing_unsure)

    def _on_department_answer(self, conv: Conversation, text: str, words: list[str], channel: str) -> list[Reply]:
        """The citizen was asked "1 or 2?"."""
        d = conv.draft
        choice = next((CHOICES[w] for w in words if w in CHOICES), None)
        if choice is not None:
            d.department = d.options[choice]
            return self._advance(conv, channel)
        if len(words) >= 4:  # not an answer but a new complaint
            conv.draft = None
            return self.respond(conv, text, channel)
        return [Reply(say("clarify", conv.lang, **self._option_names(d, conv.lang)))]

    def _on_location_answer(self, conv: Conversation, text: str, words: list[str], channel: str) -> list[Reply]:
        """The citizen was asked where the problem is."""
        d = conv.draft
        if loc := self.engine.locator.find(text):
            d.locality_id = loc.id
        elif len(words) >= 5:  # more detail rather than a place name: keep it, ask again
            d.text = f"{d.text} {text}"
        return self._advance(conv, channel)

    # -------------------------------------------------------------------------------------------
    @staticmethod
    def _follow_language(conv: Conversation, text: str):
        """Reply in the language the citizen writes in. Short replies ("1", "Kukatpally") keep the current
        language, unless they are in a native script, which identifies the language reliably."""
        if script_of(text) != "latin" or len(_words(text)) >= 3:
            conv.lang = detect_language(text)

    def _advance(self, conv: Conversation, channel: str) -> list[Reply]:
        """Ask for whatever is still missing; file the complaint once nothing is."""
        d = conv.draft
        tk = self.engine.analyze(d.text, d.locality_id, d.department)
        d.locality_id = d.locality_id or tk.locality_id
        if d.locality_id is None and d.location_asks < MAX_LOCATION_ASKS:
            d.location_asks += 1
            return [Reply(say("ask_location" if d.location_asks == 1 else "ask_location_again", conv.lang))]
        if d.department is None and tk.routing_unsure:
            d.options = self._top_two(tk)
            return [Reply(say("clarify", conv.lang, **self._option_names(d, conv.lang)))]

        tk = self.engine.submit(d.text, channel=channel, locality_id=d.locality_id, department=d.department)
        conv.draft = None
        return self._filed(tk.to_dict(), conv.lang)

    @staticmethod
    def _top_two(tk) -> list[str]:
        """The two departments to offer: the main model's pick, then the best different guess of any model."""
        ranked = [tk.department, *tk.second_opinions.values(),
                  *sorted(tk.dept_probs, key=tk.dept_probs.get, reverse=True)]
        return list(dict.fromkeys(ranked))[:2]

    @staticmethod
    def _option_names(d: Draft, lang: str) -> dict:
        a, b = (config.departments()[k] for k in d.options)
        return {"a": f"{a.icon} {a.name_in(lang)}", "b": f"{b.icon} {b.name_in(lang)}"}

    @staticmethod
    def _filed(tk: dict, lang: str) -> list[Reply]:
        dept = config.departments()[tk["department"]]
        lines = [say("filed", lang, id=tk["id"], dept=dept.name_in(lang), agency=dept.agency,
                     urgency=_word("urgency", tk["urgency"], lang), sla=dept.sla_hours)]
        if tk["reports"] > 1:
            lines.append(say("merged", lang, n=tk["reports"]))
        if not tk["locality_id"]:
            lines.append(say("no_location", lang))
        replies = [Reply("\n\n".join(lines), ticket=tk)]
        rules = {r["id"]: r for r in config.safety_rules()}
        for hit in tk["rule_hits"]:
            advice = rules[hit["rule_id"]].get("advice", {})
            text = advice.get(lang) or advice.get(config.messages()["fallback"].get(lang, "en")) or advice.get("en")
            if text:
                replies.append(Reply(say("safety", lang, advice=text)))
        return replies

    def _status(self, ticket_id: str, lang: str) -> Reply:
        row = self.engine.store.get(ticket_id)
        if not row:
            return Reply(say("not_found", lang, id=ticket_id))
        age = (datetime.now() - datetime.fromisoformat(row["created_at"])).total_seconds() / 3600
        dept = config.departments()[row["department"]]
        return Reply(say("status", lang, id=ticket_id, status=_word("statuses", row["status"], lang),
                         dept=f"{dept.icon} {dept.name_in(lang)}", where=row["locality"] or say("unknown_place", lang),
                         age=max(0, round(age))))
