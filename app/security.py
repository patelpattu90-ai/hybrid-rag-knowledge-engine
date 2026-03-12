"""
security.py — Prompt Injection Defense + PII Scrubbing

Two responsibilities:

  1. InputSanitiser   — strips prompt injection attempts from user input
                        BEFORE it reaches any LLM or retrieval call.

  2. PIIScrubber      — redacts PII patterns from text.

IMPORTANT DESIGN DECISION — Two separate pattern sets:
  _CONTEXT_PII_PATTERNS  → applied to retrieved DOC CHUNKS (conservative)
                           Only redacts things that CANNOT appear in FastAPI docs:
                           real emails, real credit cards, real SSNs.
                           Does NOT redact API keys or IPs — these appear
                           legitimately in code examples throughout the docs.

  _ANSWER_PII_PATTERNS   → applied to GENERATED ANSWERS (stricter)
                           Adds API key + JWT patterns since the LLM should not
                           be echoing real credentials back to users.

This two-tier approach prevents the scrubber from mangling documentation
context and causing "I could not find the answer" failures.
"""

import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. INPUT SANITISER — Prompt Injection Defense
# ══════════════════════════════════════════════════════════════════════════════

_INJECTION_PATTERNS = [
    # Role/instruction override attempts
    (r"(?i)\[system\].*?(\n|$)",              ""),
    (r"(?i)<\|system\|>.*?(\n|$)",            ""),
    (r"(?i)<\|im_start\|>.*?<\|im_end\|>",   ""),
    (r"(?i)\[inst\].*?\[/inst\]",             ""),
    (r"(?i)###\s*instruction[s]?:.*?(\n|$)",  ""),
    (r"(?i)###\s*system:.*?(\n|$)",           ""),

    # Embedded newline escape tricks
    (r"\\n\\n(system|assistant|user):",       " "),
    (r"\n{3,}",                               "\n\n"),

    # HTML/script tags
    (r"<script[^>]*>.*?</script>",            ""),
    (r"<[a-zA-Z][^>]{0,200}>",               ""),
]

_MAX_INPUT_CHARS = 1000


@dataclass
class SanitiseResult:
    original:     str
    sanitised:    str
    was_modified: bool
    changes:      list[str] = field(default_factory=list)

    def badge(self) -> str:
        if not self.was_modified:
            return "🔐 **Security:** Input clean"
        return f"🔐 **Security:** {len(self.changes)} injection pattern(s) stripped"


class InputSanitiser:
    def __init__(self):
        self._patterns = [
            (re.compile(pat, re.DOTALL), repl)
            for pat, repl in _INJECTION_PATTERNS
        ]

    def sanitise(self, text: str) -> SanitiseResult:
        original  = text
        sanitised = text
        changes   = []

        if len(sanitised) > _MAX_INPUT_CHARS:
            sanitised = sanitised[:_MAX_INPUT_CHARS]
            changes.append(f"Truncated to {_MAX_INPUT_CHARS} chars")

        for pattern, replacement in self._patterns:
            new_text = pattern.sub(replacement, sanitised)
            if new_text != sanitised:
                changes.append(f"Stripped: {pattern.pattern[:40]}")
                sanitised = new_text

        sanitised     = sanitised.strip()
        was_modified  = sanitised != original.strip()

        if was_modified:
            logger.warning(f"Input sanitised. Changes: {changes}")

        return SanitiseResult(
            original=original,
            sanitised=sanitised,
            was_modified=was_modified,
            changes=changes,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 2. PII SCRUBBER — Two-tier pattern sets
# ══════════════════════════════════════════════════════════════════════════════

# ── CONSERVATIVE — safe to apply to documentation chunks ─────────────────────
# Only patterns that CANNOT appear in legitimate FastAPI documentation.
# No API key pattern here — long alphanumeric strings ARE normal in docs
# (base64 examples, hash values, token examples, code identifiers).
# No broad IP pattern here — IPs appear in every FastAPI "run uvicorn" example.
_CONTEXT_PII_PATTERNS = [
    # Real email addresses (not example.com / test addresses)
    ("email",
     r"\b[A-Za-z0-9._%+\-]+@(?!example\.com|test\.com|domain\.com)[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
     "[EMAIL REDACTED]"),

    # Credit card numbers (13-16 digits with separators — very unlikely in docs)
    ("credit_card",
     r"\b(?:\d[ \-]?){13,15}\d\b",
     "[CARD REDACTED]"),

    # US SSN (XXX-XX-XXXX format — never in API docs)
    ("ssn",
     r"\b\d{3}-\d{2}-\d{4}\b",
     "[SSN REDACTED]"),
]

# ── STRICT — applied to generated answers only ────────────────────────────────
# Adds API keys and JWTs — the LLM should never echo real credentials.
_ANSWER_PII_PATTERNS = _CONTEXT_PII_PATTERNS + [
    # API keys / secrets (long random alphanumeric 40+ chars, not code identifiers)
    # Require a key-like prefix to avoid matching normal long words
    ("api_key",
     r"(?i)\b(sk-|pk-|api[-_]?key[-_]?)[A-Za-z0-9_\-]{20,}\b",
     "[KEY REDACTED]"),

    # JWT tokens (three base64url segments — very distinctive format)
    ("jwt",
     r"\b[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b",
     "[TOKEN REDACTED]"),

    # Private/non-public IP addresses in answers (not localhost/example ranges)
    ("private_ip",
     r"\b(?!127\.|0\.|192\.168\.|10\.|172\.(1[6-9]|2\d|3[01])\.)(?:\d{1,3}\.){3}\d{1,3}\b",
     "[IP REDACTED]"),
]

# Whitelist — always preserve these even if they match a pattern above
_WHITELIST_PATTERNS = [
    r"example\.com",
    r"user@",
    r"test@",
    r"admin@",
    r"127\.0\.0\.1",
    r"0\.0\.0\.0",
    r"localhost",
    r"192\.168\.",
    r"your[-_]?api[-_]?key",   # placeholder text in docs
    r"<your",                   # <your-token> style placeholders
]


@dataclass
class ScrubResult:
    original:     str
    scrubbed:     str
    redacted:     list[str]
    was_modified: bool


class PIIScrubber:
    def __init__(self):
        self._context_patterns = [
            (label, re.compile(pattern, re.IGNORECASE), replacement)
            for label, pattern, replacement in _CONTEXT_PII_PATTERNS
        ]
        self._answer_patterns = [
            (label, re.compile(pattern, re.IGNORECASE), replacement)
            for label, pattern, replacement in _ANSWER_PII_PATTERNS
        ]
        self._whitelist = [
            re.compile(p, re.IGNORECASE) for p in _WHITELIST_PATTERNS
        ]

    def _is_whitelisted(self, match_text: str) -> bool:
        return any(w.search(match_text) for w in self._whitelist)

    def _apply_patterns(self, text: str, patterns: list) -> ScrubResult:
        scrubbed = text
        redacted = []

        for label, pattern, replacement in patterns:
            def replace_fn(m, repl=replacement, lbl=label):
                if self._is_whitelisted(m.group(0)):
                    return m.group(0)
                return repl

            new_text = pattern.sub(replace_fn, scrubbed)
            if new_text != scrubbed:
                redacted.append(label)
                scrubbed = new_text

        was_modified = scrubbed != text
        if was_modified:
            logger.info(f"PII scrubbed ({len(redacted)} types): {redacted}")

        return ScrubResult(
            original=text,
            scrubbed=scrubbed,
            redacted=redacted,
            was_modified=was_modified,
        )

    def scrub_contexts(self, contexts: list[str]) -> list[str]:
        """
        Conservative scrub — applied to retrieved doc chunks.
        Only removes PII that definitely cannot be in documentation.
        Does NOT touch API key patterns or broad IP ranges.
        """
        return [self._apply_patterns(c, self._context_patterns).scrubbed for c in contexts]

    def scrub_answer(self, answer: str) -> str:
        """
        Strict scrub — applied to LLM-generated answer.
        Removes all PII including API keys, JWTs, and non-private IPs.
        """
        return self._apply_patterns(answer, self._answer_patterns).scrubbed


# ── Singletons ────────────────────────────────────────────────────────────────
input_sanitiser = InputSanitiser()
pii_scrubber    = PIIScrubber()
