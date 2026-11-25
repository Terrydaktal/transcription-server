import re

from .config import DEFAULT_GRANITE_KEYWORDS_FILE
from .log_output import service_log as print


TECHNICAL_TEXT_REPLACEMENTS = {
    "origin slash main": "origin/main",
    "origin/slash main": "origin/main",
    "d89f d01": "D89FD01",
    "d89 fd01": "D89FD01",
    "d 89 f d 01": "D89FD01",
    "copy queue": "CopyQ",
    "copy q": "CopyQ",
    "copyq": "CopyQ",
    "as rock": "ASRock",
    "as roc": "ASRock",
    "asrock": "ASRock",
    "asroc": "ASRock",
    "x bind keys": "xbindkeys",
    "xbind keys": "xbindkeys",
    "ex bind keys": "xbindkeys",
    "expined key": "xbindkeys",
    "xbindkeys": "xbindkeys",
    "mod probe": "modprobe",
    "mob probe": "modprobe",
    "mod pro": "modprobe",
    "mob pro": "modprobe",
    "modpro": "modprobe",
    "nct 6683": "NCT6683",
    "n c t 6683": "NCT6683",
    "work trees": "worktrees",
    "work tree": "worktree",
    "stage diff": "staged diff",
    "kd": "KDE",
    "s h fmt": "shfmt",
    "ssh fmt": "shfmt",
    "sh fmt": "shfmt",
    "shell check": "ShellCheck",
    "s c two zero eight eight": "SC2088",
    "sc two zero eight eight": "SC2088",
    "s c 2088": "SC2088",
    "sc 2088": "SC2088",
    "bootstrap dot sh": "bootstrap.sh",
    "cachyos": "CachyOS",
}


def normalize_technical_text(text: str) -> str:
    result = text
    for spoken, canonical in sorted(
        TECHNICAL_TEXT_REPLACEMENTS.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        result = re.sub(
            rf"\b{re.escape(spoken)}\b",
            lambda _match, replacement=canonical: replacement,
            result,
            flags=re.IGNORECASE,
        )
    return result


def preserve_keyword_casing(text: str, keywords: list[str]) -> str:
    result = text
    for canonical in sorted(keywords, key=len, reverse=True):
        result = re.sub(
            rf"(?<!\w){re.escape(canonical)}(?!\w)",
            lambda _match, replacement=canonical: replacement,
            result,
            flags=re.IGNORECASE,
        )
    return result


def technical_canonical_terms() -> list[str]:
    terms = set(TECHNICAL_TEXT_REPLACEMENTS.values())
    if DEFAULT_GRANITE_KEYWORDS_FILE.is_file():
        terms.update(
            line.strip()
            for line in DEFAULT_GRANITE_KEYWORDS_FILE.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    return sorted(terms, key=len, reverse=True)


def canonicalize_technical_text(text: str) -> str:
    normalized = normalize_technical_text(text)
    return preserve_keyword_casing(normalized, technical_canonical_terms())


def protect_technical_terms(text: str) -> tuple[str, dict[str, str]]:
    protected = text
    replacements: dict[str, str] = {}
    patterns = [
        rf"(?<!\w){re.escape(term)}(?!\w)"
        for term in technical_canonical_terms()
    ]
    patterns.extend(
        (
            r"(?<!\w)[0-9A-F]{7,40}(?!\w)",
            r"(?<!\w)--?[A-Za-z0-9][A-Za-z0-9_-]*(?!\w)",
            r"(?<!\w)[A-Za-z0-9_~.-]*(?:/|\.|_)[A-Za-z0-9_~./-]+(?!\w)",
        )
    )
    combined = re.compile("|".join(f"(?:{pattern})" for pattern in patterns), re.IGNORECASE)

    def replace(match: re.Match) -> str:
        placeholder = f"ZXQTECHTOKEN{len(replacements)}QXZ"
        replacements[placeholder] = match.group(0)
        return placeholder

    return combined.sub(replace, protected), replacements


def restore_technical_terms(text: str, replacements: dict[str, str]) -> str:
    restored = text
    for placeholder, original in replacements.items():
        restored = restored.replace(placeholder, original)
    return restored


class EnglishVariantPostProcessor:
    def __init__(self, source: str, target: str) -> None:
        from english_variant_converter import convert

        self.source = source
        self.target = target
        self._convert = convert
        print(
            f"English variant conversion enabled: source='{self.source}' -> target='{self.target}'"
        )

    def convert_text(self, text: str) -> str:
        if not text.strip() or self.source == self.target:
            return text
        protected_text, protected_terms = protect_technical_terms(text)
        converted = self._convert(protected_text, source=self.source, target=self.target)
        return restore_technical_terms(converted, protected_terms)
