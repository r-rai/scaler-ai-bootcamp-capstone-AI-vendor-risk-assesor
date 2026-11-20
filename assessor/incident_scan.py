"""
incident_scan.py — which search results report a security incident involving the vendor (SEC-9.3, `searched` evidence).

No model, on purpose: this runs inside the matcher's verdict, so the same stored results always give the same verdict and a saved
profile can be re-judged without a new search.

A result FLAGS when it (1) names the vendor (title, snippet or URL), and (2) uses event wording: breach, hack, leak, ransomware,
vulnerability, exploit, unauthorized access, post-mortem, ... It is deliberately conservative: whatever the date, a result that names the
vendor next to incident wording is something a person must read before an approval. What does NOT flag, because it is not an incident
report: the contract and SLA senses of "breach", "incident response/management" (a plan, not an event), and the vendor's own terms /
privacy / legal pages (boilerplate that promises breach notification). The same words anywhere else, including a vendor's own security
blog, still flag.

What this cannot do: see an incident the search did not return, or one worded in a way the list below misses. That is why a PASS built on
it is labelled `searched` and never "proof of absence".
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

MIN_TOKEN_LEN = 4
GENERIC_LABELS = frozenset({"www", "help", "docs", "trust", "status", "support", "blog", "api", "app", "apps", "learn", "cloud",
                            "security", "developers", "developer", "community", "legal"})

# Event wording. Every alternative stands for something that HAPPENED to a company or its software, not for a policy.
_INCIDENT = re.compile(
    r"\bbreach(?:ed|es)?\b|\bhack(?:ed|er|ers|ing)?\b|\bleak(?:ed|s|age)?\b|\bcompromis(?:e|ed|es|ing)\b|\bransomware\b"
    r"|\bcyber[- ]?attacks?\b|\battack(?:s|ed|ers?)?\b|\bexploit(?:ed|s|ation|ing)?\b|\bzero[- ]?days?\b"
    r"|\bvulnerabilit(?:y|ies)\b|\bflaws?\b|\bcve-\d{4}-\d{3,}\b|\bunauthori[sz]ed\b|\bincidents?\b|\bpost-?mortems?\b"
    r"|\bsecurity (?:advisory|bulletin|notice|alert|event)s?\b|\bexpos(?:ed|ure|ing)\b|\bphishing\b|\baccount takeovers?\b"
    r"|\bstol(?:e|en)\b|\bgained access\b|\bintrusions?\b|\bexfiltrat\w*", re.I)

# Wording that looks like the above but is not a report of an event.
_BENIGN = re.compile(
    r"\bbreach(?:es)? of (?:the |this |these |our |any )?(?:contract|agreement|terms|warranty|duty|confidence|trust)\b|\bmaterial breach(?:es)?\b"
    r"|\bsl[ao] breach(?:es)?\b|\bincident (?:response|management|handling|process|plan|policy|readiness)\b", re.I)

# Pages of the vendor's own site that are contract or policy text, not reports.
_OWN_POLICY_PATH = re.compile(r"/(?:terms|privacy|legal|polic(?:y|ies)|dpa|gdpr|tos|subprocessors?)(?:[/._-]|$)", re.I)


def vendor_tokens(vendor: str, official_domain: str | None) -> list[str]:
    """The words that mean "this vendor" in a search result: the first word of its name and the distinctive labels of its domain.
    Short words and generic sub-domain labels ("help", "docs") are not names."""
    out: list[str] = []
    words = re.findall(r"[a-z0-9]+", (vendor or "").lower())
    if words and len(words[0]) >= MIN_TOKEN_LEN:
        out.append(words[0])
    labels = (official_domain or "").lower().split(".")
    for label in labels[:-1] if len(labels) > 1 else labels:      # the last label is the TLD, never a name
        if len(label) >= MIN_TOKEN_LEN and label not in GENERIC_LABELS and label not in out:
            out.append(label)
    return out


def _squash(text: str) -> str:
    """Lower-case, every run of non-alphanumerics a single space, padded: ' notion ai ' matches 'Notion  AI's' and 'notion.so/x'."""
    return " " + re.sub(r"[^a-z0-9]+", " ", text.lower()).strip() + " "


def vendor_phrases(vendor: str, official_domain: str | None) -> list[str]:
    """What identifies THIS company in a result beyond one ordinary word: the first two words of its name and its domain.
    A one-word name is an ordinary word on its own ("Lorikeet", "Notion"), so then only the domain remains."""
    words = re.findall(r"[a-z0-9]+", (vendor or "").lower())
    out = [" ".join(words[:2])] if len(words) >= 2 else []
    if official_domain:
        out.append(official_domain.lower())
    return out


def is_anchored(result: dict, phrases: list[str], official_domain: str | None) -> bool:
    """Is this result unmistakably about the vendor: on its own host, or containing its two-word name or its domain? Naming the vendor
    (`scan_results`) is enough to FLAG a result, because a wrong flag only asks a person to look. It is not enough to COUNT as proof
    that the search covered the vendor: two articles that merely contain the word "lorikeet" say nothing about a company of that name."""
    url = str(result.get("url") or "")
    if _on_domain(url, official_domain):
        return True
    raw = f"{result.get('title') or ''} {result.get('snippet') or ''} {url}".lower()
    squashed = _squash(raw)
    for p in phrases:
        if not p:
            continue
        if "." in p:    # a domain: literal dots, and not the front of a longer host ("lorikeet.ai.example")
            if re.search(r"(?<![a-z0-9.-])" + re.escape(p) + r"(?![a-z0-9-]|\.[a-z0-9])", raw):
                return True
        elif _squash(p) in squashed:
            return True
    return False


def _names_vendor(text: str, tokens: list[str]) -> bool:
    low = text.lower()
    return any(re.search(r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])", low) for t in tokens)


def _on_domain(url: str, domain: str | None) -> bool:
    host = urlparse(url).netloc.lower().split(":")[0]
    d = (domain or "").lower()
    return bool(d) and (host == d or host.endswith("." + d))


def reports_incident(title: str, snippet: str) -> bool:
    """Event wording in a title or snippet, after the contract / SLA / response-plan senses are removed."""
    return bool(_INCIDENT.search(_BENIGN.sub(" ", f"{title} {snippet}")))


def scan_results(results: list[dict], tokens: list[str], official_domain: str | None) -> tuple[list[dict], list[dict]]:
    """(results that name the vendor, the subset of those that report an incident). Each URL counts once."""
    named: list[dict] = []
    flagged: list[dict] = []
    seen: set[str] = set()
    for r in results:
        url = str(r.get("url") or "")
        if url in seen:
            continue
        seen.add(url)
        title, snippet = str(r.get("title") or ""), str(r.get("snippet") or "")
        if not _names_vendor(f"{title} {snippet} {url}", tokens):
            continue
        named.append(r)
        if _on_domain(url, official_domain) and _OWN_POLICY_PATH.search(urlparse(url).path):
            continue
        if reports_incident(title, snippet):
            flagged.append(r)
    return named, flagged
