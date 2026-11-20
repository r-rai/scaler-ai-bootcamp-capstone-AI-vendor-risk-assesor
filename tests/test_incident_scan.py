"""Which search results count as "this result reports a security incident involving the vendor" (SEC-9.3, `searched` evidence).

The headline and snippet lists below were written BEFORE the word list in assessor/incident_scan.py (pre-registration F1,
reports/sec93_20261120/preregistration.md): every incident headline must be flagged, no benign one may be. They are synthetic
("Acme"); none is copied from a real search result. The scan is deliberately conservative: a result that names the vendor next to
incident wording flags, whatever its date, because a human must read it before an approval.
"""
import pytest

from assessor.incident_scan import is_anchored, scan_results, vendor_phrases, vendor_tokens

DOMAIN = "acme.example"
TOKENS = ["acme"]


def hit(title, snippet="", url="https://news.example/story"):
    return {"url": url, "title": title, "snippet": snippet}


INCIDENTS = [
    "Acme confirms data breach affecting customer accounts",
    "Acme hacked: attackers stole customer records",
    "Hackers breach Acme storage, exposing chat transcripts",
    "Acme discloses security incident involving unauthorized access",
    "Acme ransomware attack disrupts service for days",
    "Acme zero-day vulnerability exploited in the wild",
    "CVE-2026-1234 in Acme Desk allows remote code execution",
    "Acme post-mortem: how an attacker reached our production database",
    "Acme customer data leaked on a hacking forum",
    "Acme says some customer data was exposed after a misconfiguration",
    "Acme security advisory: customers urged to rotate API keys",
    "Acme investigating suspected compromise of internal systems",
    "Phishing campaign abused Acme accounts to reach customers",
    "Acme was the victim of a cyberattack last month",
    "Acme notifies users of account takeover wave",
    "Stolen credentials used to access Acme support system",
    "Attackers gained access to Acme's source code repository",
    "Acme incident report: unauthorized access to support tooling",
    "Acme data exposure: researchers found an open storage bucket",
    "Acme vulnerability disclosure: prompt injection lets attackers exfiltrate data",
    "Acme impacted by supply-chain attack on a dependency",
    "Acme notifies regulators of personal data breach",
    "Acme security incident post-mortem published",
    "Lawsuit filed over Acme data breach",
    "Acme AI data leak: what we know so far",
    "Acme's Q3 incident: attackers accessed internal chat",
    "Researchers disclose flaw letting anyone read Acme tenants' data",
    "Acme customers told to reset passwords after intrusion",
    "Acme patches zero-day in Desk",                      # added after the mutation check: the one word of its own
]

# Each alternative of the word list must be able to flag ON ITS OWN (found by removing them one at a time: nine were never the deciding
# word of the headlines above). One event word each, nothing else that would flag.
SINGLE_WORD = [
    "Acme hacked again", "Acme hit by ransomware", "Criminals exploited a bug in Acme", "Acme vulnerability found by researchers",
    "Unauthorized logins on Acme tenants", "Acme incident update", "Acme post-mortem", "Someone gained access to Acme's staging server",
    "Acme data exfiltrated overnight",
]

# (title, snippet, url): not security-incident reports, or not about the vendor, or the vendor's own boilerplate
BENIGN = [
    ("Acme Terms of Service", "Either party may terminate for material breach of this agreement.", "https://acme.example/legal/terms"),
    ("Can Acme agents prevent SLA breaches?", "Support teams use Acme to keep first response inside the SLA.", "https://video.example/watch"),
    ("Acme launches new AI assistant for support teams", "The assistant drafts replies and summarises tickets.", "https://news.example/a"),
    ("Acme pricing and plans", "Compare the Starter, Business and Enterprise plans.", "https://acme.example/pricing"),
    ("Acme security overview", "Our incident response plan is tested twice a year and we hold a SOC 2 Type II report.", "https://acme.example/security"),
    ("Top 10 data breaches of 2026", "A look at the largest breaches of the year and what companies learned.", "https://news.example/b"),
    ("Acme raises $50M Series B", "The round was led by Example Ventures.", "https://news.example/c"),
    ("Acme Privacy Policy", "We will notify you of a personal data breach where the law requires us to.", "https://acme.example/privacy"),
    ("Acme vs Competitor: feature comparison", "Which tool fits a 30-person support team?", "https://review.example/acme-vs"),
    ("Acme Trust Center", "Security, privacy and compliance documentation for customers and auditors.", "https://trust.acme.example/"),
    ("Court dismisses breach of contract claim between two suppliers", "Acme was named only as a customer of one party.", "https://news.example/d"),
    ("Globex hit by ransomware", "The attack on Globex halted production at three plants.", "https://news.example/e"),
]


@pytest.mark.parametrize("title", INCIDENTS)
def test_an_incident_headline_that_names_the_vendor_is_flagged(title):
    named, flagged = scan_results([hit(title)], TOKENS, DOMAIN)
    assert len(named) == 1 and len(flagged) == 1


@pytest.mark.parametrize("title", SINGLE_WORD)
def test_every_event_word_can_flag_on_its_own(title):
    assert len(scan_results([hit(title)], TOKENS, DOMAIN)[1]) == 1


@pytest.mark.parametrize("title, snippet, url", BENIGN)
def test_benign_results_are_not_flagged(title, snippet, url):
    _, flagged = scan_results([hit(title, snippet, url)], TOKENS, DOMAIN)
    assert flagged == []


def test_the_incident_wording_may_sit_in_the_snippet():
    _, flagged = scan_results([hit("Acme news", "Acme said attackers had stolen a database of customer emails.")], TOKENS, DOMAIN)
    assert len(flagged) == 1


def test_an_incident_report_about_another_company_does_not_flag_the_vendor():
    named, flagged = scan_results([hit("Globex confirms data breach", "Attackers stole Globex customer records.")], TOKENS, DOMAIN)
    assert named == [] and flagged == []


def test_the_vendor_may_be_named_only_in_the_url():
    named, flagged = scan_results([hit("Breach disclosure", "Attackers stole records.", "https://blog.example/acme-breach-disclosure")], TOKENS, DOMAIN)
    assert len(named) == 1 and len(flagged) == 1


def test_the_vendors_own_security_notice_is_still_an_incident_report():
    """Only terms / privacy / legal pages are boilerplate. A post-mortem on the vendor's own site must flag."""
    _, flagged = scan_results([hit("Investigating unauthorized access to our systems", "Post-mortem and customer guidance.",
                                   "https://acme.example/blog/security-incident")], TOKENS, DOMAIN)
    assert len(flagged) == 1


def test_a_policy_page_on_the_vendors_domain_is_boilerplate_but_the_same_words_elsewhere_are_not():
    page = ("Data Processing Addendum", "Processor shall notify Controller of a personal data breach without undue delay.")
    assert scan_results([hit(*page, url="https://acme.example/legal/dpa")], TOKENS, DOMAIN)[1] == []
    assert len(scan_results([hit(*page, url="https://evil-copy.example/acme/dpa")], TOKENS, DOMAIN)[1]) == 1   # not the vendor's domain


def test_results_are_reported_once_whatever_the_query_that_found_them():
    dup = [hit("Acme confirms data breach"), hit("Acme confirms data breach")]
    named, flagged = scan_results(dup, TOKENS, DOMAIN)
    assert len(named) == 1 and len(flagged) == 1


@pytest.mark.parametrize("vendor, domain, expected", [
    ("Zoho Desk Zia", "zoho.com", ["zoho"]),
    ("Ringg AI", "ringg.ai", ["ringg"]),
    ("LiveKit Cloud", "livekit.io", ["livekit"]),
    ("Acme AI", "help.acme.co.uk", ["acme"]),                 # generic sub-domain labels and short labels are not names
    ("Notion AI", None, ["notion"]),
    ("AI Co", None, []),                                      # nothing distinctive to look for
    ("Foo Bar", "foobar.software", ["foobar"]),                  # the last label is the TLD, never a name
    ("Acme AI", "acmeai.example", ["acme", "acmeai"]),
])
def test_vendor_tokens(vendor, domain, expected):
    assert vendor_tokens(vendor, domain) == expected


@pytest.mark.parametrize("vendor, domain, expected", [
    ("Notion AI", "notion.so", ["notion ai", "notion.so"]),
    ("Zoho Desk Zia", "zoho.com", ["zoho desk", "zoho.com"]),
    ("Anthropic", "anthropic.com", ["anthropic.com"]),          # a one-word name is an ordinary word on its own: the domain only
    ("Lorikeet CX", None, ["lorikeet cx"]),
])
def test_vendor_phrases(vendor, domain, expected):
    assert vendor_phrases(vendor, domain) == expected


@pytest.mark.parametrize("title, snippet, url, anchored", [
    ("Rainbow lorikeet care guide", "Feeding a lorikeet at home.", "https://birds.example/lorikeet", False),
    ("Lorikeet diet", "Nectar and fruit.", "https://pets.example/diet", False),
    ("Lorikeet CX raises a round", "", "https://news.example/a", True),                   # the two-word name
    ("Customer story", "How a team cut handle time with Lorikeet  CX's agent", "https://news.example/b", True),   # spacing and 's
    ("Trust centre", "", "https://trust.lorikeet.ai/", True),                                # on the vendor's own host
    ("Security", "", "https://lorikeet.ai/security", True),
    ("Docs mirror", "see lorikeet.ai/docs for more", "https://mirror.example/x", True),    # the domain written in the text
    ("A lorikeet.ai.example scam", "", "https://other.example/y", False),                    # not the domain, a look-alike
])
def test_is_anchored(title, snippet, url, anchored):
    phrases = vendor_phrases("Lorikeet CX", "lorikeet.ai")
    assert is_anchored({"url": url, "title": title, "snippet": snippet}, phrases, "lorikeet.ai") is anchored

