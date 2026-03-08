"""
tests/test_source_reliability.py

Tests for Phase E: Source Reliability Ranking System

Run with:
    cd backend
    PYTHONPATH=. python3 tests/test_source_reliability.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.evidence.source_reliability import (
    extract_domain,
    calculate_source_trust,
    calculate_source_trust_by_name,
)
from app.services.evidence.evidence_ranker import (
    rank_evidence_sources,
    filter_evidence_sources,
    rank_and_filter,
)

# ── ANSI helpers ─────────────────────────────────────────────────────────────
GREEN = "\033[92m"
RED   = "\033[91m"
RESET = "\033[0m"
BOLD  = "\033[1m"

passed = failed = 0

def check(name: str, value, expected_min: float, expected_max: float):
    global passed, failed
    ok = expected_min <= value <= expected_max
    icon = f"{GREEN}✅{RESET}" if ok else f"{RED}❌{RESET}"
    status = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"  {icon} {name}")
    if not ok:
        print(f"       Expected [{expected_min}, {expected_max}]  Got: {value}")


def section(title: str):
    print(f"\n{BOLD}━━━━ {title} ━━━━{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
section("1. extract_domain()")
# ─────────────────────────────────────────────────────────────────────────────
cases = [
    ("https://www.bbc.com/news/world",          "bbc.com"),
    ("http://reuters.com/article/123",           "reuters.com"),
    ("https://en.wikipedia.org/wiki/NASA",       "en.wikipedia.org"),
    ("naturalnews.com/article",                  "naturalnews.com"),
    ("",                                          None),
]
for url, expected in cases:
    result = extract_domain(url)
    ok = result == expected
    icon = f"{GREEN}✅{RESET}" if ok else f"{RED}❌{RESET}"
    status = "PASS" if ok else f"FAIL (got {result!r})"
    if ok: passed += 1
    else:   failed += 1
    print(f"  {icon} extract_domain('{url[:45]}') → {result!r}")


# ─────────────────────────────────────────────────────────────────────────────
section("2. calculate_source_trust() — by URL")
# ─────────────────────────────────────────────────────────────────────────────
check("WHO (gov, https)",           calculate_source_trust("https://www.who.int/news"),         0.90, 1.00)
check("NASA (gov, https)",          calculate_source_trust("https://nasa.gov/article"),         0.90, 1.00)
check("BBC (high trust, https)",    calculate_source_trust("https://bbc.com/news"),             0.90, 1.00)
check("Reuters (high trust)",       calculate_source_trust("https://reuters.com/story"),        0.90, 1.00)
check("Wikipedia (medium trust)",   calculate_source_trust("https://en.wikipedia.org/wiki/X"), 0.70, 0.85)
check("CNN (medium trust)",         calculate_source_trust("https://cnn.com/2024/story"),       0.70, 0.85)
check("Nature (high trust)",        calculate_source_trust("https://nature.com/article"),      0.90, 1.00)
check("PolitiFact (+fact bonus)",   calculate_source_trust("https://politifact.com/fact"),     0.82, 1.00)
check("Snopes (+fact bonus)",       calculate_source_trust("https://snopes.com/fact"),         0.82, 1.00)
check("NaturalNews (low trust)",    calculate_source_trust("https://naturalnews.com/x"),        0.00, 0.20)
check("Infowars (low trust)",       calculate_source_trust("https://infowars.com/post"),        0.00, 0.20)
check("Unknown blog (no match)",    calculate_source_trust("https://randomblog123.xyz/post"),   0.30, 0.50)
check("HTTP penalty (no bonus)",    calculate_source_trust("http://bbc.com/news"),              0.85, 0.95)


# ─────────────────────────────────────────────────────────────────────────────
section("3. calculate_source_trust_by_name() — publisher name only")
# ─────────────────────────────────────────────────────────────────────────────
check("'NASA' by name",             calculate_source_trust_by_name("NASA"),           0.90, 1.00)
check("'BBC News' by name",         calculate_source_trust_by_name("BBC News"),       0.85, 1.00)
check("'Reuters' by name",          calculate_source_trust_by_name("Reuters"),        0.88, 1.00)
check("'Wikipedia' by name",        calculate_source_trust_by_name("Wikipedia"),      0.70, 0.80)
check("'Gizmodo.com' by name",      calculate_source_trust_by_name("Gizmodo"),        0.60, 0.75)
check("'NaturalNews' by name",      calculate_source_trust_by_name("NaturalNews"),    0.00, 0.20)
check("'Unknown Publisher'",        calculate_source_trust_by_name("Unknown Blog"),   0.30, 0.50)


# ─────────────────────────────────────────────────────────────────────────────
section("4. rank_evidence_sources() — ordering")
# ─────────────────────────────────────────────────────────────────────────────
evidence_list = [
    {"title": "Conspiracy cure", "source": "NaturalNews",  "url": "https://naturalnews.com/p"},
    {"title": "WHO bulletin",    "source": "WHO",           "url": "https://www.who.int/news"},
    {"title": "Blog post",       "source": "randomblog",    "url": "https://randomblog.xyz/x"},
    {"title": "BBC Report",      "source": "BBC",           "url": "https://bbc.com/news/1"},
    {"title": "Wiki article",    "source": "Wikipedia",     "url": "https://en.wikipedia.org/wiki/X"},
]
ranked = rank_evidence_sources("Dummy claim", evidence_list)
print(f"  Ranked order:")
for i, e in enumerate(ranked):
    print(f"    [{i+1}] {e['source']:20s} → trustScore={e['trustScore']}")

ok_order = ranked[0]["source"] in ("WHO", "BBC") and ranked[-1]["source"] in ("NaturalNews", "randomblog")
icon = f"{GREEN}✅{RESET}" if ok_order else f"{RED}❌{RESET}"
if ok_order: passed += 1
else:         failed += 1
print(f"  {icon} Top=trusted, Bottom=untrusted")


# ─────────────────────────────────────────────────────────────────────────────
section("5. filter_evidence_sources() — threshold 0.60")
# ─────────────────────────────────────────────────────────────────────────────
filtered = filter_evidence_sources(ranked, min_trust=0.60)
all_above_threshold = all(e["trustScore"] >= 0.60 for e in filtered)
icon = f"{GREEN}✅{RESET}" if all_above_threshold else f"{RED}❌{RESET}"
if all_above_threshold: passed += 1
else:                    failed += 1
print(f"  {icon} All {len(filtered)} filtered sources have trustScore ≥ 0.60")
for e in filtered:
    print(f"      ✓ {e['source']:20s} → {e['trustScore']}")

discarded = [e["source"] for e in ranked if e["trustScore"] < 0.60]
print(f"  Discarded: {discarded}")


# ─────────────────────────────────────────────────────────────────────────────
section("6. rank_and_filter() — max 5 sources")
# ─────────────────────────────────────────────────────────────────────────────
result = rank_and_filter("Dummy claim", evidence_list, min_trust=0.60, max_sources=5)
ok_cap = len(result) <= 5
icon = f"{GREEN}✅{RESET}" if ok_cap else f"{RED}❌{RESET}"
if ok_cap: passed += 1
else:       failed += 1
print(f"  {icon} Returned {len(result)} sources (max 5)")


# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'━'*52}")
total = passed + failed
color = GREEN if failed == 0 else RED
print(f"{color}{BOLD}  Results: {passed}/{total} tests passed  ({failed} failed){RESET}")
print(f"{'━'*52}\n")
