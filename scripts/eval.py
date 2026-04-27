"""
Evaluation harness for the agent.

Runs a fixed set of natural-language requests through the agent and
checks whether the recommendations match what the user asked for.
Prints a pass/fail summary plus a confidence score per case.

Run with:
    python -m scripts.eval
"""

import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent import run_agent

CASES = [
    {
        "name": "Pop workout",
        "input": "I want pop workout music, high energy",
        "checks": {
            "min_results": 3,
            "allowed_genres": ["pop", "edm", "hip-hop", "rock", "metal", "indie pop"],
            "min_avg_energy": 0.7,
        },
    },
    {
        "name": "Lofi study",
        "input": "give me lofi study music",
        "checks": {
            "min_results": 3,
            "allowed_genres": ["lofi", "ambient", "jazz", "classical"],
            "max_avg_energy": 0.6,
        },
    },
    {
        "name": "Sad late night",
        "input": "sad moody music for late night",
        "checks": {
            "min_results": 2,
            "allowed_moods": ["moody", "chill", "relaxed"],
        },
    },
    {
        "name": "Acoustic folk",
        "input": "I want acoustic folk or country songs",
        "checks": {
            "min_results": 1,
            "allowed_genres": ["folk", "country", "classical"],
        },
    },
    {
        "name": "No acoustic chill",
        "input": "chill music but no acoustic stuff",
        "checks": {
            "min_results": 1,
            "no_acoustic": True,
        },
    },
    {
        "name": "Latin happy",
        "input": "happy latin music",
        "checks": {
            "min_results": 1,
            "allowed_genres": ["latin", "pop", "indie pop"],
            "allowed_moods": ["happy"],
        },
    },
    {
        "name": "Empty input",
        "input": "",
        "checks": {
            "expect_refusal": True,
        },
    },
    {
        "name": "Intense rock",
        "input": "intense rock or metal, no chill stuff",
        "checks": {
            "min_results": 2,
            "allowed_genres": ["rock", "metal", "edm"],
            "min_avg_energy": 0.7,
        },
    },
]


def get_recommendations(trace: List[Dict]) -> List[Dict]:
    last = []
    for step in trace:
        if step.get("type") == "tool_result" and step.get("name") == "recommend_songs":
            if isinstance(step.get("result"), list):
                last = step["result"]
    return last


def count_retries(trace: List[Dict]) -> int:
    retries = 0
    for step in trace:
        if step.get("type") == "tool_result" and step.get("name") == "validate_results":
            issues = step.get("result", {}).get("issues", [])
            if issues and not step["result"].get("ok"):
                retries += 1
    return retries


def confidence(retries: int) -> float:
    if retries == 0:
        return 1.0
    if retries == 1:
        return 0.7
    if retries == 2:
        return 0.4
    return 0.2


def check_case(case: Dict, result: Dict) -> Dict:
    checks = case["checks"]
    failures = []

    if checks.get("expect_refusal"):
        ans = (result.get("answer") or "").lower()
        if "tell me" not in ans and "what kind" not in ans:
            failures.append("expected a refusal/clarifier for empty input")
        return {"failures": failures, "recs": [], "retries": 0}

    recs = get_recommendations(result.get("trace", []))
    retries = count_retries(result.get("trace", []))

    if "min_results" in checks and len(recs) < checks["min_results"]:
        failures.append(f"only {len(recs)} results, expected >= {checks['min_results']}")

    if "allowed_genres" in checks and recs:
        bad = [r for r in recs if r.get("genre") not in checks["allowed_genres"]]
        if len(bad) > len(recs) // 2:
            failures.append(f"too many off-genre picks: {[r['title'] + '/' + r['genre'] for r in bad]}")

    if "allowed_moods" in checks and recs:
        bad = [r for r in recs if r.get("mood") not in checks["allowed_moods"]]
        if len(bad) > len(recs) // 2:
            failures.append(f"too many off-mood picks: {[r['title'] + '/' + r['mood'] for r in bad]}")

    if "min_avg_energy" in checks and recs:
        avg = sum((r.get("energy") or 0) for r in recs) / len(recs)
        if avg < checks["min_avg_energy"]:
            failures.append(f"avg energy {avg:.2f} below min {checks['min_avg_energy']}")

    if "max_avg_energy" in checks and recs:
        avg = sum((r.get("energy") or 0) for r in recs) / len(recs)
        if avg > checks["max_avg_energy"]:
            failures.append(f"avg energy {avg:.2f} above max {checks['max_avg_energy']}")

    if checks.get("no_acoustic") and recs:
        bad = [r["title"] for r in recs if (r.get("acousticness") or 0) > 0.6]
        if bad:
            failures.append(f"acoustic songs leaked through: {bad}")

    return {"failures": failures, "recs": recs, "retries": retries}


def main():
    print("=" * 70)
    print("Music Agent Evaluation")
    print("=" * 70)

    passed = 0
    total_conf = 0.0
    rows = []

    for i, case in enumerate(CASES, 1):
        print(f"\n[{i}/{len(CASES)}] {case['name']}: {case['input']!r}")
        try:
            result = run_agent(case["input"])
        except Exception as e:
            print(f"  ERROR: {e}")
            rows.append((case["name"], "ERROR", 0.0, str(e)))
            continue

        check = check_case(case, result)
        ok = len(check["failures"]) == 0
        conf = confidence(check["retries"])
        if ok:
            passed += 1
        total_conf += conf

        status = "PASS" if ok else "FAIL"
        notes = "; ".join(check["failures"]) if check["failures"] else f"{len(check['recs'])} recs, {check['retries']} retries"
        print(f"  -> {status}  confidence={conf:.2f}  {notes}")
        rows.append((case["name"], status, conf, notes))

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{passed}/{len(CASES)} cases passed")
    print(f"average confidence: {total_conf / len(CASES):.2f}")
    print()
    print(f"{'Case':<20} {'Status':<8} {'Conf':<6} {'Notes'}")
    print("-" * 70)
    for name, status, conf, notes in rows:
        short_notes = notes if len(notes) < 40 else notes[:37] + "..."
        print(f"{name:<20} {status:<8} {conf:<6.2f} {short_notes}")


if __name__ == "__main__":
    main()
