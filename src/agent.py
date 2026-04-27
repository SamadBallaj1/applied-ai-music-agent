"""
Agent that wraps the rule-based recommender.

Reads a natural language request, picks taste preferences, calls
recommend_songs, checks the results, and retries if something is off.
"""

import json
import logging
import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

try:
    from recommender import load_songs, recommend_songs
except ModuleNotFoundError:
    from src.recommender import load_songs, recommend_songs

load_dotenv()

MODEL = "gpt-4o-mini"

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "agent.log")


def _setup_logger() -> logging.Logger:
    log = logging.getLogger("music_agent")
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(LOG_FILE)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    log.propagate = False
    return log


logger = _setup_logger()

SYSTEM_PROMPT = """You are a music recommendation assistant.

You have two tools:
- recommend_songs: scores the catalog and returns top picks
- validate_results: checks the picks against the user's constraints and flags issues

Workflow for every request:
1. Read the user's words and pick taste preferences. Pull out hard constraints
   (e.g. "no acoustic" -> exclude_acoustic=true, "low energy" -> max_energy=0.4).
2. Call recommend_songs with those preferences and constraints.
3. Call validate_results with the same user message and the picks you got back.
4. If validate_results returns any issues, adjust your preferences or constraints
   and call recommend_songs again. Then validate again.
5. Once validation passes (or after one retry), write a short, friendly answer
   that names the top picks and explains why they fit, in plain English.

Catalog:
- Genres: pop, lofi, rock, ambient, jazz, synthwave, indie pop, hip-hop, r&b,
  country, classical, edm, latin, metal, folk
- Moods: happy, chill, intense, relaxed, focused, moody
- Energy, danceability, valence are floats from 0.0 to 1.0.

If the user says "study" or "focus", treat that as mood=focused or chill, low-mid energy.
If the user says "workout" or "gym", treat that as high energy and intense or happy mood.
Always call recommend_songs. Never invent songs.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "recommend_songs",
            "description": "Score the catalog against the user's preferences and return the top K songs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "genre": {"type": "string"},
                    "mood": {"type": "string"},
                    "energy": {"type": "number", "minimum": 0, "maximum": 1},
                    "danceability": {"type": "number", "minimum": 0, "maximum": 1},
                    "valence": {"type": "number", "minimum": 0, "maximum": 1},
                    "likes_acoustic": {"type": "boolean"},
                    "k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                    "mode": {
                        "type": "string",
                        "enum": ["balanced", "genre-first", "mood-first", "energy-focused"],
                        "default": "balanced",
                    },
                    "diversity": {"type": "boolean", "default": True},
                    "exclude_acoustic": {
                        "type": "boolean",
                        "default": False,
                        "description": "Hard filter: drop any song with acousticness > 0.6.",
                    },
                    "min_energy": {"type": "number", "minimum": 0, "maximum": 1},
                    "max_energy": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["genre", "mood", "energy"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_results",
            "description": "Check if the recommended songs match the user's request. Returns a list of issues to fix.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_message": {"type": "string"},
                    "results": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "The list returned by recommend_songs.",
                    },
                },
                "required": ["user_message", "results"],
            },
        },
    },
]


class MusicAgent:
    MAX_RETRIES = 2

    def __init__(self, songs: List[Dict], model: str = MODEL):
        self.client = OpenAI()
        self.songs = songs
        self.model = model
        self._last_results: List[Dict] = []
        self._retry_count = 0

    def _filter(self, results: list, args: dict) -> list:
        out = []
        for s, score, why in results:
            if args.get("exclude_acoustic") and s.get("acousticness", 0) > 0.6:
                continue
            if "min_energy" in args and s.get("energy", 0) < args["min_energy"]:
                continue
            if "max_energy" in args and s.get("energy", 0) > args["max_energy"]:
                continue
            out.append((s, score, why))
        return out

    def _recommend(self, args: dict) -> List[Dict]:
        user_prefs = {
            k: v
            for k, v in args.items()
            if k in ("genre", "mood", "energy", "danceability", "valence", "likes_acoustic")
        }
        k = args.get("k", 5)
        mode = args.get("mode", "balanced")
        diversity = args.get("diversity", True)
        results = recommend_songs(user_prefs, self.songs, k=max(k, 10), mode=mode, diversity=diversity)
        results = self._filter(results, args)[:k]
        out = [
            {
                "title": s["title"],
                "artist": s["artist"],
                "genre": s["genre"],
                "mood": s["mood"],
                "energy": s.get("energy"),
                "acousticness": s.get("acousticness"),
                "score": score,
                "why": why,
            }
            for s, score, why in results
        ]
        self._last_results = out
        return out

    def _validate(self, user_message: str, results: list) -> Dict:
        issues = []
        msg = (user_message or "").lower()

        if not results:
            issues.append("no songs returned")

        if "no acoustic" in msg or "not acoustic" in msg:
            bad = [r["title"] for r in results if (r.get("acousticness") or 0) > 0.6]
            if bad:
                issues.append(f"user said no acoustic but these are highly acoustic: {bad}")

        if "low energy" in msg or "calm" in msg or "chill" in msg:
            high = [r["title"] for r in results if (r.get("energy") or 0) > 0.7]
            if high:
                issues.append(f"user wanted low energy but these are high energy: {high}")

        if "high energy" in msg or "workout" in msg or "intense" in msg or "gym" in msg:
            low = [r["title"] for r in results if (r.get("energy") or 0) < 0.6]
            if low:
                issues.append(f"user wanted high energy but these are low energy: {low}")

        if not issues:
            return {"ok": True, "issues": []}

        self._retry_count += 1
        if self._retry_count > self.MAX_RETRIES:
            return {
                "ok": True,
                "best_effort": True,
                "issues": issues,
                "note": "Retry budget used up. Catalog cannot fully satisfy. Write a final answer that acknowledges the limitation honestly.",
            }
        return {"ok": False, "issues": issues, "retries_left": self.MAX_RETRIES - self._retry_count + 1}

    def _call_tool(self, name: str, args: dict, user_message: str):
        if name == "recommend_songs":
            return self._recommend(args)
        if name == "validate_results":
            results = args.get("results") or self._last_results
            return self._validate(args.get("user_message", user_message), results)
        raise ValueError(f"Unknown tool: {name}")

    def run(self, user_message: str, max_steps: int = 8) -> Dict:
        if not user_message or not user_message.strip():
            logger.info("RUN | empty input, asking user to clarify")
            return {"answer": "Please tell me what kind of music you want.", "trace": []}

        self._retry_count = 0
        self._last_results = []
        logger.info(f"RUN | new request | user_message={user_message!r}")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        trace = []

        for step_num in range(1, max_steps + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=TOOLS,
                )
            except OpenAIError as e:
                logger.exception("OPENAI_ERROR | call failed")
                return {
                    "answer": f"The AI service hit an error: {e}. Try again in a moment.",
                    "trace": trace,
                    "error": str(e),
                }

            msg = response.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    logger.info(f"STEP {step_num} | PLAN -> TOOL | {tc.function.name} | args={args}")
                    trace.append({"type": "tool_call", "name": tc.function.name, "args": args})
                    try:
                        result = self._call_tool(tc.function.name, args, user_message)
                    except Exception as e:
                        logger.exception("TOOL_ERROR | tool execution failed")
                        result = {"error": str(e)}

                    if tc.function.name == "recommend_songs" and isinstance(result, list):
                        logger.info(f"STEP {step_num} | OBSERVATION | {len(result)} songs returned")
                    elif tc.function.name == "validate_results" and isinstance(result, dict):
                        if result.get("ok") and not result.get("best_effort"):
                            logger.info(f"STEP {step_num} | VALIDATION_PASS")
                        elif result.get("best_effort"):
                            logger.warning(f"STEP {step_num} | RETRY_BUDGET_EXHAUSTED | issues={result.get('issues')}")
                        else:
                            logger.warning(f"STEP {step_num} | VALIDATION_FAIL | issues={result.get('issues')} | retries_left={result.get('retries_left')}")
                    trace.append({"type": "tool_result", "name": tc.function.name, "result": result})
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(result),
                        }
                    )
                continue

            logger.info(f"STEP {step_num} | FINAL_ANSWER")
            trace.append({"type": "final", "content": msg.content})
            return {"answer": msg.content, "trace": trace}

        logger.warning("RUN | max_steps reached without final answer")
        return {"answer": "Agent stopped without a final answer.", "trace": trace}


def run_agent(user_message: str, songs_path: Optional[str] = None, max_steps: int = 8) -> Dict:
    if songs_path is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        songs_path = os.path.join(base, "data", "songs.csv")
    songs = load_songs(songs_path)
    return MusicAgent(songs).run(user_message, max_steps=max_steps)


if __name__ == "__main__":
    import sys

    msg = " ".join(sys.argv[1:]) or "I want chill study music, no acoustic"
    result = run_agent(msg)

    print("=" * 60)
    print(f"User: {msg}")
    print("=" * 60)
    for step in result["trace"]:
        if step["type"] == "tool_call":
            print(f"\n[PLAN -> {step['name']}] args={step['args']}")
        elif step["type"] == "tool_result":
            if step["name"] == "recommend_songs":
                print(f"[OBSERVATION] {len(step['result'])} songs returned")
                for s in step["result"][:3]:
                    print(f"  - {s['title']} by {s['artist']} ({s['genre']}) score={s['score']}")
            else:
                print(f"[OBSERVATION] validation: {step['result']}")
        elif step["type"] == "final":
            print(f"\n[AGENT FINAL ANSWER]\n{step['content']}")
    print()
