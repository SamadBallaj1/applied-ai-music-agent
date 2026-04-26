"""
Agentic layer on top of the rule-based recommender.

The agent reads a natural-language request, picks taste preferences,
calls recommend_songs as a tool, and writes a friendly final answer.
"""

import json
import os
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI

try:
    from recommender import load_songs, recommend_songs
except ModuleNotFoundError:
    from src.recommender import load_songs, recommend_songs

load_dotenv()

MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are a music recommendation assistant.

You have one tool: recommend_songs. Always use it. Do not invent songs.

Steps for every request:
1. Read the user's words and pick taste preferences.
2. Call recommend_songs with those preferences.
3. Read the results and write a short, friendly answer that names the top picks
   and explains why they fit, in plain English.

Catalog facts:
- Genres: pop, lofi, rock, ambient, jazz, synthwave, indie pop, hip-hop, r&b,
  country, classical, edm, latin, metal, folk
- Moods: happy, chill, intense, relaxed, focused, moody
- Energy, danceability, valence are floats from 0.0 to 1.0.

If the user says "study" or "focus", treat that as mood=focused or chill, low-mid energy.
If the user says "workout" or "gym", treat that as high energy and intense or happy mood.
If the user mentions acoustic, set likes_acoustic=true.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "recommend_songs",
            "description": "Score the catalog against the user's preferences and return the top K songs with explanations.",
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
                },
                "required": ["genre", "mood", "energy"],
            },
        },
    }
]


class MusicAgent:
    def __init__(self, songs: List[Dict], model: str = MODEL):
        self.client = OpenAI()
        self.songs = songs
        self.model = model

    def _call_tool(self, name: str, args: dict) -> List[Dict]:
        if name != "recommend_songs":
            raise ValueError(f"Unknown tool: {name}")
        user_prefs = {
            k: v
            for k, v in args.items()
            if k in ("genre", "mood", "energy", "danceability", "valence", "likes_acoustic")
        }
        k = args.get("k", 5)
        mode = args.get("mode", "balanced")
        diversity = args.get("diversity", True)
        results = recommend_songs(user_prefs, self.songs, k=k, mode=mode, diversity=diversity)
        return [
            {
                "title": song["title"],
                "artist": song["artist"],
                "genre": song["genre"],
                "mood": song["mood"],
                "score": score,
                "why": explanation,
            }
            for song, score, explanation in results
        ]

    def run(self, user_message: str, max_steps: int = 5) -> Dict:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        trace = []

        for _ in range(max_steps):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS,
            )
            msg = response.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments)
                    trace.append({"type": "tool_call", "name": tc.function.name, "args": args})
                    result = self._call_tool(tc.function.name, args)
                    trace.append({"type": "tool_result", "result": result})
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(result),
                        }
                    )
                continue

            trace.append({"type": "final", "content": msg.content})
            return {"answer": msg.content, "trace": trace}

        return {"answer": "Agent stopped without a final answer.", "trace": trace}


def run_agent(user_message: str, songs_path: str = None, max_steps: int = 5) -> Dict:
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
            print(f"\n[PLAN -> TOOL] {step['name']}({step['args']})")
        elif step["type"] == "tool_result":
            print(f"[OBSERVATION] {len(step['result'])} songs returned")
            for s in step["result"][:3]:
                print(f"  - {s['title']} by {s['artist']} ({s['genre']}) score={s['score']}")
        elif step["type"] == "final":
            print(f"\n[AGENT FINAL ANSWER]\n{step['content']}")
    print()
