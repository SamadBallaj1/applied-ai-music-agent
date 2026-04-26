"""
Command line runner.

Two modes:
- Default: runs the three demo profiles through the rule-based recommender.
- --agent "your message": runs the natural-language agent.
"""

import argparse
import os

try:
    from recommender import load_songs, recommend_songs, RANKING_MODES
except ModuleNotFoundError:
    from src.recommender import load_songs, recommend_songs, RANKING_MODES

from tabulate import tabulate

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def print_recommendations(title, recs):
    print(f"\n{title}")
    print("=" * len(title))
    rows = []
    for song, score, explanation in recs:
        rows.append([
            song["title"],
            song["artist"],
            song["genre"],
            f"{score:.2f}",
            explanation
        ])
    print(tabulate(rows, headers=["Song", "Artist", "Genre", "Score", "Why"], tablefmt="grid"))


def run_profile(name, user_prefs, songs):
    print(f"\n{'*' * 60}")
    print(f"  Profile: {name}")
    print(f"  Prefs: {user_prefs}")
    print(f"{'*' * 60}")

    recs = recommend_songs(user_prefs, songs, k=5)
    print_recommendations(f"{name} - Balanced Mode", recs)

    recs_div = recommend_songs(user_prefs, songs, k=5, diversity=True)
    print_recommendations(f"{name} - Balanced + Diversity", recs_div)

    recs_genre = recommend_songs(user_prefs, songs, k=3, mode="genre-first")
    print_recommendations(f"{name} - Genre-First Mode (top 3)", recs_genre)

    recs_mood = recommend_songs(user_prefs, songs, k=3, mode="mood-first")
    print_recommendations(f"{name} - Mood-First Mode (top 3)", recs_mood)


def run_demo_profiles():
    songs = load_songs(os.path.join(BASE_DIR, "data", "songs.csv"))
    print(f"Loaded {len(songs)} songs")

    profiles = {
        "High-Energy Pop Fan": {
            "genre": "pop",
            "mood": "happy",
            "energy": 0.85,
            "danceability": 0.8,
            "valence": 0.8,
            "likes_acoustic": False,
        },
        "Chill Lofi Listener": {
            "genre": "lofi",
            "mood": "chill",
            "energy": 0.35,
            "danceability": 0.55,
            "valence": 0.6,
            "likes_acoustic": True,
            "mood_tag": "peaceful",
        },
        "Intense Rock Lover": {
            "genre": "rock",
            "mood": "intense",
            "energy": 0.92,
            "danceability": 0.65,
            "valence": 0.45,
            "likes_acoustic": False,
            "mood_tag": "aggressive",
        },
    }

    for name, prefs in profiles.items():
        run_profile(name, prefs, songs)

    print("\n" + "=" * 60)
    print("Done! Check the results above for each profile.")


def run_agent_mode(message):
    try:
        from agent import run_agent
    except ModuleNotFoundError:
        from src.agent import run_agent

    result = run_agent(message)

    print("=" * 60)
    print(f"User: {message}")
    print("=" * 60)
    for step in result["trace"]:
        if step["type"] == "tool_call":
            print(f"\n[PLAN -> {step['name']}] args={step['args']}")
        elif step["type"] == "tool_result":
            if step["name"] == "recommend_songs":
                print(f"[OBSERVATION] {len(step['result'])} songs returned")
                for s in step["result"][:5]:
                    print(f"  - {s['title']} by {s['artist']} ({s['genre']}) score={s['score']}")
            else:
                print(f"[OBSERVATION] validation: {step['result']}")
        elif step["type"] == "final":
            print(f"\n[AGENT FINAL ANSWER]\n{step['content']}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Music Recommender")
    parser.add_argument(
        "--agent",
        nargs="+",
        help='Run the natural-language agent. Example: --agent "I want chill study music"',
    )
    args = parser.parse_args()

    if args.agent:
        run_agent_mode(" ".join(args.agent))
    else:
        run_demo_profiles()


if __name__ == "__main__":
    main()
