import os
import sys
import ssl
import urllib.parse
import urllib.request
import json

import certifi
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from recommender import load_songs, recommend_songs, RANKING_MODES
from agent import run_agent, stream_agent

YT_KEY = os.environ.get("YOUTUBE_API_KEY")
SSL_CTX = ssl.create_default_context(cafile=certifi.where())


@st.cache_data(show_spinner=False)
def youtube_video_id(query: str):
    if not YT_KEY:
        return None
    params = urllib.parse.urlencode({
        "part": "snippet",
        "q": query,
        "type": "video",
        "videoCategoryId": "10",
        "videoEmbeddable": "true",
        "topicId": "/m/04rlf",
        "maxResults": 1,
        "key": YT_KEY,
    })
    url = f"https://www.googleapis.com/youtube/v3/search?{params}"
    try:
        with urllib.request.urlopen(url, timeout=8, context=SSL_CTX) as r:
            data = json.loads(r.read())
        items = data.get("items", [])
        if items:
            return items[0]["id"]["videoId"]
    except Exception:
        return None
    return None

st.set_page_config(page_title="Samad's Music Agent", page_icon="🎵", layout="centered")
st.title("🎵 Samad's Music Agent")

songs = load_songs(os.path.join(os.path.dirname(__file__), "data", "songs.csv"))


def get_recs_from_trace(trace):
    last = []
    for step in trace:
        if step.get("type") == "tool_result" and step.get("name") == "recommend_songs":
            if isinstance(step.get("result"), list):
                last = step["result"]
    return last


GENRE_COLORS = {
    "pop": "#ec4899", "lofi": "#8b5cf6", "rock": "#ef4444", "ambient": "#06b6d4",
    "jazz": "#f59e0b", "synthwave": "#a855f7", "indie pop": "#f472b6",
    "hip-hop": "#22c55e", "r&b": "#fb923c", "country": "#84cc16",
    "classical": "#94a3b8", "edm": "#3b82f6", "latin": "#f43f5e",
    "metal": "#dc2626", "folk": "#65a30d",
}


def render_song_card(song, idx=0):
    search_query = f"{song['genre']} {song['mood']} music {song['title']}"
    full_query = f"{song['title']} {song['artist']} {song['genre']}"
    encoded = urllib.parse.quote(full_query)
    video_id = youtube_video_id(search_query)
    yt_embed = f"https://www.youtube.com/embed/{video_id}" if video_id else None
    yt_search = f"https://www.youtube.com/results?search_query={encoded}"
    sp_search = f"https://open.spotify.com/search/{encoded}"
    color = GENRE_COLORS.get(song.get("genre", ""), "#6366f1")

    initials = "".join(w[0] for w in song["title"].split()[:2]).upper()

    card_html = f"""
    <div style="
        background: linear-gradient(135deg, {color}22 0%, #1a1a2e 100%);
        border: 1px solid {color}44;
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 10px;
        display: flex;
        gap: 16px;
        align-items: center;
    ">
        <div style="
            min-width: 72px; height: 72px;
            background: linear-gradient(135deg, {color} 0%, {color}88 100%);
            border-radius: 12px;
            display: flex; align-items: center; justify-content: center;
            font-size: 24px; font-weight: 700; color: white;
            box-shadow: 0 4px 12px {color}55;
        ">{initials}</div>
        <div style="flex: 1; min-width: 0;">
            <div style="font-size: 17px; font-weight: 600; color: #f5f5f5; margin-bottom: 4px;">
                {song['title']}
            </div>
            <div style="font-size: 13px; color: #a0a0a0; margin-bottom: 6px;">
                {song['artist']}
            </div>
            <div style="display: flex; gap: 6px; flex-wrap: wrap;">
                <span style="font-size: 11px; padding: 2px 8px; background: {color}33; color: {color}; border-radius: 10px;">{song['genre']}</span>
                <span style="font-size: 11px; padding: 2px 8px; background: #ffffff14; color: #d0d0d0; border-radius: 10px;">{song['mood']}</span>
                <span style="font-size: 11px; padding: 2px 8px; background: #ffffff14; color: #d0d0d0; border-radius: 10px;">score {song['score']:.2f}</span>
            </div>
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)
    if yt_embed:
        components.iframe(yt_embed, height=200)
    b1, b2, _ = st.columns([1, 1, 3])
    with b1:
        st.link_button("▶ YouTube", yt_search, use_container_width=True)
    with b2:
        st.link_button("♫ Spotify", sp_search, use_container_width=True)
    st.write("")


manual_tab, agent_tab = st.tabs(["Manual mode", "Talk to the agent"])

with manual_tab:
    st.subheader("Your Taste Profile")

    col1, col2 = st.columns(2)
    with col1:
        genre = st.selectbox("Favorite genre", sorted(set(s["genre"] for s in songs)))
        mood = st.selectbox("Favorite mood", sorted(set(s["mood"] for s in songs)))
        mood_tag = st.selectbox("Mood tag (optional)", [""] + sorted(set(s["mood_tag"] for s in songs)))
    with col2:
        energy = st.slider("Target energy", 0.0, 1.0, 0.7)
        danceability = st.slider("Danceability", 0.0, 1.0, 0.7)
        valence = st.slider("Valence (happiness)", 0.0, 1.0, 0.7)

    likes_acoustic = st.checkbox("I like acoustic songs")

    st.divider()
    st.subheader("Recommendation Settings")

    col3, col4 = st.columns(2)
    with col3:
        mode = st.selectbox("Ranking mode", list(RANKING_MODES.keys()))
    with col4:
        k = st.slider("Number of results", 1, 10, 5)

    diversity = st.checkbox("Enable diversity (penalize repeat artists/genres)")

    if st.button("Get Recommendations"):
        user_prefs = {
            "genre": genre,
            "mood": mood,
            "energy": energy,
            "danceability": danceability,
            "valence": valence,
            "likes_acoustic": likes_acoustic,
        }
        if mood_tag:
            user_prefs["mood_tag"] = mood_tag

        recs = recommend_songs(user_prefs, songs, k=k, mode=mode, diversity=diversity)

        st.subheader("Your Recommendations")
        for song, score, explanation in recs:
            with st.container():
                st.markdown(f"**{song['title']}** by {song['artist']} ({song['genre']})")
                st.markdown(f"Score: **{score:.2f}**")
                st.caption(explanation)
                st.divider()

with agent_tab:
    st.subheader("Talk to the agent")
    st.caption("Ask in plain English. The agent plans, calls the recommender, checks its work, and replies. Each pick is playable below.")

    if "agent_history" not in st.session_state:
        st.session_state.agent_history = []

    for entry in st.session_state.agent_history:
        with st.chat_message("user"):
            st.markdown(entry["user"])
        with st.chat_message("assistant"):
            st.markdown(entry["answer"])
            if entry.get("songs"):
                for song in entry["songs"]:
                    render_song_card(song)
            if entry.get("trace"):
                with st.expander("Show the agent's reasoning steps"):
                    for step in entry["trace"]:
                        if step["type"] == "tool_call":
                            st.markdown(f"**plan -> tool call:** `{step['name']}`")
                            st.json(step["args"])
                        elif step["type"] == "tool_result":
                            st.markdown(f"**observation from `{step['name']}`:**")
                            st.json(step["result"])

    user_msg = st.chat_input("e.g. chill study music, no acoustic")
    if user_msg:
        import time
        with st.chat_message("user"):
            st.markdown(user_msg)
        with st.chat_message("assistant"):
            trace = []
            songs_returned = []
            answer = ""
            timeline = []

            def render_timeline(active_label=None):
                rows = []
                for label in timeline:
                    rows.append(f"""
                    <div class="tl-row" style="display:flex;gap:14px;align-items:flex-start;position:relative;padding:6px 0;animation:tlfade .35s ease-out;">
                      <div style="min-width:14px;height:14px;border-radius:50%;background:#22c55e;box-shadow:0 0 0 3px rgba(34,197,94,0.18);margin-top:5px;"></div>
                      <div style="color:#e5e5e5;font-size:14px;line-height:1.5;">{label}</div>
                    </div>
                    """)
                if active_label:
                    rows.append(f"""
                    <div class="tl-row" style="display:flex;gap:14px;align-items:flex-start;position:relative;padding:6px 0;">
                      <div style="min-width:14px;height:14px;border-radius:50%;background:#a0a0a0;margin-top:5px;animation:tlpulse 1.1s ease-in-out infinite;"></div>
                      <div style="color:#a0a0a0;font-size:14px;line-height:1.5;">{active_label}</div>
                    </div>
                    """)
                return f"""
                <style>
                @keyframes tlpulse {{0%{{opacity:.4;transform:scale(.85)}}50%{{opacity:1;transform:scale(1.15)}}100%{{opacity:.4;transform:scale(.85)}}}}
                @keyframes tlfade {{from{{opacity:0;transform:translateY(-4px)}}to{{opacity:1;transform:translateY(0)}}}}
                </style>
                <div style="border-left:2px solid #2d1b69;padding-left:18px;margin:4px 0 4px 6px;">
                  {''.join(rows)}
                </div>
                """

            with st.status("🤖 Samad's Agent thinking...", expanded=True) as status:
                placeholder = st.empty()
                placeholder.html(render_timeline("reading your request"))
                time.sleep(0.5)

                for evt in stream_agent(user_msg):
                    t = evt["type"]
                    if t == "tool_call":
                        trace.append({"type": "tool_call", "name": evt["name"], "args": evt["args"]})
                        if evt["name"] == "recommend_songs":
                            args = evt["args"]
                            tags = []
                            for k in ("genre", "mood"):
                                if args.get(k):
                                    tags.append(f"<code>{k}={args[k]}</code>")
                            if "energy" in args:
                                tags.append(f"<code>energy={args['energy']}</code>")
                            if args.get("exclude_acoustic"):
                                tags.append("<code>no acoustic</code>")
                            timeline.append(f"planning: pulled out {' '.join(tags)}")
                            placeholder.html(render_timeline("calling the recommender"))
                            status.update(label="🔧 calling the recommender...", expanded=True)
                            time.sleep(0.45)
                        elif evt["name"] == "validate_results":
                            placeholder.html(render_timeline("checking picks against your request"))
                            status.update(label="🔍 checking the picks...", expanded=True)
                            time.sleep(0.45)
                    elif t == "tool_result":
                        trace.append({"type": "tool_result", "name": evt["name"], "result": evt["result"]})
                        if evt["name"] == "recommend_songs" and isinstance(evt["result"], list):
                            songs_returned = evt["result"]
                            timeline.append(f"got {len(songs_returned)} songs back from the recommender")
                            placeholder.html(render_timeline("about to validate"))
                            time.sleep(0.45)
                        elif evt["name"] == "validate_results" and isinstance(evt["result"], dict):
                            r = evt["result"]
                            if r.get("ok") and not r.get("best_effort"):
                                timeline.append("validation passed, picks match request")
                            elif r.get("best_effort"):
                                timeline.append(f"retry budget hit, catalog limits: <code>{r.get('issues')}</code>. writing honest answer")
                            else:
                                timeline.append(f"issues found: <code>{r.get('issues')}</code>. retrying ({r.get('retries_left')} retries left)")
                            placeholder.html(render_timeline("planning next step"))
                            time.sleep(0.5)
                    elif t == "final":
                        answer = evt["content"]
                        timeline.append("wrote final answer")
                        placeholder.html(render_timeline(None))
                        status.update(label="✅ done", state="complete", expanded=False)
                    elif t == "error":
                        timeline.append(f"error: {evt['content']}")
                        placeholder.html(render_timeline(None))
                        status.update(label="❌ error", state="error")

            st.markdown(answer)
            for song in songs_returned:
                render_song_card(song)

        st.session_state.agent_history.append(
            {
                "user": user_msg,
                "answer": answer,
                "songs": songs_returned,
                "trace": trace,
            }
        )
