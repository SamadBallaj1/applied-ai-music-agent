# Model Card: Music Recommender with an Agent

## 1. Model Name

**VibeMatch 2.0** — an agent layer on top of the original VibeMatch 1.0 rule-based recommender.

## 2. Intended Use

Take a natural language music request like "chill study music, no acoustic" and pick songs from a small CSV catalog of 20 songs. Built for a Codepath assignment, not real users or production traffic.

## 3. How it works

There are two parts:

- **Rule-based scorer** (from modules 1-3). Scores each song against the user's preferences using a point system: genre match, mood match, energy closeness, plus smaller bonuses for danceability, valence, acousticness, and popularity. Returns top K with explanations.
- **Agent layer** (new). Uses OpenAI `gpt-4o-mini` with tool calling. The agent reads natural language, picks taste preferences, calls the scorer as a tool, then calls a `validate_results` tool that checks whether the picks match the original request. If validation flags issues, the agent adjusts and retries (up to 2 times). If the catalog cannot satisfy the request, the agent gives up gracefully and writes an honest answer.

Every step is logged to `logs/agent.log` with a timestamp.

## 4. Data

20 songs, 15 attributes each (genre, mood, energy, tempo, valence, danceability, acousticness, popularity, release decade, mood tag, instrumentalness, liveness, etc.). Genres include pop, lofi, rock, ambient, jazz, synthwave, indie pop, hip-hop, r&b, country, classical, edm, latin, metal, folk.

The dataset leans toward pop and lofi. Several genres (folk, country, classical, metal, latin, jazz, ambient, hip-hop) only have 1 song each. This is the single biggest source of bias in the system.

## 5. Strengths

- Natural language input. The agent figures out preferences without making the user touch sliders.
- Self-checking. The validation tool catches obvious mismatches and forces a retry.
- Transparent. Every plan step, tool call, observation, and retry is logged. Easy to explain.
- Honest failure. When the catalog cannot satisfy a request, the agent says so instead of pretending.
- Multiple ranking modes (balanced, genre-first, mood-first, energy-focused) and a diversity penalty carried over from the original.

## 6. Limitations and biases

- **Tiny, unbalanced catalog.** With only 20 songs and many genres at 1 song each, the agent has little to work with. A request for "intense rock" or "acoustic folk" pulls in off-genre picks because there are not enough on-genre options.
- **Genre matching is exact.** "Indie pop" and "pop" are treated as totally different. Same with "r&b" and "soul" if it existed.
- **Popularity bonus skews picks.** Mainstream songs get a small advantage even when niche songs would fit better.
- **English only.** No language detection, no translation. The agent might recommend a song in a language the user does not speak.
- **No user history.** The agent does not learn from skips or replays.
- **Prompt injection not handled.** A user could try to override the agent's instructions in their message. I did not add a defense for this.

## 7. Could it be misused?

It is hard to weaponize a 20-song recommender, but a few honest concerns:

- The agent could be tricked into ignoring the tool and free-styling song recommendations that do not exist in the catalog. I tried to prevent this in the system prompt, but the prompt is not airtight.
- If someone copied this design and pointed it at a much bigger catalog, the agent could amplify whatever bias is in that data. A platform that mostly stocks one genre would push that genre at every user, regardless of taste.
- API cost abuse. A bad actor could spam the Streamlit app with long requests to burn through OpenAI credits.

How I would prevent these: input length limits, a simple allow-list of genres the model is allowed to claim, rate limiting on the Streamlit endpoint, and a stricter system prompt that refuses anything off topic.

## 8. Evaluation

I built a small eval harness in `scripts/eval.py` with 8 fixed natural language test cases. Each case has a list of checks (minimum results, allowed genres, energy range, no-acoustic constraint, etc.). Confidence is derived from how many retries the agent needed (0 retries = 1.0 confidence, 1 retry = 0.7, 2 retries = 0.4).

On my last full run: **5 out of 8 cases passed, average confidence 0.89.** The 3 fails were:
- "Acoustic folk or country" — only 1 folk and 1 country song in the catalog, so other genres leaked in.
- "Chill, no acoustic" — most chill songs in the catalog are acoustic, so the agent could not find enough matches.
- "Intense rock" — agent surfaced too many off-genre picks because of the small catalog.

All three fails are catalog problems, not agent bugs. That was a useful finding.

## 9. AI collaboration during this project

I used a few AI tools while building this: Claude inside VS Code, GitHub Copilot in the editor, and OpenAI Codex for some quick refactors. I wrote the architecture and made the design calls myself. The AI helped with boilerplate, syntax I did not remember, and second opinions when I was stuck.

**One time the AI suggestion was helpful:** I was trying to figure out how to make the agent check its own work without doing a hard filter inside the recommender. The AI suggested making validation a separate tool that the agent calls explicitly, plus a retry budget so the agent gives up if the catalog cannot satisfy. That two-piece design ended up being the cleanest part of the project. Without the budget my first version looped forever.

**One time the AI suggestion was flawed:** When I first built the validation flow, the AI wrote it so the model was supposed to pass the song results back into `validate_results` as an argument. In practice the model did not pass them, so the validator kept saying "no songs returned" and the agent retried in circles. I had to track the last results on the agent class itself and fall back to that when the model forgot to pass them. Catching this took a real test run, not just reading the code, which was a good reminder that AI suggestions look correct but can break in real loops.

## 10. What surprised me

How much the validation step changed the agent's behavior. Same model, same system prompt, but with validation in the loop, the agent stopped confidently lying about acoustic songs. It actually flagged the mismatch and tried something different. That felt closer to "intelligence" than I expected from a small model.

The other surprise was how fast the catalog became the bottleneck. Once the agent started working well, every interesting failure was traced back to "we just do not have enough songs in genre X." It made me think the AI part is actually the easier half of these systems.

## 11. Future work

- Bigger, more balanced catalog. At least 5 songs per genre.
- Fuzzy genre matching so "indie pop" partly counts as "pop".
- Light memory of past requests within a session.
- Add a proper guardrail against prompt injection.
- Try a smaller open model (like Llama 3 via Groq) and compare costs and quality.

## 12. Personal reflection

The biggest takeaway is that the algorithm is the simple part. The hard parts were the data, the validation, and deciding when to give up. Real recommender teams must spend most of their time on those, not on the math.

Working with AI tools while building this felt like pair programming with a fast junior. They saved me time on boilerplate and offered ideas I would not have thought of, but I had to verify everything because some of their suggestions broke quietly in production-like conditions.
