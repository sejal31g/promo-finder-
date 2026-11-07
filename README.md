# Promo Finder

Reads a microdrama screenplay (one `.txt` per episode, with timestamps) and proposes 20-60 s promo clips:
exact start/end per segment, the hook, the cliffhanger, why it works, and a ranked shortlist.

Built for the Kuku AI Generalist assignment (Problem 1). The screenplays are confidential and are **not** in this repo.

## Run

```
pip install -r requirements.txt
export LLM_PROVIDER=gemini          # gemini | anthropic | openai (any OpenAI-compatible API)
export LLM_API_KEY=...              # a free key is enough
export LLM_MODEL=...                # optional; each provider has a default
python -m promo_finder.run path/to/screenplays        # whole pipeline, ~15 model calls for 50 episodes
```
Windows PowerShell: `$env:LLM_PROVIDER="gemini"` etc. Results land in `output/` (`promos_report.md`, `promos_ranked.csv`, `promos_ranked.json`).
Every model call is cached in `data/llm_cache`, so re-running resumes. To use another model, change only the env vars.
Only the Gemini path has been run live; the Anthropic and OpenAI-compatible paths are written but untested.

## Web app

```
pip install -r requirements.txt
streamlit run app.py
```
Upload a .zip (or several .txt files) of one show's episode screenplays, enter a model key in the sidebar, and press Find promos.
Results show ranked clips with exact timestamps and download as CSV, Markdown or JSON. Keys are used only for the session.
Hosted on Streamlit Community Cloud (main file `app.py`); the app contains no show content.
For the command line, copy `cast_overrides.example.json` to `cast_overrides.json` and edit it.

## How it works

| Step | Module | Who does it |
|---|---|---|
| 1. Parse screenplays into timestamped events | `parse.py` | code |
| 2. Show bible: cast with name variants, big twists, arcs | `bible.py` | model, then code checks it against the real speaker labels |
| 3. Propose candidates per episode (5 episodes per call) | `candidates.py` | model picks by event id; code converts to times, snaps to whole lines, enforces 20-60 s, verifies the hook is at the start and the cliffhanger at the end, and trims if not |
| 4. Blind judge: hook, cliffhanger, standalone, stakes | `score.py` | model sees only the clip text, nothing from steps 2-3 |
| 5. Rank, dedupe, spread picks, flag risks, write report | `rank.py` | code |

Human in the loop: `cast_overrides.json` (fix wrong character merges, per episode range) and the final editor review of the shortlist.

## Known limits
- Script only, no video: hooks are judged from text and camera notes.
- Candidates stay inside one episode (multi-segment within an episode works; cross-episode joins are not built).
- The judge is lenient and its scores cluster; treat the top 10 as a shortlist, not a precise order.
- Spoiler, late-episode and intimate-content flags are for the editor; they do not change scores.
