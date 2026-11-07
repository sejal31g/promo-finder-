import os, json, shutil, tempfile
import requests
import streamlit as st
from promo_finder import service

st.set_page_config(page_title="Promo Finder", layout="wide")

DEFAULT_MODEL = {"gemini": "gemini-3.7-flash", "anthropic": "claude-sonnet-4-6", "openai": "gpt-4o-mini"}


def secret(name):
    try: return st.secrets.get(name)
    except Exception: return None


# ---------------------------------------------------------------- sidebar: model settings
with st.sidebar:
    st.header("Model")
    provider = st.selectbox("Provider", ["gemini", "anthropic", "openai"],
                            format_func={"gemini": "Google Gemini", "anthropic": "Anthropic", "openai": "OpenAI-compatible"}.get)
    model = st.text_input("Model name", value=DEFAULT_MODEL[provider], help="Any model the key can use. Swapping it is the only change needed.")
    fallback = st.text_input("Fallback model (optional)", value="", help="Used automatically if the main model is overloaded or out of daily quota. Each model has its own quota.")
    base_url = st.text_input("Base URL", value="https://api.openai.com/v1") if provider == "openai" else ""
    demo_key = secret("LLM_API_KEY") if provider == "gemini" else None
    use_demo = st.checkbox("Use the demo key (first 10 episodes only)", value=False) if demo_key else False
    api_key = demo_key if use_demo else st.text_input("API key", type="password", help="Used only for this session. Never stored or logged.")
    if provider == "gemini" and api_key and st.button("List models for this key"):
        try:
            r = requests.get("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200", headers={"x-goog-api-key": api_key}, timeout=30)
            r.raise_for_status()
            names = [m["name"].replace("models/", "") for m in r.json()["models"] if "generateContent" in m.get("supportedGenerationMethods", [])]
            st.code("\n".join(names))
        except Exception as e:
            st.error(f"Could not list models: {e}")
    st.caption("Free keys allow only a few dozen calls a day. One 50-episode show needs about 15.")
    with st.expander("Advanced: cast overrides"):
        ov_text = st.text_area("JSON", value="", height=110, placeholder='{"BLOND NOBLEMAN": {"to": "Rupert", "episodes": [27, 44]}}',
                               help="Fix wrong character merges: label, canonical name, episode range.")
    if st.button("Clear my data"):
        shutil.rmtree(st.session_state.get("work", ""), ignore_errors=True)
        st.session_state.clear(); st.rerun()

# ---------------------------------------------------------------- main
st.title("Promo Finder")
st.write("Upload a microdrama's episode screenplays. Get ranked 20 to 60 second promo clips, each with exact start and end timestamps, the hook, the cliffhanger and why it works.")
st.caption("Uploaded text is sent to the model provider you select. It is kept only in temporary storage for this session; use Clear my data to remove it.")

if "work" not in st.session_state:
    st.session_state.work = tempfile.mkdtemp(prefix="promo_")
work = st.session_state.work

uploads = st.file_uploader("Screenplays: a .zip, or several .txt files (one per episode, with timestamps)", type=["zip", "txt"], accept_multiple_files=True)

if uploads:
    sig = [(u.name, u.size) for u in uploads]
    if st.session_state.get("sig") != sig:
        try:
            st.session_state.info = service.prepare([(u.name, u.getvalue()) for u in uploads], work)
            st.session_state.sig = sig; st.session_state.pop("result", None)
        except ValueError as e:
            st.session_state.pop("info", None); st.error(str(e))
    info = st.session_state.get("info")
    if info:
        st.success(f"Read {info['episodes']} episodes: {info['events']:,} timestamped events, {info['dialogue']:,} dialogue lines.")
        if info["warnings"]:
            with st.expander(f"{len(info['warnings'])} parsing warnings"): st.write("\n".join(info["warnings"][:50]))
        limit_cap = 10 if use_demo else info["episodes"]
        n_use = st.number_input("Episodes to analyse (use a small number for a quick test)", min_value=1, max_value=info["episodes"],
                                value=min(info["episodes"], limit_cap), step=1)
        n_use = min(int(n_use), limit_cap)
        st.caption(f"This will use about {service.estimate_calls(n_use)} model calls and take a few minutes.")
        if st.button("Find promos", type="primary", disabled=not api_key):
            overrides = None
            if ov_text.strip():
                try: overrides = json.loads(ov_text)
                except json.JSONDecodeError: st.error("Cast overrides are not valid JSON."); st.stop()
            service.limit_episodes(work, n_use)
            cfg = {"provider": provider, "model": model.strip(), "fallback_model": fallback.strip(), "key": api_key, "base_url": base_url}
            with st.status("Working...", expanded=True) as status:
                try:
                    st.session_state.result = service.run_pipeline(work, cfg, overrides, on_step=lambda s: st.write(s))
                    status.update(label="Done", state="complete")
                except Exception as e:
                    status.update(label="Stopped", state="error")
                    st.error(str(e))
                    st.caption("Finished steps are cached for this session: click Find promos again to resume, or set a fallback model in the sidebar.")
        if not api_key:
            st.info("Enter an API key in the sidebar to run.")

res = st.session_state.get("result")
if res:
    if res["stopped"]:
        st.warning(f"The run stopped early: {res['stopped']} Showing what finished.")
    if res["bible_problems"]:
        with st.expander("Show-bible check found issues (review the cast)"): st.write(res["bible_problems"])
    ranked = res["ranked"]
    st.subheader(f"{len(ranked)} promos ranked")
    if not ranked:
        st.info("No clips passed. Try more episodes or a different model.")
    top = st.slider("Show top", 1, max(1, len(ranked)), min(10, max(1, len(ranked)))) if len(ranked) > 1 else 1
    d1, d2, d3 = st.columns(3)
    d1.download_button("Download CSV", res["csv"], "promos_ranked.csv", "text/csv")
    d2.download_button("Download report (Markdown)", res["md"], "promos_report.md", "text/markdown")
    d3.download_button("Download JSON", res["json"], "promos_ranked.json", "application/json")
    for r in ranked[:top]:
        s = r["score"]
        with st.container(border=True):
            st.markdown(f"**#{r['rank']}  {r['id']}**  ·  {r['duration']:.0f} s  ·  score {r['adjusted']}  ({s['verdict']})")
            for sg in r["segments"]:
                st.markdown(f"Episode {sg['episode']}: `{sg['start_str']}` to `{sg['end_str']}`")
            st.caption(f"Hook {s['hook']}  ·  Cliffhanger {s['cliffhanger']}  ·  Standalone {s['standalone']}  ·  Stakes {s['stakes']}")
            st.markdown(f"**Hook:** {r['hook']}")
            st.markdown(f"**Cliffhanger:** {r['cliffhanger']}")
            st.markdown(f"**Why it works:** {s['one_line']}")
            if s["confusion"].lower() != "none": st.markdown(f"**Weak point:** {s['confusion']}")
            for note in r["review"]: st.warning(note)
            with st.expander("Clip transcript"): st.text(r["clip_text"])
    if len(ranked) > top:
        st.subheader("Full ranking")
        st.dataframe([{"#": r["rank"], "clip": r["id"], "segments": " + ".join(f"Ep{x['episode']} {x['start_str']}-{x['end_str']}" for x in r["segments"]),
                       "sec": round(r["duration"]), "score": r["adjusted"], "verdict": r["score"]["verdict"]} for r in ranked], hide_index=True, width="stretch")

with st.expander("How it works, and its limits"):
    st.markdown("""
**Pipeline.** Code parses the screenplays. A model builds a short show bible (cast, twists). A model proposes clips by line id; code turns ids into times,
cuts only at whole lines, enforces 20 to 60 s, checks the hook is at the start and the cliffhanger at the end, and trims if not.
A second, blind model call judges each clip from its text alone. Code ranks the clips and flags risks.

**Limits.** Script only, no video. Clips stay inside one episode. The judge is lenient and scores cluster, so treat the top 10 as a shortlist, not a precise order.
Spoiler, late-episode and intimate-content flags are for an editor to review; they do not change scores.
""")
