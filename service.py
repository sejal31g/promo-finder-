"""Logic behind the web app (no Streamlit imports, so it can be tested). Upload -> parse -> pipeline -> results."""
import io, os, json, math, glob, shutil, zipfile, csv
from . import parse, bible as bible_mod, candidates, score, rank, llm

MAX_BYTES, MAX_FILES, MAX_ONE = 25_000_000, 300, 5_000_000


def save_uploads(files, dest):
    """files: list of (filename, bytes). Accepts .txt files and .zip archives of .txt files. Returns number of .txt saved."""
    shutil.rmtree(dest, ignore_errors=True); os.makedirs(dest)
    n = total = 0
    def put(base, data):
        nonlocal n
        d = os.path.join(dest, f"{n:04d}"); os.makedirs(d)
        open(os.path.join(d, base), "wb").write(data); n += 1
    for name, data in files:
        total += len(data)
        if total > MAX_BYTES: raise ValueError("Upload is larger than 25 MB.")
        if name.lower().endswith(".zip"):
            try: z = zipfile.ZipFile(io.BytesIO(data))
            except zipfile.BadZipFile: raise ValueError(f"{name} is not a valid zip file.")
            for m in z.infolist():
                base = os.path.basename(m.filename)
                if m.is_dir() or not base.lower().endswith(".txt") or base.startswith(".") or "__MACOSX" in m.filename: continue
                if m.file_size == 0 or m.file_size > MAX_ONE: continue
                put(base, z.read(m))
        elif name.lower().endswith(".txt"):
            if 0 < len(data) <= MAX_ONE: put(os.path.basename(name), data)
        if n > MAX_FILES: raise ValueError(f"More than {MAX_FILES} files.")
    if n == 0: raise ValueError("No .txt screenplay files found in the upload.")
    return n


def prepare(files, work):
    """Save + parse (code only, no model calls). Returns a summary dict."""
    n = save_uploads(files, os.path.join(work, "src"))
    parsed = os.path.join(work, "parsed")
    shutil.rmtree(parsed, ignore_errors=True)
    parse.main(os.path.join(work, "src"), parsed)
    eps = [json.load(open(f)) for f in sorted(glob.glob(f"{parsed}/episode_*.json"))]
    return {"files": n, "episodes": len(eps), "events": sum(len(e["events"]) for e in eps),
            "dialogue": sum(1 for e in eps for v in e["events"] if v["kind"] == "dialogue"),
            "warnings": [f"Episode {e['episode']}: {w}" for e in eps for w in e["warnings"]],
            "numbers": [e["episode"] for e in eps]}


def limit_episodes(work, n):
    """Keep only the first n episodes (by number) so a quick test costs few model calls."""
    files = sorted(glob.glob(f"{work}/parsed/episode_*.json"))
    for f in files[n:]: os.remove(f)
    return min(n, len(files))


def estimate_calls(n_eps):
    return 1 + math.ceil(n_eps / 5) + math.ceil(0.8 * n_eps / 10)    # bible + candidate batches + scoring batches


def run_pipeline(work, cfg, overrides=None, on_step=lambda s: None, sleep=3.0):
    """Runs bible -> candidates -> blind scoring -> ranking. Returns a results dict; raises on hard errors."""
    cfg = dict(cfg, cache_dir=os.path.join(work, "cache"))
    llm.CONFIG.set(cfg)                                      # per-session settings, never shared between users
    p = lambda f: os.path.join(work, f)
    ov = None
    if overrides:
        ov = p("cast_overrides.json"); open(ov, "w").write(json.dumps(overrides))
    on_step("Building the show bible (cast, twists, arcs)")
    bible = bible_mod.main(p("parsed"), p("bible.json"))
    if bible is None: raise RuntimeError("The model returned nothing for the show bible.")
    on_step("Finding candidate clips, 5 episodes per call")
    st1 = candidates.main(p("parsed"), p("bible.json"), p("candidates.json"), sleep=sleep, batch=5, overrides_path=ov or "no_overrides.json")
    stopped = st1.get("stopped")
    st2 = {"scored": 0, "total": 0, "stopped": None}
    if not stopped and st1["kept"]:
        on_step("Judging every clip blind (hook, cliffhanger, standalone, stakes)")
        st2 = score.main(p("candidates.json"), p("bible.json"), p("scores.json"), sleep=sleep)
        stopped = st2.get("stopped")
    elif not os.path.exists(p("scores.json")):
        json.dump({}, open(p("scores.json"), "w"))
    on_step("Ranking and writing the report")
    rank.main(p("candidates.json"), p("scores.json"), p("bible.json"), p("output"))
    out = p("output")
    ranked = [r for r in json.load(open(f"{out}/promos_ranked.json")) if "rank" in r]
    return {"ranked": ranked, "candidates": st1["kept"], "scored": st2["scored"], "stopped": stopped,
            "bible_problems": bible.get("_check", []),
            "csv": open(f"{out}/promos_ranked.csv", encoding="utf-8").read(),
            "md": open(f"{out}/promos_report.md", encoding="utf-8").read(),
            "json": open(f"{out}/promos_ranked.json", encoding="utf-8").read()}
