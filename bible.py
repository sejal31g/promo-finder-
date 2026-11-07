"""Step 3: build a 'show bible' from the parsed episodes.
Per-episode summaries come free from the loglines (code). The model only has to
reconcile the cast, find the big twists and group episodes into arcs.
Usage: python -m promo_finder.bible data/parsed data/bible.json
"""
import json, glob, sys, collections
from .llm import ask_json


def load(parsed_dir):
    return [json.load(open(f)) for f in sorted(glob.glob(f"{parsed_dir}/episode_*.json"))]


def speaker_table(eps):
    seen = collections.defaultdict(lambda: {"lines": 0, "first": None, "last": None})
    for ep in eps:
        for e in ep["events"]:
            if e["kind"] == "dialogue":
                s = seen[e["speaker"]]
                s["lines"] += 1
                s["first"] = s["first"] or ep["episode"]
                s["last"] = ep["episode"]
    return dict(seen)


def build_prompt(eps, speakers):
    eplines = "\n".join(f"EP{e['episode']}: {e['logline']} || CONTINUITY: {e['continuity']}" for e in eps)
    spk = "\n".join(f"{n} (lines={v['lines']}, eps {v['first']}-{v['last']})"
                    for n, v in sorted(speakers.items(), key=lambda x: -x[1]["lines"]))
    return f"""Below are the episode loglines and continuity notes for the first {len(eps)} episodes of a vertical
microdrama (1-2 min episodes), followed by every speaker label used in the scripts.

Return JSON with exactly these keys:
- "premise": 2 sentences, written so a viewer who has never seen the show understands the setup.
- "cast": list of {{"name": canonical name, "aliases": [speaker labels from the list that are this same person],
   "role": short phrase, "first_episode": int}}. Cover EVERY speaker label in the list, including generic ones
   (e.g. MATRIARCH, THUG); if a generic label is clearly a named character, make it an alias, otherwise list it
   as its own minor character. Only merge labels when the episode notes support it. If unsure, keep them separate.
- "twists": list of {{"episode": int, "what": one sentence, "magnitude": "major" or "minor"}}. Major = reveals that
   define the show's central mystery or change a main relationship. These are what promos should NOT give away.
- "arcs": list of {{"name": str, "episodes": [first, last], "summary": one sentence}} covering all episodes.
- "central_hooks": 3 short phrases describing what makes this show addictive (the core fantasy/conflict).

Use only the information below. Do not invent events.

EPISODES:
{eplines}

SPEAKER LABELS:
{spk}
"""


def check(bible, speakers, n_eps):
    """Code-side verification of the model output."""
    problems = []
    covered = set()
    for c in bible.get("cast", []):
        covered.add(c.get("name")); covered.update(c.get("aliases", []))
    missing = [s for s in speakers if s not in covered]
    if missing: problems.append(f"speaker labels not in cast: {missing}")
    unknown = [a for c in bible.get("cast", []) for a in c.get("aliases", []) if a not in speakers]
    if unknown: problems.append(f"aliases that are not real labels (invented?): {unknown}")
    for t in bible.get("twists", []):
        if not isinstance(t.get("episode"), int) or not 1 <= t["episode"] <= n_eps:
            problems.append(f"twist with bad episode: {t}")
    cov = set()
    for a in bible.get("arcs", []):
        try: cov.update(range(a["episodes"][0], a["episodes"][1] + 1))
        except Exception: problems.append(f"bad arc: {a}")
    gaps = [i for i in range(1, n_eps + 1) if i not in cov]
    if gaps: problems.append(f"episodes not in any arc: {gaps}")
    return problems


def main(parsed_dir, out):
    eps = load(parsed_dir)
    speakers = speaker_table(eps)
    prompt = build_prompt(eps, speakers)
    print(f"{len(eps)} episodes, {len(speakers)} speaker labels, prompt ~{len(prompt)//4} tokens")
    bible = ask_json(prompt, tag="bible")
    if bible is None:
        return None
    first = {}
    for ep in eps:                                   # first_episode is computed from the data, not trusted from the model
        for e in ep["events"]:
            if e["kind"] == "dialogue": first.setdefault(e["speaker"].upper(), ep["episode"])
    for c in bible.get("cast", []):
        hits = [first[a.upper()] for a in c.get("aliases", []) + [c["name"]] if a.upper() in first]
        if hits: c["first_episode"] = min(hits)
    bible["episode_summaries"] = {str(e["episode"]): e["logline"] for e in eps}   # free, from code
    bible["_check"] = check(bible, speakers, len(eps))
    json.dump(bible, open(out, "w"), indent=1)
    print("saved", out, "| problems:", bible["_check"] or "none")
    return bible


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
