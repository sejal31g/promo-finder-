"""Step 6: blind judge. The model sees ONLY each clip's text (no bible, no generator claims),
so it cannot fill in context a real stranger would not have.
Usage: python -m promo_finder.score data/candidates.json data/bible.json data/scores.json [--batch 10] [--seed N]
"""
import json, os, sys, time, random, argparse
from .llm import ask_json, QuotaError

SYSTEM = "You are a tough, experienced short-form video editor. Reply with valid JSON only."
PROMPT = """You will judge {n} candidate promo clips cut from a vertical microdrama. Each clip is shown as text: a
transcript with times relative to the clip start, plus camera/action notes (SHOT/ACTION). A "--- cut ---" line means
the clip jumps to a later moment. You know NOTHING else about the show. Judge each clip INDEPENDENTLY, as a stranger
scrolling TikTok/Instagram would experience it.

Score each 0-10 (integers):
- hook: do the first 3 seconds stop a scrolling stranger? (10 = slap, shock or reveal right away; 3 = slow setup or generic talk)
- cliffhanger: does the clip END on an unresolved moment that makes the stranger NEED the next part? (10 = open
  question or threat at its peak; 3 = ends calm, resolved, or on a reaction shot)
- standalone: can a stranger follow who is who, what is happening and why it matters from this clip alone? (10 = fully
  clear; 3 = needs earlier episodes)
- stakes: emotional intensity and drama of the clip as a whole.
Be strict and use the full range: an average clip is 4-6 and 9-10 is rare. Speaker labels are for you; a viewer only
hears names that are spoken and sees what the SHOT notes describe.

For each clip also write: "first_3s" (what the viewer sees and hears in the first 3 seconds), "open_question" (the one
question the viewer is left asking at the very end, or "none"), "confusion" (the main thing a stranger would not
understand, or "none"), "verdict" ("post", "maybe" or "skip") and "one_line" (why a stranger would or would not keep watching).

Return {{"scores": [{{"id": "...", "hook": 0, "cliffhanger": 0, "standalone": 0, "stakes": 0, "first_3s": "...",
"open_question": "...", "confusion": "...", "verdict": "post", "one_line": "..."}}]}} with exactly one entry per clip, same ids.

{clips}"""
KEYS = ("hook", "cliffhanger", "standalone", "stakes")


def neutralize(text, bible):
    """'Alexander, Duke of Ashburn' -> 'Alexander': titles are story knowledge a stranger does not have."""
    for c in (bible or {}).get("cast", []):
        if "," in c["name"]:
            text = text.replace(c["name"], c["name"].split(",")[0])
    return text


def build_prompt(batch, bible):
    clips = "\n\n".join(f"### CLIP {c['id']} ({c['duration']:.0f}s)\n{neutralize(c['clip_text'], bible)}" for c in batch)
    return PROMPT.format(n=len(batch), clips=clips)


def clean(entry):
    """Coerce to ints 0-10; return None if unusable."""
    try:
        out = {k: max(0, min(10, int(round(float(entry[k]))))) for k in KEYS}
    except (KeyError, TypeError, ValueError):
        return None
    for k in ("first_3s", "open_question", "confusion", "verdict", "one_line"):
        out[k] = str(entry.get(k, "")).strip()
    if out["verdict"] not in ("post", "maybe", "skip"): out["verdict"] = "maybe"
    return out


def main(cand_path, bible_path, out, batch=10, sleep=6.0, seed=None):
    cands = json.load(open(cand_path)); bible = json.load(open(bible_path))
    scores, order = {}, list(cands)
    if seed is not None: random.Random(seed).shuffle(order)       # position-bias / reliability test
    todo, rnd = order, 0
    while todo and rnd < 3:
        for i in range(0, len(todo), batch):
            group = todo[i:i + batch]
            try:
                res = ask_json(build_prompt(group, bible), system=SYSTEM, temperature=0.0, tag=f"score_s{seed}_r{rnd}_{i//batch:02d}")
            except QuotaError as e:
                json.dump(scores, open(out, "w"), indent=1); print(f"\nSTOPPED: {e}\nSaved {len(scores)} scores. Re-run the same command later."); return {"scored": len(scores), "total": len(cands), "stopped": str(e)}
            if res is None: continue
            ids = {c["id"] for c in group}
            for e in res.get("scores", []):
                s = clean(e)
                if s and e.get("id") in ids: scores[e["id"]] = s
            json.dump(scores, open(out, "w"), indent=1)
            print(f"scored {len(scores)}/{len(cands)}"); time.sleep(sleep)
        todo = [c for c in order if c["id"] not in scores]; rnd += 1   # retry anything the model skipped
    json.dump(scores, open(out, "w"), indent=1)                       # always write, even if nothing was scored
    print(f"\n{len(scores)} of {len(cands)} clips scored -> {out}" + (f" | UNSCORED: {[c['id'] for c in todo]}" if todo else ""))
    return {"scored": len(scores), "total": len(cands), "stopped": None}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cands"); ap.add_argument("bible"); ap.add_argument("out")
    ap.add_argument("--batch", type=int, default=10); ap.add_argument("--sleep", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args(); main(a.cands, a.bible, a.out, a.batch, a.sleep, a.seed)
