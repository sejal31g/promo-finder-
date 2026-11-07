"""Step 4: propose promo candidates per episode, then validate every one with code.
The model picks moments by event id (never by timestamp); code converts to times, snaps to
whole lines, and enforces the rules. Rejected candidates are logged, not hidden.

Usage: python -m promo_finder.candidates data/parsed data/bible.json data/candidates.json [--episodes 1-10] [--batch 5] [--sleep 6]
"""
import json, glob, sys, os, re, time, argparse
from .llm import ask_json, QuotaError
from .clip import alias_map, build_view, render_view, snap, window_text, clip_text
from .parse import fmt

MIN_TOTAL, MAX_TOTAL, MAX_SEGS = 20.0, 60.0, 3
HOOK_TYPES = "slap | humiliation | shocking_line | reveal | threat | betrayal | rescue | romantic_tension | other"

PROMPT = """You are finding PROMO clips for a vertical microdrama. A promo is a 20-60 second clip posted on
TikTok/Instagram/YouTube to make a stranger want to watch the show.

A good promo:
1. HOOK: the first 3 seconds contain something that stops a scrolling viewer (a slap, a public humiliation, a
   shocking line, a reveal, a threat, a betrayal). Start exactly where the hook starts, with no warm-up.
2. CLIFFHANGER: it ends at a peak, unresolved moment, BEFORE the answer or the reaction that settles it. Never
   end on a calm, resolved or explanatory line. The viewer must need to know what happens next.
3. STANDS ALONE: someone who has never seen the show understands who is doing what to whom and why it matters.
4. Does not give away the show's biggest twists (listed below). Tease the moment just before a twist instead.

SHOW: {premise}

CHARACTERS (labels in the scripts are already merged to these names):
{cast}

BIG TWISTS (do not build a promo that reveals these):
{twists}

EPISODE SUMMARIES (to judge what is still unresolved when a clip ends; never include material from the episode after a clip):
{summaries}

SCRIPTS. Each line is "#id start-end text". Ids look like E12-7 (episode 12, line 7). SHOT = camera note, ACTION = stage action.
{scripts}

Return JSON: {{"candidates": [ ... ]}} with UP TO {n} candidates PER EPISODE, best first. Fewer is fine; skip an
episode that has nothing strong. Each candidate stays inside ONE episode:
{{"segments": [{{"start_id": "E12-7", "end_id": "E12-31"}}],   // 1-3 ranges, in time order, same episode
 "hook_id": "E12-7",       // the ONE line/shot that is the hook. It must be the clip's first event or within 3 s of the start.
 "hook": "what happens in the first 3 seconds",
 "hook_type": "{hook_types}",
 "cliffhanger_id": "E12-31", // the ONE line/shot that is the cliffhanger. The clip must END on it (within 3 s).
 "cliffhanger": "the unresolved moment the clip stops on",
 "why": "one line: why a stranger keeps watching",
 "context_gap": "what a first-time viewer would not know, or 'none'"}}

Rules: total length of a candidate must be 25-55 seconds (use the timestamps; stay clear of the 20/60 limits).
Do not start with warm-up and do not keep going after the peak: a calm reaction shot after the
cliffhanger kills it. End on an open question, threat or shock, not on an answer. Candidates must not overlap each other. Use only ids that appear above. Do not invent events."""


def parse_range(s, n):
    if not s: return range(1, n + 1)
    a, _, b = s.partition("-"); return range(int(a), int(b or a) + 1)


def cast_for(eps, bible):
    present = {e["speaker"].upper() for ep in eps for e in ep["events"] if e["kind"] == "dialogue"}
    out = []
    for c in bible.get("cast", []):
        if present & {a.upper() for a in c.get("aliases", []) + [c["name"]]}:
            out.append(f"- {c['name']}: {c.get('role', '')}")
    return "\n".join(out) or "(none)"


def build_prompt(batch, eps_by_n, bible, views, n=4):
    tw = "\n".join(f"- Episode {t['episode']}: {t['what']}" for t in bible.get("twists", []) if t.get("magnitude") == "major")
    last = batch[-1]["episode"]
    shown = batch + ([eps_by_n[last + 1]] if last + 1 in eps_by_n else [])
    summ = "\n".join(f"Episode {e['episode']}: {e['logline']}" for e in shown)
    scripts = "\n\n".join(f"=== EPISODE {e['episode']} ({e['duration']:.0f}s) ===\n{render_view(views[e['episode']])}" for e in batch)
    return PROMPT.format(premise=bible.get("premise", ""), cast=cast_for(batch, bible), twists=tw or "(none listed)",
                         summaries=summ, scripts=scripts, n=n, hook_types=HOOK_TYPES)


def parse_ids(cand):
    """'E12-7' -> (12, 7). Returns (episode, [(start_n, end_n), ...]) or raises ValueError."""
    segs, eps_seen = [], set()
    for s in cand.get("segments") or []:
        out = []
        for key in ("start_id", "end_id"):
            m = re.fullmatch(r"\s*E(\d+)-(\d+)\s*", str(s.get(key)))
            if not m: raise ValueError(f"bad id {s.get(key)!r}")
            eps_seen.add(int(m.group(1))); out.append(int(m.group(2)))
        segs.append(tuple(out))
    if len(eps_seen) != 1: raise ValueError("candidate mixes episodes (cross-episode joins not supported yet)" if eps_seen else "no segments")
    return eps_seen.pop(), segs


def one_id(x, ep_num):
    m = re.fullmatch(r"\s*E(\d+)-(\d+)\s*", str(x))
    return int(m.group(2)) if m and int(m.group(1)) == ep_num else None


def validate(ep, view, cand, idx, pairs=None):
    """Return (record, None) if the candidate passes every rule, else (None, reason)."""
    by_id = {v["id"]: v for v in view}
    if pairs is None:   # plain ints (used by tests)
        pairs = [(s.get("start_id"), s.get("end_id")) for s in cand.get("segments") or []]
    if not 1 <= len(pairs) <= MAX_SEGS:
        return None, f"segments must be a list of 1-{MAX_SEGS}"
    segs = []
    for a, b in pairs:
        if a not in by_id or b not in by_id: return None, f"unknown event id {a}-{b}"
        if a > b: return None, f"start_id {a} after end_id {b}"
        t0, t1, log = snap(ep, by_id[a]["start"], by_id[b]["end"])
        if t1 - t0 < 2: return None, f"segment too short ({t1 - t0:.1f}s)"
        if segs and t0 < segs[-1]["end"]: return None, "segments overlap or out of order"
        segs.append({"episode": ep["episode"], "start": t0, "end": t1, "start_str": fmt(t0), "end_str": fmt(t1), "snap": log})
    # --- tighten: the clip must start at its hook and stop on its cliffhanger (checked by code, not trusted) ---
    notes, hook_ok, cliff_ok = [], False, False
    h, c = one_id(cand.get("hook_id"), ep["episode"]), one_id(cand.get("cliffhanger_id"), ep["episode"])
    h, c = by_id.get(h), by_id.get(c)
    first, last = segs[0], segs[-1]
    if h and first["start"] - 0.05 <= h["start"] <= first["end"]:
        if h["start"] - first["start"] > 3.0:
            t0n, _, lg = snap(ep, h["start"], first["end"])
            notes.append(f"start trimmed {first['start']}->{t0n} to reach the hook")
            first["start"] = t0n; first["start_str"] = fmt(t0n)
        hook_ok = True
    if c and last["start"] <= c["end"] <= last["end"] + 0.05:
        if last["end"] - c["end"] > 3.0:
            _, t1n, lg = snap(ep, last["start"], c["end"])
            notes.append(f"end trimmed {last['end']}->{t1n} to stop on the cliffhanger")
            last["end"] = t1n; last["end_str"] = fmt(t1n)
        cliff_ok = True
    total = round(sum(s["end"] - s["start"] for s in segs), 1)
    if notes and total < MIN_TOTAL:                    # trimming would break the length rule: keep original, flag it
        for sg, (a, b) in zip(segs, pairs):
            sg["start"], sg["end"], _ = snap(ep, by_id[a]["start"], by_id[b]["end"])
            sg["start_str"], sg["end_str"] = fmt(sg["start"]), fmt(sg["end"])
        notes, total = [f"NOT trimmed: would fall under {MIN_TOTAL:.0f}s"], round(sum(s["end"] - s["start"] for s in segs), 1)
    if not MIN_TOTAL <= total <= MAX_TOTAL: return None, f"duration {total}s outside {MIN_TOTAL:.0f}-{MAX_TOTAL:.0f}s"
    t0, t1 = segs[0]["start"], segs[-1]["end"]
    rec = {"id": f"E{ep['episode']:03d}-C{idx}", "episode": ep["episode"], "segments": segs, "duration": total,
           "hook": cand.get("hook", ""), "hook_type": cand.get("hook_type", ""), "cliffhanger": cand.get("cliffhanger", ""),
           "why": cand.get("why", ""), "context_gap": cand.get("context_gap", ""),
           "opening": window_text(view, t0, t0 + 3), "closing": window_text(view, t1 - 3, t1),
           "clip_text": "\n--- cut ---\n".join(clip_text(view, s["start"], s["end"]) for s in segs),
           "flags": {"starts_at_scene_start": any(abs(sc["start"] - t0) < 0.3 for sc in ep["scenes"] if sc["start"] is not None),
                     "ends_at_episode_end": t1 >= ep["duration"] - 0.5,
                     "opens_without_dialogue": not any(v["kind"] == "dialogue" and v["start"] < t0 + 3 and v["end"] > t0 for v in view),
                     "was_snapped": any(s["snap"] for s in segs),
                     "hook_id_verified": hook_ok, "cliffhanger_id_verified": cliff_ok,
                     "needs_tightening": any(n.startswith("NOT") for n in notes) or not (hook_ok and cliff_ok)},
           "trim_log": notes}
    return rec, None


def main(parsed_dir, bible_path, out, episodes=None, sleep=6.0, batch=5, overrides_path="cast_overrides.json"):
    eps = [json.load(open(f)) for f in sorted(glob.glob(f"{parsed_dir}/episode_*.json"))]
    by_n = {e["episode"]: e for e in eps}
    bible = json.load(open(bible_path))
    amap = alias_map(bible, json.load(open(overrides_path)) if os.path.exists(overrides_path) else None)
    todo = [e for e in eps if not episodes or e["episode"] in episodes]
    views = {e["episode"]: build_view(e, amap) for e in eps}
    keep, rejected, counter = [], [], {}

    def save():
        json.dump(keep, open(out, "w"), indent=1)
        json.dump(rejected, open(out.replace(".json", "_rejected.json"), "w"), indent=1)

    for i in range(0, len(todo), batch):
        group = todo[i:i + batch]
        tag = f"cand_ep{group[0]['episode']:03d}-{group[-1]['episode']:03d}"
        try:
            res = ask_json(build_prompt(group, by_n, bible, views), tag=tag)
        except QuotaError as e:
            save(); print(f"\nSTOPPED: {e}\nPartial results saved ({len(keep)} kept). Re-run the same command later: finished batches are cached.")
            return {"kept": len(keep), "rejected": len(rejected), "stopped": str(e)}
        if res is None: continue                                    # dryrun
        for c in res.get("candidates", []):
            try:
                n, pairs = parse_ids(c)
                if n not in by_n or n not in {g["episode"] for g in group}: raise ValueError(f"id from episode {n}, not in this batch")
            except ValueError as err:
                rejected.append({"episode": None, "reason": str(err), "candidate": c}); continue
            counter[n] = counter.get(n, 0) + 1
            rec, why = validate(by_n[n], views[n], c, counter[n], pairs)
            if rec: keep.append(rec)
            else: rejected.append({"episode": n, "reason": why, "candidate": c})
        print(f"episodes {group[0]['episode']}-{group[-1]['episode']}: total kept {len(keep)}, rejected {len(rejected)}")
        save(); time.sleep(sleep)
    if keep or rejected:
        print(f"\n{len(keep)} candidates kept, {len(rejected)} rejected -> {out}")
    save()
    return {"kept": len(keep), "rejected": len(rejected), "stopped": None}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("parsed"); ap.add_argument("bible"); ap.add_argument("out")
    ap.add_argument("--episodes", default=None, help="e.g. 1-5 or 12"); ap.add_argument("--sleep", type=float, default=6.0)
    ap.add_argument("--batch", type=int, default=5, help="episodes per model call (free tiers allow few calls per day)")
    a = ap.parse_args()
    eps_all = len(glob.glob(f"{a.parsed}/episode_*.json"))
    main(a.parsed, a.bible, a.out, set(parse_range(a.episodes, eps_all)) if a.episodes else None, a.sleep, a.batch)
