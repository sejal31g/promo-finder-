"""Step 7: rank, dedupe, diversify, and write the deliverables. Pure code, no model calls.
Usage: python -m promo_finder.rank data/candidates.json data/scores.json data/bible.json out_dir [--top 10]
"""
import json, csv, os, re, sys, argparse, statistics as st

W = {"hook": 0.30, "cliffhanger": 0.30, "standalone": 0.25, "stakes": 0.15}
TIGHTEN_PENALTY, NEIGHBOR_PENALTY = 0.5, 0.6
INTIMATE = re.compile(r"\b(in bed|the bed|bare[- ]chested|bare skin|chemise|shirtless|naked|undress\w*)\b", re.I)
LATE_EPISODE = 25   # judge overrates "standalone" for clips that rely on the story so far


def overall(s): return round(sum(W[k] * s[k] for k in W), 2)


def overlaps(a, b):
    """Same episode and the clips share more than 30% of the shorter one."""
    for sa in a["segments"]:
        for sb in b["segments"]:
            if sa["episode"] != sb["episode"]: continue
            inter = min(sa["end"], sb["end"]) - max(sa["start"], sb["start"])
            if inter > 0.3 * min(sa["end"] - sa["start"], sb["end"] - sb["start"]): return True
    return False


def seg_str(c): return " + ".join(f"Ep{s['episode']} {s['start_str']}-{s['end_str']}" for s in c["segments"])


def rank(cands, scores, bible, late=LATE_EPISODE):
    twist_eps = {t["episode"]: t["what"] for t in bible.get("twists", []) if t.get("magnitude") == "major"}
    rows = []
    for c in cands:
        s = scores.get(c["id"])
        if not s: continue
        adj = overall(s) - (TIGHTEN_PENALTY if c["flags"]["needs_tightening"] else 0)
        eps_in = {sg["episode"] for sg in c["segments"]}
        note = [f"spoiler check: within one episode of a major twist ({twist_eps[t]})" for t in sorted(twist_eps) if any(abs(e - t) <= 1 for e in eps_in)]
        if c["flags"]["needs_tightening"]: note.append("hook/cliffhanger position not verified or trim failed")
        if max(eps_in) > late: note.append(f"late-episode clip (ep>{late}): relies on the story so far; judge likely overrates 'standalone'")
        if INTIMATE.search(c["clip_text"]): note.append("intimate content: check platform policy")
        rows.append({**c, "score": s, "overall": overall(s), "adjusted": round(adj, 2), "review": note})
    rows.sort(key=lambda r: -r["adjusted"])
    kept, dropped = [], []
    for r in rows:
        (dropped if any(overlaps(r, k) for k in kept) else kept).append(r)       # overlapping: keep the better one
    order, pool = [], [r for r in kept if r["score"]["verdict"] != "skip"]
    skipped = [r for r in kept if r["score"]["verdict"] == "skip"]
    while pool:                                                                  # greedy: avoid 3 clips from one scene
        best = max(pool, key=lambda r: r["adjusted"] - NEIGHBOR_PENALTY * sum(abs(o["episode"] - r["episode"]) <= 1 for o in order))
        best["after_spread"] = round(best["adjusted"] - NEIGHBOR_PENALTY * sum(abs(o["episode"] - best["episode"]) <= 1 for o in order), 2)
        order.append(best); pool.remove(best)
    for i, r in enumerate(order, 1): r["rank"] = i
    return order, skipped, dropped


def write(order, skipped, dropped, out_dir, top):
    os.makedirs(out_dir, exist_ok=True)
    json.dump(order + skipped, open(f"{out_dir}/promos_ranked.json", "w"), indent=1)
    with open(f"{out_dir}/promos_ranked.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "id", "verdict", "segments", "duration_s", "overall", "hook", "cliffhanger", "standalone", "stakes",
                    "hook_text", "cliffhanger_text", "why_it_works", "review_flags", "after_spread"])
        for r in order:
            s = r["score"]
            w.writerow([r["rank"], r["id"], s["verdict"], seg_str(r), r["duration"], r["overall"], s["hook"], s["cliffhanger"],
                        s["standalone"], s["stakes"], r["hook"], r["cliffhanger"], s["one_line"], "; ".join(r["review"]), r["after_spread"]])
    L = [f"# Promo shortlist: {len(order)} clips ranked ({len(skipped)} judged 'skip', {len(dropped)} overlapping duplicates removed)\n",
         f"Score = 0.30 hook + 0.30 cliffhanger + 0.25 standalone + 0.15 stakes (blind judge, 0-10). "
         f"Clips whose hook/cliffhanger position could not be verified lose {TIGHTEN_PENALTY}. "
         f"Ranking spreads picks out: a clip loses {NEIGHBOR_PENALTY} for each higher-ranked clip within one episode, so the order can differ from the raw score. "
         f"Scores inside the top 10 are close together; treat the list as a shortlist, not a precise order.\n", f"## Post these first (top {top})\n"]
    for r in order[:top]:
        s = r["score"]
        L += [f"**{r['rank']}. {r['id']}: {seg_str(r)} ({r['duration']:.0f}s)**  score {r['adjusted']} (after spread {r['after_spread']}) "
              f"(hook {s['hook']}, cliffhanger {s['cliffhanger']}, standalone {s['standalone']}, stakes {s['stakes']})",
              f"- Hook: {r['hook']}", f"- Cliffhanger: {r['cliffhanger']}", f"- Why it works: {s['one_line']}"]
        if s["confusion"].lower() != "none": L.append(f"- Weak point: {s['confusion']}")
        if r["review"]: L.append(f"- REVIEW: {' | '.join(r['review'])}")
        L.append("")
    if len(order) > top:
        L.append(f"## Rest of the ranking\n\n| # | clip | segments | s | score | after spread | verdict | flags |\n|---|---|---|---|---|---|---|---|")
        for r in order[top:]:
            L.append(f"| {r['rank']} | {r['id']} | {seg_str(r)} | {r['duration']:.0f} | {r['adjusted']} | {r['after_spread']} | {r['score']['verdict']} | {len(r['review'])} |")
    open(f"{out_dir}/promos_report.md", "w", encoding="utf-8").write("\n".join(L))


def main(cand_path, score_path, bible_path, out_dir, top=10):
    cands, scores, bible = (json.load(open(p)) for p in (cand_path, score_path, bible_path))
    order, skipped, dropped = rank(cands, scores, bible)
    write(order, skipped, dropped, out_dir, top)
    print(f"{len(order)} ranked, {len(skipped)} skipped by judge, {len(dropped)} duplicates removed -> {out_dir}/")
    if order: print("top 5:", [(r["rank"], r["id"], r["adjusted"]) for r in order[:5]])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cands"); ap.add_argument("scores"); ap.add_argument("bible"); ap.add_argument("out_dir")
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args(); main(a.cands, a.scores, a.bible, a.out_dir, a.top)
