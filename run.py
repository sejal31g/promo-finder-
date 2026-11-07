"""Run the whole pipeline on a folder of screenplay .txt files (one show).
Usage: python -m promo_finder.run <screenplay_dir> [work_dir=data] [--from-step N]
Steps: 1 parse, 2 bible, 3 candidates, 4 score, 5 rank. Every model call is cached, so a re-run resumes where it stopped.
"""
import sys, os, argparse
from . import parse, bible, candidates, score, rank

STEPS = ["parse", "bible", "candidates", "score", "rank"]


def main(src, work="data", start=1, out_dir="output", batch=5):
    p = lambda f: os.path.join(work, f)
    os.makedirs(work, exist_ok=True)
    if start <= 1: print("\n[1/5] parse"); parse.main(src, p("parsed"))
    if start <= 2: print("\n[2/5] show bible"); bible.main(p("parsed"), p("bible.json"))
    if start <= 3: print("\n[3/5] candidates"); candidates.main(p("parsed"), p("bible.json"), p("candidates.json"), batch=batch)
    if start <= 4: print("\n[4/5] blind scoring"); score.main(p("candidates.json"), p("bible.json"), p("scores.json"))
    if start <= 5: print("\n[5/5] rank + report"); rank.main(p("candidates.json"), p("scores.json"), p("bible.json"), out_dir)
    print(f"\nDone. Open {out_dir}/promos_report.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("work", nargs="?", default="data")
    ap.add_argument("--from-step", type=int, default=1, help="1-5, see STEPS")
    ap.add_argument("--out", default="output"); ap.add_argument("--batch", type=int, default=5)
    a = ap.parse_args(); main(a.src, a.work, a.from_step, a.out, a.batch)
