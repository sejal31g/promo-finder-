"""Shared helpers: the numbered script view the model sees, boundary snapping, clip text."""
import json, os
from .parse import fmt


def alias_map(bible, overrides=None):
    """label -> canonical display name (from the bible cast), plus optional human overrides.
    overrides: {"BLOND NOBLEMAN": {"to": "Rupert", "episodes": [27, 50]}}"""
    m = {}
    for c in (bible or {}).get("cast", []):
        for a in c.get("aliases", []) + [c["name"]]:
            m[a.upper()] = c["name"]
    return {"base": m, "over": overrides or {}}


def canon(label, ep_num, amap):
    o = amap["over"].get(label.upper())
    for entry in ([o] if isinstance(o, dict) else (o or [])):
        if entry["episodes"][0] <= ep_num <= entry["episodes"][1]:
            return entry["to"]
    return amap["base"].get(label.upper(), label)


def build_view(ep, amap):
    """All events sorted by time, numbered 1..n. Cuts are expressed as these ids."""
    evs = sorted(ep["events"], key=lambda e: (e["start"], e["end"]))
    view = []
    for i, e in enumerate(evs, 1):
        v = dict(e, id=i, label=f"E{ep['episode']}-{i}")
        if e["kind"] == "dialogue":
            who = canon(e["speaker"], ep["episode"], amap)
            tag = f" ({e['emotion']})" if e.get("emotion") else ""
            vo = " (V.O.)" if any("V" in m.upper() for m in e.get("mods", [])) else ""
            v["line"] = f'{who}{vo}{tag}: "{e["text"]}"'
        else:
            v["line"] = f"{ {'camera': 'SHOT', 'action': 'ACTION', 'sound': 'SOUND'}[e['kind']] }: {e['text']}"
        view.append(v)
    return view


def render_view(view):
    return "\n".join(f"#{v['label']} {fmt(v['start'])}-{fmt(v['end'])} {v['line']}" for v in view)


def snap(ep, t0, t1, tol=0.05):
    """Never cut mid dialogue line: pull start back / push end forward to whole lines."""
    log, dlg = [], [e for e in ep["events"] if e["kind"] == "dialogue"]
    moved = True
    while moved:
        moved = False
        for e in dlg:
            if e["start"] + tol < t0 < e["end"] - tol:
                log.append(f"start {t0}->{e['start']} (mid-line)"); t0 = e["start"]; moved = True
            if e["start"] + tol < t1 < e["end"] - tol:
                log.append(f"end {t1}->{e['end']} (mid-line)"); t1 = e["end"]; moved = True
    return round(t0, 2), round(t1, 2), log


def window_text(view, t0, t1, only=("dialogue", "action", "camera")):
    return " | ".join(v["line"] for v in view if v["start"] < t1 and v["end"] > t0 and v["kind"] in only)


def clip_text(view, t0, t1):
    """Everything inside the clip, with times relative to the clip start (for blind judging)."""
    rows = [f"[{max(v['start'] - t0, 0):04.1f}s] {v['line']}" for v in view
            if v["start"] >= t0 - 0.05 and v["end"] <= t1 + 0.05 and v["kind"] != "sound"]
    return "\n".join(rows)
