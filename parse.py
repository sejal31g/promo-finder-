"""Parse Kuku-style screenplay .txt files into structured JSON.

Pure code, no AI. Tolerant of header case differences and speaker-label variants.
Usage: python -m promo_finder.parse <input_dir> <output_dir>
"""
import re, json, sys, glob, os

TIME = r"(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)"          # mm:ss.s  or  hh:mm:ss.s
NON_DIALOGUE = {"CAMERA": "camera", "ACTION": "action", "SOUND FX": "sound"}
MODIFIERS = re.compile(r"^(V\.?O\.?|O\.?S\.?|O\.?C\.?|CONT'?D\.?)$", re.I)


def to_sec(h, m, s):
    return round(int(h or 0) * 3600 + int(m) * 60 + float(s), 2)


def parse_time(txt):
    m = re.search(TIME, txt)
    return to_sec(*m.groups()) if m else None


def parse_label(label):
    """'JULIANA (V.O.) (Tender, Safe)' -> speaker, modifiers, emotion."""
    groups = re.findall(r"\(([^)]*)\)", label)
    speaker = re.sub(r"\s*\([^)]*\)", "", label).strip()
    mods = [g for g in groups if MODIFIERS.match(g.strip())]
    emo = [g for g in groups if not MODIFIERS.match(g.strip())]
    return speaker, mods, (emo[-1].strip() if emo else None)


def parse_episode(path, fallback_num=None):
    text = open(path, encoding="utf-8", errors="replace").read()
    base = os.path.basename(path)
    m = re.search(r"episode[-_ ]?(\d+)", base, re.I) or re.search(r"(\d+)", base)
    num = int(m.group(1)) if m else (fallback_num or 0)
    ep = {"episode": num, "file": os.path.basename(path), "warnings": []}

    for key, name in (("logline", "EPISODE LOGLINE"), ("continuity", "CONTINUITY NOTES")):
        m = re.search(rf"^{name}:\s*(.+)$", text, re.I | re.M)
        ep[key] = m.group(1).strip() if m else ""
        if not m:
            ep["warnings"].append(f"missing {name}")

    body = text.split("--- SCRIPT START ---", 1)[-1].split("--- SCRIPT END ---", 1)[0]
    if "--- SCRIPT START ---" not in text:
        ep["warnings"].append("no SCRIPT START marker")

    # scenes (case-insensitive)
    scenes = []
    for blk in re.split(r"(?im)^SCENE\s+\d+\s*$", body)[1:]:
        h = re.search(r"(?i)^scene heading:\s*(.+)$", blk, re.M)
        s = re.search(r"(?i)^scene start:\s*(.+)$", blk, re.M)
        e = re.search(r"(?i)^scene end:\s*(.+)$", blk, re.M)
        scenes.append({"heading": h.group(1).strip() if h else "",
                       "start": parse_time(s.group(1)) if s else None,
                       "end": parse_time(e.group(1)) if e else None})

    # events: **LABEL:** / Start: (t) | End: (t) / content lines
    events = []
    pat = re.compile(r"^\*\*(.+?):\*\*\s*$", re.M)
    heads = list(pat.finditer(body))
    for i, m in enumerate(heads):
        chunk = body[m.end(): heads[i + 1].start() if i + 1 < len(heads) else len(body)]
        label = m.group(1).strip()
        tm = re.search(r"Start:\s*\(([^)]*)\)\s*\|\s*End:\s*\(([^)]*)\)", chunk, re.I)
        if not tm:
            ep["warnings"].append(f"event without times: {label[:40]}")
            continue
        start, end = parse_time(tm.group(1)), parse_time(tm.group(2))
        content = chunk[tm.end():]
        # drop anything from the next SCENE header onward
        content = re.split(r"(?im)^SCENE\s+\d+\s*$", content)[0].strip()
        kind = NON_DIALOGUE.get(label.upper(), "dialogue")
        ev = {"kind": kind, "start": start, "end": end, "text": " ".join(content.split())}
        if kind == "dialogue":
            ev["speaker"], ev["mods"], ev["emotion"] = parse_label(label)
            ev["text"] = ev["text"].strip('"“” ')
        events.append(ev)

    # scene end times can be missing: fall back to event extents
    ep["events"] = events
    ep["scenes"] = scenes
    ep["duration"] = max([e["end"] for e in events if e["end"] is not None] +
                         [s["end"] for s in scenes if s["end"] is not None] + [0])
    return ep


def validate(ep):
    w = ep["warnings"]
    if not ep["events"]: w.append("no events parsed")
    if not any(e["kind"] == "dialogue" for e in ep["events"]): w.append("no dialogue")
    if not ep["scenes"]: w.append("no scenes parsed")
    for e in ep["events"]:
        if e["start"] is None or e["end"] is None or e["end"] < e["start"]:
            w.append(f"bad times {e['start']}-{e['end']} ({e['kind']})")
        elif e["start"] < 0 or e["end"] > ep["duration"] + 0.01:
            w.append("event past episode end")
    if ep["duration"] < 20 or ep["duration"] > 300:
        w.append(f"odd duration {ep['duration']}s")
    return ep


def fmt(t):
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def render_for_llm(ep, include_camera=False):
    """Compact text view: dialogue + action (camera/sound dropped) with timestamps."""
    out = [f"EPISODE {ep['episode']} | {fmt(ep['duration'])} long | {ep['logline']}"]
    for e in sorted(ep["events"], key=lambda x: (x["start"], x["end"])):
        if e["kind"] in ("camera", "sound") and not include_camera:
            continue
        t = f"[{fmt(e['start'])}-{fmt(e['end'])}]"
        if e["kind"] == "dialogue":
            tag = f" ({e['emotion']})" if e.get("emotion") else ""
            vo = " (V.O.)" if any("V" in m.upper() for m in e["mods"]) else ""
            out.append(f'{t} {e["speaker"]}{vo}{tag}: "{e["text"]}"')
        else:
            out.append(f"{t} {e['kind'].upper()}: {e['text']}")
    return "\n".join(out)


def main(inp, outdir):
    os.makedirs(outdir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(inp, "**", "*.txt"), recursive=True))
    report, total_w = [], 0
    for i, f in enumerate(files, 1):
        ep = validate(parse_episode(f, fallback_num=i))
        json.dump(ep, open(os.path.join(outdir, f"episode_{ep['episode']:03d}.json"), "w"), indent=1)
        n = len(ep["warnings"]); total_w += n
        report.append((ep["episode"], ep["duration"], len(ep["scenes"]), len(ep["events"]),
                       sum(e["kind"] == "dialogue" for e in ep["events"]), n))
        for w in ep["warnings"]:
            print(f"  ep{ep['episode']:03d} WARN: {w}")
    print(f"\nparsed {len(files)} episodes, {total_w} warnings")
    return report


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
