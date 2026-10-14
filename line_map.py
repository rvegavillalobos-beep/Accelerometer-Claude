"""Line layout map: station positions on a plant layout image and line sections from recording file names.

* Station labels ("ST 540", "FT 002", two-line "ST / 390", vertical labels) are read from the layout image with
  Tesseract OCR. Only the darkest pixels are kept (station labels are black text, the CAD drawing is grey), every
  label is read on its own, and the result is matched against the station numbers that appear in the file names.
  Labels that cannot be read exactly (e.g. partly covered by an equipment number) are reported as missing and are
  placed by hand with one click; they are never guessed from a similar number, which is often another station.
* Recording file names follow a simple convention: ``St<from>-<to> [note]`` for a section between two stations,
  ``St<station> [note]`` for a single station. ``St981-Buffer`` (a named place) and ``St540Loop540`` (a loop that
  starts and ends at one station) are understood as well.
* Coordinates are image pixels of the original layout (x to the right, y downwards).
"""

from __future__ import annotations

import os as _os

_LOADED_MTIME = _os.path.getmtime(__file__)  # used by app.py to detect stale modules

import base64
import csv
import io
import json
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

# --------------------------------------------------------------------------- #
# Station names
# --------------------------------------------------------------------------- #
_NUM = r"\d{2,4}(?:\.\d{1,2})?"
_TOK = rf"(?:(?:ST|FT)\s*)?{_NUM}"


def station_key(text, default_prefix: str = "ST") -> str | None:
    """Canonical key: 'St 540' / 'ST540' / '540' -> 'ST540', 'ft 2' is not a station, 'Buffer' -> 'BUFFER'."""
    s = re.sub(r"\s+", "", str(text or "").upper())
    if not s:
        return None
    m = re.fullmatch(rf"(ST|FT)?({_NUM})", s)
    if m:
        return (m.group(1) or default_prefix) + m.group(2)
    s = re.sub(r"[^A-Z0-9.]", "", s)
    return s or None


def station_display(key: str | None) -> str:
    """'ST540' -> 'ST 540', 'BUFFER' -> 'Buffer'."""
    if not key:
        return ""
    m = re.fullmatch(rf"(ST|FT)({_NUM})", key)
    if m:
        return f"{m.group(1)} {m.group(2)}"
    return key.title()


def parse_recording_name(name: str) -> dict:
    """Section described by a recording file name.

    'St395-860.txt' -> ST395 to ST860;  'St981-Buffer.txt' -> ST981 to BUFFER;  'St989.txt' -> ST989;
    'St540 deindex.txt' -> ST540, note 'deindex';  'St540Loop540.txt' -> ST540, note 'Loop540'.
    """
    stem = re.sub(r"\s*\(\d+\)\s*$", "", str(name))  # " (2)" added to duplicate uploads
    stem = re.sub(r"\.(txt|csv|tsv|log|dat)$", "", stem, flags=re.I).strip()
    m = re.match(rf"^\s*(?P<a>{_TOK})\s*(?:-|–|—|_|>|→|\s+to\s+)\s*(?P<b>{_TOK})(?![\d.])(?P<rest>.*)$", stem, re.I)
    if m is None:  # second end is a named place, e.g. St981-Buffer (needs a dash or 'to')
        m = re.match(rf"^\s*(?P<a>{_TOK})\s*(?:-|–|—|>|→|\s+to\s+)\s*(?P<b>[A-Za-z][A-Za-z0-9]*)(?P<rest>.*)$", stem, re.I)
    if m is not None:
        a = station_key(m["a"])
        b = station_key(m["b"], default_prefix=a[:2] if a and a[:2] in ("ST", "FT") else "ST")
        return {"From": a, "To": b if b != a else None, "Note": m["rest"].strip(" _-")}
    m = re.match(rf"^\s*(?P<a>{_TOK})(?![\d.])(?P<rest>.*)$", stem, re.I)
    if m is not None:
        return {"From": station_key(m["a"]), "To": None, "Note": m["rest"].strip(" _-")}
    return {"From": None, "To": None, "Note": stem}


def sections_table(recordings: list[str]) -> pd.DataFrame:
    """Default section of every recording, as displayed in the editable table."""
    rows = []
    for rec in recordings:
        p = parse_recording_name(rec)
        rows.append(
            {
                "Recording": rec,
                "From": station_display(p["From"]),
                "To": station_display(p["To"]),
                "Via": "",
                "Note": p["Note"],
            }
        )
    return pd.DataFrame(rows, columns=["Recording", "From", "To", "Via", "Note"])


def section_keys(row) -> tuple[str | None, str | None, list[str]]:
    """Station keys of one row of the sections table (From, To, Via list)."""
    a = station_key(row.get("From"))
    b = station_key(row.get("To"), default_prefix=a[:2] if a and a[:2] in ("ST", "FT") else "ST")
    via = [k for k in (station_key(v) for v in re.split(r"[,;/]+", str(row.get("Via") or ""))) if k]
    if b == a:
        b = None
    return a, b, via


def wanted_stations(sections: pd.DataFrame) -> list[str]:
    keys: list[str] = []
    for _, row in sections.iterrows():
        a, b, via = section_keys(row)
        for k in [a, *via, b]:
            if k and k not in keys:
                keys.append(k)
    return keys


# --------------------------------------------------------------------------- #
# Layout image
# --------------------------------------------------------------------------- #
def _open_gray(image_bytes: bytes) -> np.ndarray:
    from PIL import Image

    im = Image.open(io.BytesIO(image_bytes))
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    return np.asarray(im.convert("L"), dtype=np.uint8)


def image_size(image_bytes: bytes) -> tuple[int, int]:
    from PIL import Image

    with Image.open(io.BytesIO(image_bytes)) as im:
        return int(im.width), int(im.height)


def display_image_uri(image_bytes: bytes, max_side: int = 1800) -> str:
    """Grey-scale PNG data URI of the layout for the map background (down-scaled when very large)."""
    from PIL import Image

    g = _open_gray(image_bytes)
    im = Image.fromarray(g)
    f = max_side / max(im.size)
    if f < 1:
        im = im.resize((max(1, int(im.width * f)), max(1, int(im.height * f))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# --------------------------------------------------------------------------- #
# OCR
# --------------------------------------------------------------------------- #
_WHITELIST = "STF0123456789."
_CONFUSIONS = str.maketrans({"O": "0", "D": "0", "Q": "0", "B": "8", "I": "1", "L": "1", "Z": "2", "G": "6"})


def ocr_available() -> bool:
    return shutil.which("tesseract") is not None


def _tesseract(img, psm: int) -> list[dict]:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    cmd = ["tesseract", "stdin", "stdout", "--psm", str(psm), "-c", f"tessedit_char_whitelist={_WHITELIST}", "tsv"]
    env = {**_os.environ, "OMP_THREAD_LIMIT": "1"}
    try:
        out = subprocess.run(cmd, input=buf.getvalue(), capture_output=True, timeout=120, env=env).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows = list(csv.reader(io.StringIO(out.decode("utf-8", "replace")), delimiter="\t", quoting=csv.QUOTE_NONE))
    if not rows:
        return []
    hdr = rows[0]
    res = []
    for r in rows[1:]:
        if len(r) < 12 or not r[11].strip():
            continue
        d = dict(zip(hdr, r))
        try:
            res.append(
                {
                    "text": d["text"].strip(),
                    "left": int(d["left"]),
                    "top": int(d["top"]),
                    "width": int(d["width"]),
                    "height": int(d["height"]),
                    "conf": float(d["conf"]),
                    "line": (int(d["block_num"]), int(d["par_num"]), int(d["line_num"])),
                }
            )
        except (KeyError, ValueError):
            continue
    return res


def _classify(text: str) -> tuple[str, str] | None:
    """('prefix', 'ST'), ('number', '540'), ('station', 'ST540') or None for noise."""
    t = text.upper().strip(".,:;|!'\"`-_ ")
    if not t:
        return None
    t5 = t.replace("5T", "ST").replace("S7", "ST").replace("F7", "FT")
    if t5 in ("ST", "FT", "S", "T"):
        return ("prefix", "ST" if t5 in ("ST", "S", "T") else "FT")
    m = re.fullmatch(r"(ST|FT)\.?(\S+)", t5)
    if m:
        num = m.group(2).translate(_CONFUSIONS).replace("S", "5")
        if re.fullmatch(_NUM, num):
            return ("station", m.group(1) + num)
    num = t.translate(_CONFUSIONS).replace("S", "5")
    if re.fullmatch(_NUM, num) and sum(c.isdigit() for c in t) >= 2:
        return ("number", num)
    m = re.search(r"(ST|FT)\.?(\d{3}(?:\.\d)?)(?!\d)", t5) if len(t5) <= 9 else None  # e.g. '3ST920'
    if m:
        return ("station", m.group(1) + m.group(2))
    return None


def _tokens_to_labels(tokens: list[dict]) -> list[dict]:
    """Join 'ST' / 'FT' tokens with the number to their right or below (two-line labels)."""
    items = []
    for t in tokens:
        c = _classify(t["text"])
        if c is not None:
            items.append({**t, "kind": c[0], "value": c[1]})
    used = set()
    labels = []
    for i, p in enumerate(items):
        if p["kind"] != "prefix":
            continue
        best, best_d = None, None
        for j, q in enumerate(items):
            if j in used or q["kind"] != "number":
                continue
            h = max(p["h"], q["h"], 1.0)
            dx, dy = q["x"] - p["x"], q["y"] - p["y"]
            right = 0 < dx < 4.5 * h and abs(dy) < 0.6 * h
            below = abs(dx) < 2.5 * h and 0.5 * h < dy < 2.4 * h
            if right or below:
                d = abs(dx) + abs(dy)
                if best_d is None or d < best_d:
                    best, best_d = j, d
        if best is not None:
            q = items[best]
            used.add(best)
            used.add(i)
            labels.append(
                {
                    "key": p["value"] + q["value"],
                    "x": q["x"],
                    "y": 0.5 * (p["y"] + q["y"]),
                    "h": q["h"],
                    "conf": min(p["conf"], q["conf"]),
                    "prefixed": True,
                    "strong_prefix": len(p["text"].strip(".,:;|!'\"`-_ ")) >= 2,  # 'ST', not a lone 'T'
                    "text": f"{p['text']} {q['text']}",
                }
            )
    for i, q in enumerate(items):
        if i in used:
            continue
        if q["kind"] == "station":
            labels.append({**_pick(q), "key": q["value"], "prefixed": True})
        elif q["kind"] == "number" and len(q["value"].split(".")[0]) >= 3:
            labels.append({**_pick(q), "key": "ST" + q["value"], "prefixed": False})
    return labels


def _pick(t: dict) -> dict:
    return {"x": t["x"], "y": t["y"], "h": t["h"], "conf": t["conf"], "text": t["text"]}


def _clean_upscaled(gray: np.ndarray, up: float, dark_threshold: int, pad: int = 24):
    """Enlarge the grey image first and keep only the dark pixels afterwards: smoother letters for the OCR."""
    from PIL import Image

    h, w = gray.shape
    im = Image.fromarray(gray).resize((max(1, int(w * up)), max(1, int(h * up))), Image.BICUBIC)
    b = np.where(np.asarray(im) < dark_threshold, 0, 255).astype(np.uint8)
    if pad:
        b = np.pad(b, pad, constant_values=255)
    return Image.fromarray(b)


def _ocr_sheet(crops: list, psm: int, sheet_width: int = 3000, gap: int = 110) -> list[list[dict]]:
    """OCR many small crops with one Tesseract call: the crops are laid out on one sheet with wide white gaps.

    Returns, for every crop, its tokens with positions in the crop's own pixel coordinates."""
    from PIL import Image

    placed, x, y, row_h = [], gap, gap, 0
    for im in crops:
        w, h = im.size
        if x + w + gap > sheet_width and x > gap:
            x, y, row_h = gap, y + row_h + gap, 0
        placed.append((x, y, w, h))
        x += w + gap
        row_h = max(row_h, h)
    sheet = Image.new("L", (sheet_width, y + row_h + gap), 255)
    for im, (px, py, _w, _h) in zip(crops, placed):
        sheet.paste(im, (px, py))
    out: list[list[dict]] = [[] for _ in crops]
    tol = gap / 3
    for t in _tesseract(sheet, psm):
        l, tp, r, b = t["left"], t["top"], t["left"] + t["width"], t["top"] + t["height"]
        for i, (px, py, w, h) in enumerate(placed):
            # the whole word must lie inside one crop: words glued across crops are dropped
            if px - tol <= l and r <= px + w + tol and py - tol <= tp and b <= py + h + tol:
                out[i].append(dict(t, cx=0.5 * (l + r) - px, cy=0.5 * (tp + b) - py))
                break
    return out


def read_layout_labels(image_bytes: bytes, dark_threshold: int = 90, workers: int = 3) -> list[dict]:
    """Station labels found on the layout: list of {key, x, y, h, conf, prefixed, votes, text} in image pixels.

    Every label is cut out, enlarged and cleaned separately (OCR of small text is much better that way), and read
    in a few variants (scale, darkness threshold, page mode); all labels of one variant go to Tesseract on one
    sheet, so the whole layout takes about a dozen OCR calls. A label read the same way several times gets more
    votes. Vertical labels are turned both ways."""
    if not ocr_available():
        return []
    g = _open_gray(image_bytes)
    H, W = g.shape
    s = max(W, H) / 1000.0  # sizes below are tuned for a layout about 1000 px wide
    ink = g < dark_threshold
    up = max(1.0, 4.0 / s)  # OCR works best with text about 40 px high

    merged = ndi.binary_dilation(ink, structure=np.ones((max(3, int(5 * s)), max(3, int(4 * s))), bool))
    lab, _n = ndi.label(merged)
    boxes = []
    for idx, sl in enumerate(ndi.find_objects(lab)):
        if sl is None:
            continue
        ys, xs = sl
        h, w = ys.stop - ys.start, xs.stop - xs.start
        if not (6 * s <= h <= 70 * s and 5 * s <= w <= 170 * s):
            continue
        if int(ink[ys, xs][lab[ys, xs] == idx + 1].sum()) < 20 * s * s:
            continue
        pad = int(4 * s) + 2
        boxes.append(
            (max(0, xs.start - pad), max(0, ys.start - pad), min(W, xs.stop + pad), min(H, ys.stop + pad), h > 1.3 * w)
        )
    horiz = [b for b in boxes if not b[4]]
    vert = [b for b in boxes if b[4]]

    variants = [(6, up, dark_threshold), (11, up, dark_threshold), (6, up * 1.25, dark_threshold),
                (11, up * 1.25, dark_threshold), (6, up, dark_threshold + 30), (6, up, dark_threshold - 20),
                (11, up, dark_threshold - 20)]
    jobs = []
    for psm, f, thr in variants:
        jobs.append(("h", 0, psm, f, thr))
        for angle in (90, -90):
            jobs.append(("v", angle, psm, f, thr))

    def crop_tokens(box, tl, f) -> list[dict]:
        x0, y0, x1, y1, vertical = box
        if not vertical:
            return [
                {"text": t["text"], "x": x0 + t["cx"] / f, "y": y0 + t["cy"] / f, "h": t["height"] / f, "conf": t["conf"]}
                for t in tl
            ]
        # vertical label: read in order; positions inside a turned crop are not mapped back (centre is enough)
        joined = re.sub(r"\s+", "", "".join(t["text"] for t in sorted(tl, key=lambda t: t["cx"]))).upper()
        joined = joined.replace("5T", "ST")
        st_m = re.search(r"(ST|FT)\.?(\d{3}(?:\.\d)?)(?!\d)", joined)
        num_m = re.search(r"(?<!\d)\d{3}(?!\d)", joined)
        text = st_m.group(1) + st_m.group(2) if st_m else (num_m.group(0) if num_m else "")
        if not text:
            return []
        return [
            {
                "text": text,
                "x": 0.5 * (x0 + x1),
                "y": 0.5 * (y0 + y1),
                "h": max([t["height"] for t in tl], default=0) / f,
                "conf": max([t["conf"] for t in tl], default=0.0),
            }
        ]

    def crop_image(box, f, thr, angle):
        x0, y0, x1, y1, _v = box
        im = _clean_upscaled(g[y0:y1, x0:x1], f, thr, pad=0)
        return im.rotate(angle, expand=True) if angle else im

    def run(job):
        kind, angle, psm, f, thr = job
        sel = horiz if kind == "h" else vert
        if not sel:
            return kind, []
        res = _ocr_sheet([crop_image(b, f, thr, angle) for b in sel], psm)
        return kind, [crop_tokens(b, tl, f) for b, tl in zip(sel, res)]

    labels: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = [ex.submit(run, j) for j in jobs]
        # the whole cleaned image once as well (catches labels that were cut into several pieces)
        big = _clean_upscaled(g, up, dark_threshold, pad=0)
        full = [
            {
                "text": t["text"],
                "x": (t["left"] + 0.5 * t["width"]) / up,
                "y": (t["top"] + 0.5 * t["height"]) / up,
                "h": t["height"] / up,
                "conf": t["conf"],
            }
            for t in _tesseract(big, 11)
        ]
        labels += _tokens_to_labels(full)
        for fut in futures:
            _kind, per_crop = fut.result()
            all_toks = [t for toks in per_crop for t in toks]
            # 'ST' and the number of a two-line label are separate pieces: pair them across pieces
            labels += _tokens_to_labels(all_toks)

    # merge the same key found several times at (almost) the same place: votes
    merged_labels: list[dict] = []
    for lb in sorted(labels, key=lambda d: -d["conf"]):
        twin = next(
            (
                m
                for m in merged_labels
                if m["key"] == lb["key"] and abs(m["x"] - lb["x"]) < 15 * s and abs(m["y"] - lb["y"]) < 15 * s
            ),
            None,
        )
        if twin is None:
            merged_labels.append(dict(lb, votes=1))
        else:
            twin["votes"] += 1
            twin["prefixed"] = twin["prefixed"] or lb["prefixed"]
    for m in merged_labels:
        m["x"], m["y"], m["h"], m["conf"] = float(m["x"]), float(m["y"]), float(m["h"]), float(m["conf"])
        m.pop("strong_prefix", None)
    return merged_labels


# --------------------------------------------------------------------------- #
# Station table
# --------------------------------------------------------------------------- #
def resolve_stations(labels: list[dict], wanted: list[str], manual: dict, removed: list[str]) -> pd.DataFrame:
    """Station positions: stations placed by hand, then labels read by the OCR.

    Only exact reads are used. A label that is partly covered (e.g. by an equipment number) is not guessed from a
    similar reading, because a similar number is often another real station; it is listed as missing instead."""
    rows: dict[str, dict] = {}
    heights = [lb["h"] for lb in labels if lb.get("prefixed")]
    h_ref = float(np.median(heights)) if heights else 0.0

    def score(lb):
        return (lb.get("prefixed", False), lb.get("votes", 1), lb["h"] >= 0.8 * h_ref, lb["conf"])

    by_key: dict[str, list[dict]] = {}
    for lb in labels:
        by_key.setdefault(lb["key"], []).append(lb)
    for key, lbs in by_key.items():
        best = max(lbs, key=score)
        if not best.get("prefixed") and key not in wanted and best["h"] < 0.8 * h_ref:
            continue  # small numbers on the drawing (equipment markers), not station labels
        rows[key] = {
            "Key": key,
            "Station": station_display(key),
            "x": best["x"],
            "y": best["y"],
            "Source": "Detected",
            "Read as": best.get("text", ""),
        }
    for key, m in manual.items():
        rows[key] = {
            "Key": key,
            "Station": m.get("name") or station_display(key),
            "x": float(m["x"]),
            "y": float(m["y"]),
            "Source": "Placed by hand",
            "Read as": "",
        }
    for key in removed:
        if key not in manual:
            rows.pop(key, None)
    df = pd.DataFrame(list(rows.values()), columns=["Key", "Station", "x", "y", "Source", "Read as"])
    df["Needed"] = df["Key"].isin(wanted)
    return df.sort_values(["Needed", "Key"], ascending=[False, True]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Sections on the map
# --------------------------------------------------------------------------- #
def section_paths(sections: pd.DataFrame, stations: pd.DataFrame) -> list[dict]:
    """Geometry of every row of the sections table: polyline through From, Via..., To (or one point)."""
    pos = {r.Key: (float(r.x), float(r.y)) for r in stations[["Key", "x", "y"]].itertuples(index=False)}
    out = []
    for _, row in sections.iterrows():
        a, b, via = section_keys(row)
        keys = [k for k in [a, *via, b] if k]
        missing = [k for k in keys if k not in pos]
        out.append(
            {
                "Recording": row["Recording"],
                "keys": keys,
                "single": b is None,
                "missing": missing,
                "path": [pos[k] for k in keys if k in pos] if not missing else [],
                "label": (
                    station_display(a) + (f" → {station_display(b)}" if b else "") + (f" · {row['Note']}" if row.get("Note") else "")
                    if a
                    else str(row["Recording"])
                ),
                "section": (a, b, tuple(via)),
            }
        )
    return out


def path_midpoint(path: list[tuple[float, float]]) -> tuple[float, float]:
    if len(path) == 1:
        return path[0]
    seg = [np.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in zip(path[:-1], path[1:])]
    half = 0.5 * sum(seg)
    for (p0, p1), L in zip(zip(path[:-1], path[1:]), seg):
        if half <= L and L > 0:
            f = half / L
            return (p0[0] + f * (p1[0] - p0[0]), p0[1] + f * (p1[1] - p0[1]))
        half -= L
    return path[-1]


# --------------------------------------------------------------------------- #
# Save / load
# --------------------------------------------------------------------------- #
def pack_layout_map(state: dict) -> bytes:
    """Layout image, OCR result, hand placements and section table in one JSON file."""
    doc = {
        "format": "got-line-map/1",
        "image_name": state.get("image_name", "layout.png"),
        "image": base64.b64encode(state["image"]).decode() if state.get("image") else None,
        "labels": state.get("labels") or [],
        "manual": state.get("manual") or {},
        "removed": sorted(state.get("removed") or []),
        "sections": state.get("sections_effective") or {},
        "threshold": state.get("threshold", 90),
    }
    return json.dumps(doc, indent=1).encode()


def unpack_layout_map(data: bytes) -> dict:
    doc = json.loads(data.decode("utf-8"))
    if doc.get("format", "").split("/")[0] != "got-line-map":
        raise ValueError("Not a layout map file saved by this app.")
    return {
        "image": base64.b64decode(doc["image"]) if doc.get("image") else None,
        "image_name": doc.get("image_name", "layout.png"),
        "labels": doc.get("labels") or [],
        "manual": {k: dict(v) for k, v in (doc.get("manual") or {}).items()},
        "removed": list(doc.get("removed") or []),
        "sections": doc.get("sections") or {},
        "threshold": int(doc.get("threshold", 90)),
    }
