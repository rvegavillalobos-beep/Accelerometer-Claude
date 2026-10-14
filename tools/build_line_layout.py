"""Build the embedded line layout from the drawing of the conveyor path.

Run once whenever the layout drawing changes (not used by the app at runtime):

    pip install opencv-python-headless scikit-image scipy numpy pillow
    python tools/build_line_layout.py

Inputs (assets/source/):
    line_layout_paths.png   layout with the conveyor path drawn in red. White boxes with a red border carry the
                            name of a station whose label is hidden (e.g. "ST395"); the light grey-blue box with
                            the word "buffer" is the buffer.
    layout_labels.png       the same layout without the red path (any scale / crop): used to restore the drawing
                            under the red lines in the background image.

Outputs:
    assets/line_layout.png  background image (red path removed)
    line_layout_data.py     conveyor graph (nodes, edges with polylines) and the station table

Station positions: STATION_LABELS below gives the centre of every station label in line_layout_paths.png pixels.
Each station is attached to the nearest point of the drawn path (stations farther than MAX_SNAP_PX are rejected).
To add or move a station, edit STATION_LABELS and run the script again.
"""

from __future__ import annotations

import io
import pprint
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parents[1]
SRC_PATHS = ROOT / "assets" / "source" / "line_layout_paths.png"
SRC_CLEAN = ROOT / "assets" / "source" / "layout_labels.png"
OUT_IMAGE = ROOT / "assets" / "line_layout.png"
OUT_DATA = ROOT / "line_layout_data.py"

MAX_SNAP_PX = 45.0
BUFFER_COLOR = (222, 229, 236)

# Centre of each station label in line_layout_paths.png (x, y) [px]. Boxes drawn by hand are found automatically.
STATION_LABELS: dict[str, tuple[float, float]] = {
    "FT001": (199, 111),
    "FT002": (333, 105),
    "FT003": (420, 225),
    "FT004": (427, 356),
    "ST850": (144, 153),
    "ST390": (166, 259),
    "ST820": (131, 312),
    "ST830": (199, 305),
    "ST829": (100, 343),
    "ST820.1": (140, 350),
    "ST381": (232, 360),
    "ST380": (137, 429),
    "ST810.1": (140, 460),
    "ST810": (205, 445),
    "ST780": (346, 494),
    "ST130": (354, 438),
    "ST870": (517, 211),
    "ST410": (488, 262),
    "ST400": (485, 323),
    "ST860": (497, 370),
    "ST430": (547, 291),
    "ST880": (612, 282),
    "ST431": (660, 278),
    "ST433": (655, 392),
    "ST890": (605, 410),
    "ST900": (593, 490),
    "ST910": (712, 485),
    "ST450": (756, 485),
    "ST920": (804, 476),
    "ST460": (803, 398),
    "ST940": (779, 341),
    "ST470": (864, 303),
    "ST950": (932, 303),
    "ST540": (876, 386),
    "ST500": (969, 388),
    "ST960": (903, 431),
    "ST961": (948, 490),
    "ST490": (910, 497),
    "ST491": (911, 535),
    "ST962": (869, 546),
    "ST970": (915, 588),
    "ST550": (1017, 566),
    "ST981": (1016, 460),
    "ST989": (1053, 494),
    "ST710": (1066, 212),
}
NAMED_PLACES = {"BUFFER": "Buffer"}  # key -> display name of places that are not numbered stations

NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


# --------------------------------------------------------------------------- #
# Image analysis
# --------------------------------------------------------------------------- #
def red_mask(rgb: np.ndarray) -> np.ndarray:
    a = rgb.astype(int)
    return (a[..., 0] - np.maximum(a[..., 1], a[..., 2])) > 30


def find_buffer(rgb: np.ndarray) -> dict:
    a = rgb.astype(int)
    m = np.all(np.abs(a - np.array(BUFFER_COLOR)) <= 4, axis=2)
    lab, n = ndi.label(m)
    sizes = ndi.sum(m, lab, range(1, n + 1))
    ys, xs = np.nonzero(lab == int(np.argmax(sizes)) + 1)
    return {"x0": int(xs.min()), "y0": int(ys.min()), "x1": int(xs.max()) + 1, "y1": int(ys.max()) + 1}


def ocr_line(gray: np.ndarray) -> str:
    im = Image.fromarray(gray)
    im = im.resize((im.width * 4, im.height * 4), Image.BICUBIC)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    out = subprocess.run(["tesseract", "stdin", "stdout", "--psm", "7"], input=buf.getvalue(), capture_output=True)
    return out.stdout.decode().strip().replace(" ", "").upper()


def find_name_boxes(rgb: np.ndarray, red: np.ndarray) -> list[dict]:
    """White boxes framed in red, with the station name written inside."""
    gray = np.asarray(Image.fromarray(rgb).convert("L"))
    inside = ndi.binary_fill_holes(red) & ~red
    lab, _n = ndi.label(inside)
    boxes = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        region = lab[sl] == i
        if region.sum() < 300 or (gray[sl][region] >= 245).mean() < 0.8:
            continue
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        boxes.append({"key": ocr_line(gray[y0 + 3 : y1 - 3, x0 + 3 : x1 - 3]), "x0": x0, "y0": y0, "x1": x1, "y1": y1})
    return boxes


def degree(sk: np.ndarray) -> np.ndarray:
    return (ndi.convolve(sk.astype(int), np.ones((3, 3), int), mode="constant") - sk) * sk


def prune_spurs(sk: np.ndarray, min_len: int = 8) -> np.ndarray:
    """Remove short dead ends that branch off a line (overshoots at hand-drawn corners)."""
    sk = sk.copy()
    for _ in range(4):
        deg = degree(sk)
        changed = False
        for y, x in np.argwhere(deg == 1):
            chain, cur, prev = [(y, x)], (y, x), None
            while len(chain) <= min_len:
                nbrs = [
                    (cur[0] + dy, cur[1] + dx)
                    for dy, dx in NB8
                    if 0 <= cur[0] + dy < sk.shape[0]
                    and 0 <= cur[1] + dx < sk.shape[1]
                    and sk[cur[0] + dy, cur[1] + dx]
                    and (cur[0] + dy, cur[1] + dx) not in chain
                ]
                if len(nbrs) != 1 or deg[nbrs[0]] >= 3:
                    break
                prev, cur = cur, nbrs[0]
                chain.append(cur)
            last = chain[-1]
            touches_junction = any(
                0 <= last[0] + dy < sk.shape[0]
                and 0 <= last[1] + dx < sk.shape[1]
                and deg[last[0] + dy, last[1] + dx] >= 3
                and (last[0] + dy, last[1] + dx) not in chain
                for dy, dx in NB8
            )
            if len(chain) <= min_len and touches_junction:
                for p in chain:
                    sk[p] = False
                changed = True
        if not changed:
            break
    return sk


def skeleton_graph(sk: np.ndarray) -> tuple[list[list[float]], list[dict]]:
    """Nodes (line ends and junctions) and edges (pixel chains simplified to polylines)."""
    deg = degree(sk)
    lab, n = ndi.label(sk & (deg != 2), structure=np.ones((3, 3)))
    nodes, node_of = [], {}
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        nodes.append([float(xs.mean()), float(ys.mean())])
        for p in zip(ys.tolist(), xs.tolist()):
            node_of[p] = i - 1
    edges, seen = [], set()
    for p, i in list(node_of.items()):
        for dy, dx in NB8:
            q = (p[0] + dy, p[1] + dx)
            if not (0 <= q[0] < sk.shape[0] and 0 <= q[1] < sk.shape[1]) or not sk[q]:
                continue
            if q in node_of:
                j = node_of[q]
                if j != i and (min(i, j), max(i, j), "direct") not in seen:
                    seen.add((min(i, j), max(i, j), "direct"))
                    edges.append({"a": i, "b": j, "px": [p, q]})
                continue
            chain, prev, cur, j = [p, q], p, q, None
            while True:
                nxt = [
                    (cur[0] + ddy, cur[1] + ddx)
                    for ddy, ddx in NB8
                    if 0 <= cur[0] + ddy < sk.shape[0]
                    and 0 <= cur[1] + ddx < sk.shape[1]
                    and sk[cur[0] + ddy, cur[1] + ddx]
                    and (cur[0] + ddy, cur[1] + ddx) != prev
                    and (cur[0] + ddy, cur[1] + ddx) not in chain
                ]
                other = [t for t in nxt if t in node_of and node_of[t] != i]
                same = [t for t in nxt if t in node_of and node_of[t] == i]
                if other:
                    j = node_of[other[0]]
                    chain.append(other[0])
                    break
                if same:
                    if len(chain) > 6:  # a real loop back to the same junction
                        j = i
                        chain.append(same[0])
                    break  # otherwise a step around the junction itself, not an edge
                free = [t for t in nxt if t not in node_of]
                if not free:
                    break
                prev, cur = cur, free[0]
                chain.append(cur)
            if j is None:
                continue
            key = (min(i, j), max(i, j), tuple(sorted([chain[1], chain[-2]])))
            if key in seen:
                continue
            seen.add(key)
            edges.append({"a": i, "b": j, "px": chain})
    for e in edges:
        pts = np.array([(x, y) for y, x in e.pop("px")], dtype=np.float32)
        poly = cv2.approxPolyDP(pts.reshape(-1, 1, 2), 1.2, False).reshape(-1, 2)
        e["pts"] = [[float(x), float(y)] for x, y in poly]
        e["pts"][0], e["pts"][-1] = list(nodes[e["a"]]), list(nodes[e["b"]])
    return nodes, edges


# --------------------------------------------------------------------------- #
# Graph editing
# --------------------------------------------------------------------------- #
def poly_len(pts) -> float:
    p = np.asarray(pts, float)
    return float(np.hypot(*np.diff(p, axis=0).T).sum()) if len(p) > 1 else 0.0


def project(pts, x, y):
    """Nearest point of a polyline: (distance, segment index, point)."""
    best = (np.inf, 0, None)
    for i in range(len(pts) - 1):
        (x0, y0), (x1, y1) = pts[i], pts[i + 1]
        dx, dy = x1 - x0, y1 - y0
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 == 0 else min(1.0, max(0.0, ((x - x0) * dx + (y - y0) * dy) / L2))
        px, py = x0 + t * dx, y0 + t * dy
        d = float(np.hypot(x - px, y - py))
        if d < best[0]:
            best = (d, i, [float(px), float(py)])
    return best


def split_edge(nodes, edges, k, seg, pt, snap_node_px=4.0) -> int:
    """Insert a node on edge k at point pt (or reuse an end node that is very close)."""
    e = edges[k]
    for end in (e["a"], e["b"]):
        if np.hypot(nodes[end][0] - pt[0], nodes[end][1] - pt[1]) <= snap_node_px:
            return end
    nodes.append(list(pt))
    m = len(nodes) - 1
    a_pts = e["pts"][: seg + 1] + [list(pt)]
    b_pts = [list(pt)] + e["pts"][seg + 1 :]
    edges[k] = {"a": e["a"], "b": m, "pts": a_pts}
    edges.append({"a": m, "b": e["b"], "pts": b_pts})
    return m


def nearest_edge(edges, x, y, exclude_node=None):
    best = (np.inf, None, None, None)
    for k, e in enumerate(edges):
        if exclude_node is not None and exclude_node in (e["a"], e["b"]):
            continue
        d, seg, pt = project(e["pts"], x, y)
        if d < best[0]:
            best = (d, k, seg, pt)
    return best


def degrees(nodes, edges):
    deg = [0] * len(nodes)
    for e in edges:
        deg[e["a"]] += 1
        deg[e["b"]] += 1
    return deg


def inside(box, x, y, m):
    return box["x0"] - m <= x <= box["x1"] + m and box["y0"] - m <= y <= box["y1"] + m


# --------------------------------------------------------------------------- #
# Background image
# --------------------------------------------------------------------------- #
def background(rgb: np.ndarray, red: np.ndarray, boxes: list[dict], buf: dict) -> np.ndarray:
    """Red path removed: pixels under the lines come from the label layout (registered), else inpainted."""
    H, W = red.shape
    gray_v2 = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    line = red.copy()
    border = np.zeros_like(red)
    for b in boxes:
        sl = (slice(max(0, b["y0"] - 4), b["y1"] + 4), slice(max(0, b["x0"] - 4), b["x1"] + 4))
        border[sl] = red[sl]
    line &= ~border
    squiggle = np.zeros_like(red)
    squiggle[buf["y0"] : buf["y1"], buf["x0"] : buf["x1"]] = red[buf["y0"] : buf["y1"], buf["x0"] : buf["x1"]]
    line &= ~squiggle
    mask = cv2.dilate(line.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    out = cv2.inpaint(rgb, mask.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
    if SRC_CLEAN.exists():
        old = np.asarray(Image.open(SRC_CLEAN).convert("L"))
        orb = cv2.ORB_create(8000)
        k1, d1 = orb.detectAndCompute(old, None)
        k2, d2 = orb.detectAndCompute(cv2.cvtColor(out, cv2.COLOR_RGB2GRAY), None)
        good = [
            p[0]
            for p in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
            if len(p) == 2 and p[0].distance < 0.8 * p[1].distance
        ]
        p1 = np.float32([k1[g.queryIdx].pt for g in good])
        p2 = np.float32([k2[g.trainIdx].pt for g in good])
        M, inl = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=10000)
        if M is not None and inl is not None and inl.sum() > 200:
            warp = cv2.warpAffine(old, M, (W, H), flags=cv2.INTER_CUBIC, borderValue=255)
            covered = cv2.warpAffine(np.full_like(old, 255), M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0) > 0
            use = mask & covered
            out[use] = np.repeat(warp[use][:, None], 3, axis=1)
            print(f"  background: {int(use.sum())} line pixels restored from {SRC_CLEAN.name}, "
                  f"{int((mask & ~covered).sum())} inpainted")
    out[border] = (70, 70, 70)  # name boxes keep a dark frame
    out[squiggle] = BUFFER_COLOR  # spell-check marks under the word 'buffer'
    del gray_v2
    return out


# --------------------------------------------------------------------------- #
def main() -> None:
    rgb = np.asarray(Image.open(SRC_PATHS).convert("RGB")).copy()
    H, W = rgb.shape[:2]
    red = red_mask(rgb)
    buf = find_buffer(rgb)
    boxes = find_name_boxes(rgb, red)
    print("name boxes:", [b["key"] for b in boxes], "| buffer:", buf)

    path = red.copy()
    for b in boxes:
        path[max(0, b["y0"] - 6) : b["y1"] + 6, max(0, b["x0"] - 6) : b["x1"] + 6] = False
    path[buf["y0"] - 2 : buf["y1"] + 2, buf["x0"] - 2 : buf["x1"] + 2] = False
    lab, n = ndi.label(path, structure=np.ones((3, 3)))
    sizes = ndi.sum(path, lab, range(1, n + 1))
    for i, s in enumerate(sizes, start=1):
        if s < 60:
            path[lab == i] = False
    nodes, edges = skeleton_graph(prune_spurs(skeletonize(path)))
    print(f"skeleton graph: {len(nodes)} nodes, {len(edges)} edges")

    # name boxes and the buffer become nodes; line ends touching them are connected to them
    special: dict[str, int] = {}
    deg = degrees(nodes, edges)
    ends = [i for i, d in enumerate(deg) if d == 1]
    for b in boxes + [dict(buf, key="BUFFER")]:
        touching = [i for i in ends if inside(b, nodes[i][0], nodes[i][1], 10)]
        if not touching:
            print("  WARNING: nothing connects to", b["key"])
            continue
        cx, cy = 0.5 * (b["x0"] + b["x1"]), 0.5 * (b["y0"] + b["y1"])
        if b["key"] != "BUFFER":
            xs = [nodes[i][0] for i in touching]
            ys = [nodes[i][1] for i in touching]
            vertical = len(touching) > 1 and (max(ys) - min(ys)) > (max(xs) - min(xs))
            cx, cy = (float(np.mean(xs)), cy) if vertical else (cx, float(np.mean(ys)))
        nodes.append([cx, cy])
        m = len(nodes) - 1
        special[b["key"]] = m
        for i in touching:
            x, y = nodes[i]
            pts = [[x, y], [cx, y], [cx, cy]] if b["key"] == "BUFFER" else [[x, y], [cx, cy]]
            edges.append({"a": i, "b": m, "pts": pts})
        ends = [i for i in ends if i not in touching]

    # small gaps of the hand drawing: line ends close to another end or to another line
    for i in list(ends):
        x, y = nodes[i]
        others = [j for j in ends if j != i and np.hypot(nodes[j][0] - x, nodes[j][1] - y) <= 8]
        if others:
            j = others[0]
            if not any({e["a"], e["b"]} == {i, j} for e in edges):
                edges.append({"a": i, "b": j, "pts": [list(nodes[i]), list(nodes[j])]})
                print(f"  bridged line ends {i}-{j}")
            continue
        d, k, seg, pt = nearest_edge(edges, x, y, exclude_node=i)
        if k is not None and d <= 10:
            m = split_edge(nodes, edges, k, seg, pt)
            edges.append({"a": i, "b": m, "pts": [list(nodes[i]), list(nodes[m])]})
            print(f"  joined line end {i} to the line {d:.1f} px away")

    # stations: attached to the nearest point of the path
    stations: dict[str, list] = {}
    for key, m in special.items():
        stations[key] = [NAMED_PLACES.get(key, display(key)), m, 0.0]
    for key, (x, y) in STATION_LABELS.items():
        if key in stations:
            continue
        d, k, seg, pt = nearest_edge(edges, x, y)
        if d > MAX_SNAP_PX:
            print(f"  {key}: {d:.0f} px from the path, left out")
            continue
        m = split_edge(nodes, edges, k, seg, pt)
        stations[key] = [display(key), m, round(float(d), 1)]

    for e in edges:
        e["len"] = poly_len(e["pts"])
    deg = degrees(nodes, edges)
    print(f"final graph: {len(nodes)} nodes, {len(edges)} edges, {len(stations)} stations, "
          f"line ends: {sum(1 for d in deg if d == 1)}")

    Image.fromarray(background(rgb, red, boxes, buf)).convert("L").save(OUT_IMAGE, optimize=True)
    write_data(W, H, nodes, edges, stations)
    print("wrote", OUT_IMAGE.relative_to(ROOT), "and", OUT_DATA.relative_to(ROOT))


def display(key: str) -> str:
    import re

    m = re.fullmatch(r"(ST|FT)(\d+(?:\.\d+)?)", key)
    return f"{m.group(1)} {m.group(2)}" if m else key.title()


def write_data(W, H, nodes, edges, stations) -> None:
    r = lambda v: round(float(v), 1)  # noqa: E731
    node_list = [(r(x), r(y)) for x, y in nodes]
    edge_list = [(e["a"], e["b"], r(e["len"]), [(r(x), r(y)) for x, y in e["pts"]]) for e in edges]
    st = {k: (v[0], v[1]) for k, v in sorted(stations.items())}
    text = (
        '"""Conveyor line layout embedded in the app. Generated by tools/build_line_layout.py from\n'
        "assets/source/line_layout_paths.png - edit STATION_LABELS in that script and run it again instead of\n"
        'editing this file by hand.\n\nCoordinates are pixels of assets/line_layout.png (x to the right, y downwards)."""\n\n'
        "import os as _os\n\n_LOADED_MTIME = _os.path.getmtime(__file__)  # used by app.py to detect stale modules\n\n"
        f"IMAGE_FILE = \"assets/line_layout.png\"\nIMAGE_SIZE = ({W}, {H})\n\n"
        "# station key -> (display name, node)\n"
        f"STATIONS = {pprint.pformat(st, width=118, sort_dicts=True)}\n\n"
        "# node -> (x, y)\n"
        f"NODES = {pprint.pformat(node_list, width=118, compact=True)}\n\n"
        "# (node a, node b, length [px], polyline [(x, y), ...]) - the conveyor path between two nodes\n"
        f"EDGES = {pprint.pformat(edge_list, width=118, compact=True)}\n"
    )
    OUT_DATA.write_text(text)


if __name__ == "__main__":
    main()
