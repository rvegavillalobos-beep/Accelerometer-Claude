"""Whole route: the recordings of the line segments added up, as if one part had travelled all of them.

Each recording is simulated on its own, starting with the cluster in its nominal position on the GOT. The cluster
shift is a translation of the centre of mass plus a rotation, in the carrier (GOT) frame, which is the same for every
recording. For shifts of millimetres on a 2 m cluster the shifts of the segments simply add up:

    U = sum(u_i),   phi = sum(phi_i),   corner shift c = U + (R(phi) - I) r_c

with r_c the corner position relative to the centre of mass. Shifts in opposite directions cancel, so the shift at
the end of the route can be smaller than the sum of the shifts of the segments.
"""

from __future__ import annotations

import os as _os

_LOADED_MTIME = _os.path.getmtime(__file__)  # used by app.py to detect stale modules

import numpy as np
import pandas as pd


def corner_shifts_mm(dx_mm: float, dy_mm: float, rot_deg: float, rel_m) -> np.ndarray:
    """Displacement magnitude of each corner [mm] for a CoM shift (dx, dy) and a rotation."""
    rel = np.asarray(rel_m, dtype=float) * 1e3  # corner positions relative to the CoM [mm]
    ph = np.deg2rad(rot_deg)
    c, s = np.cos(ph), np.sin(ph)
    rx = (c - 1.0) * rel[:, 0] - s * rel[:, 1]
    ry = s * rel[:, 0] + (c - 1.0) * rel[:, 1]
    return np.hypot(dx_mm + rx, dy_mm + ry)


def accumulate(parts: list[dict], rel_m) -> pd.DataFrame:
    """Shift after every segment, in route order.

    parts: dicts with 'name', 'label', 'dx_mm', 'dy_mm', 'rot_deg', 'corner_mm' (the segment on its own),
    'n_events', 'slip_time_s'. Returns one row per segment with the segment's own shift and the cumulative shift.
    """
    rows = []
    X = Y = R = 0.0
    for p in parts:
        X += p["dx_mm"]
        Y += p["dy_mm"]
        R += p["rot_deg"]
        rows.append(
            {
                "Recording": p["name"],
                "Section": p.get("label", p["name"]),
                "Slip events": int(p.get("n_events", 0)),
                "Slip time [s]": round(float(p.get("slip_time_s", 0.0)), 2),
                "Segment corner shift [mm]": round(float(p["corner_mm"]), 2),
                "Segment rotation [deg]": round(float(p["rot_deg"]), 3),
                "Cumulative dX [mm]": round(X, 2),
                "Cumulative dY [mm]": round(Y, 2),
                "Cumulative rotation [deg]": round(R, 3),
                "Cumulative max corner shift [mm]": round(float(corner_shifts_mm(X, Y, R, rel_m).max()), 2),
            }
        )
    return pd.DataFrame(rows)


def totals(parts: list[dict], rel_m) -> dict:
    X = float(sum(p["dx_mm"] for p in parts))
    Y = float(sum(p["dy_mm"] for p in parts))
    R = float(sum(p["rot_deg"] for p in parts))
    return {
        "dx_mm": X,
        "dy_mm": Y,
        "rot_deg": R,
        "corner_mm": float(corner_shifts_mm(X, Y, R, rel_m).max()) if parts else 0.0,
        "sum_segment_corner_mm": float(sum(p["corner_mm"] for p in parts)),
        "n_events": int(sum(p.get("n_events", 0) for p in parts)),
        "slip_time_s": float(sum(p.get("slip_time_s", 0.0) for p in parts)),
    }


def effective_mu(mus, values, measured: float) -> tuple[float | None, str]:
    """μs at which the predicted value (decreasing with μ) equals the measured one, by linear interpolation.

    Returns (μ, note); μ is None when the measurement lies outside the predicted range."""
    m = np.asarray(mus, dtype=float)
    v = np.asarray(values, dtype=float)
    order = np.argsort(m)
    m, v = m[order], v[order]
    if measured <= 0 or len(m) < 2:
        return None, ""
    if measured > v.max():
        return None, f"larger than predicted for every μs in the range (≥ {m[np.argmax(v)]:.3f}): μs below {m[0]:.3f}"
    if measured < v.min():
        return None, f"smaller than predicted for every μs in the range: μs above {m[-1]:.3f}"
    for i in range(len(m) - 1):
        a, b = v[i], v[i + 1]
        if (a - measured) * (b - measured) <= 0 and a != b:
            return float(m[i] + (measured - a) / (b - a) * (m[i + 1] - m[i])), ""
        if a == measured:
            return float(m[i]), ""
    return float(m[-1]), ""
