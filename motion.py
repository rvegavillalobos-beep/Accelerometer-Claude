"""
motion.py
=========

Travel direction of the workpiece carrier along the conveyor.

The direction of travel cannot be read from the inertial load alone: the
inertial load points against the acceleration, so it vanishes at constant
speed and points backwards when the carrier speeds up and forwards when it
brakes. Integrating the acceleration over long periods does not work either,
because small slopes of the conveyor and sensor offsets make the velocity
drift.

This module therefore estimates the travel direction per motion segment:

1. A carrier rolling on a conveyor vibrates; at standstill the signal is quiet.
   Motion segments are found from the vibration level of the vertical
   acceleration.
2. For each segment the velocity is integrated only over a short window after
   the start (from rest) and before the stop (back to rest). Both windows give
   an independent estimate of the travel direction; when they agree the
   estimate is reliable.
3. Conveyors move the carrier along one of its own axes, so the direction is
   snapped to +X, -X, +Y or -Y of the carrier (optional).

Directions can be overridden manually (e.g. from video).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

from data_io import G0, Kinematics
from slip_model import SimulationResult

DIRECTIONS = {
    "+X (Front)": (1.0, 0.0),
    "-X (Rear)": (-1.0, 0.0),
    "+Y (Left)": (0.0, 1.0),
    "-Y (Right)": (0.0, -1.0),
}
STATE_STOPPED, STATE_MOVING, STATE_UNKNOWN, STATE_ROTATING = 0, 1, 2, 3


@dataclass(frozen=True)
class TravelSettings:
    vib_threshold_g: float = 0.008  # vertical vibration (1-s RMS) above this = carrier rolling
    min_move_s: float = 1.5  # shorter motion segments are ignored
    merge_gap_s: float = 1.0  # quiet gaps shorter than this do not split a segment
    window_s: float = 2.5  # integration window after the start / before the stop
    min_speed: float = 0.08  # m/s, minimum speed for a window to count as evidence
    snap: bool = True  # snap the direction to the carrier axes


@dataclass(frozen=True)
class MotionSegment:
    i0: int
    i1: int
    t0: float
    t1: float
    auto_label: str  # direction label, "Uncertain" or "Rotating (turntable)"
    confidence: str  # "High" | "Medium" | "Low" | "-"
    speed: float  # m/s (estimate), NaN if unknown
    ux: float  # unit direction (carrier frame), NaN if unknown
    uy: float


def _rstd(x: np.ndarray, w: int) -> np.ndarray:
    m = uniform_filter1d(x, w, mode="nearest")
    m2 = uniform_filter1d(x * x, w, mode="nearest")
    return np.sqrt(np.clip(m2 - m * m, 0.0, None))


def _snap(vx: float, vy: float) -> str:
    ang = np.rad2deg(np.arctan2(vy, vx))
    k = int(np.round(ang / 90.0)) % 4
    return ["+X (Front)", "+Y (Left)", "-X (Rear)", "-Y (Right)"][k]


def _unit(vx: float, vy: float) -> tuple[float, float]:
    n = float(np.hypot(vx, vy))
    return (vx / n, vy / n) if n > 0 else (np.nan, np.nan)


def detect_motion(
    kin: Kinematics, res: SimulationResult, rotations: list, s: TravelSettings = TravelSettings()
) -> list[MotionSegment]:
    """Motion segments with their estimated travel direction (carrier frame)."""
    fs = kin.fs
    n = len(kin.t)
    vib = _rstd(kin.fz, max(3, int(round(fs))))
    moving = vib > s.vib_threshold_g
    d = np.diff(np.concatenate([[0], moving.astype(int), [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0] - 1
    segs: list[list[int]] = []
    gap = int(round(s.merge_gap_s * fs))
    for a, b in zip(starts, ends):
        if segs and a - segs[-1][1] <= gap:
            segs[-1][1] = b
        else:
            segs.append([int(a), int(b)])
    segs = [g for g in segs if (g[1] - g[0]) / fs >= s.min_move_s]

    # acceleration at the cluster CoM (turntable rotation terms removed when the sensor offset is set)
    ax = G0 * res.fhx_com
    ay = G0 * res.fhy_com
    rot_mask = np.zeros(n, dtype=bool)
    for r in rotations:
        if r.kind == "Turntable":
            rot_mask[r.i0 : r.i1 + 1] = True

    out: list[MotionSegment] = []
    pre = int(round(0.5 * fs))
    for a, b in segs:
        dur = (b - a) / fs
        if rot_mask[a : b + 1].mean() > 0.5:
            out.append(MotionSegment(a, b, kin.t[a], kin.t[b], "Rotating (turntable)", "-", np.nan, np.nan, np.nan))
            continue
        w = int(round(min(s.window_s, dur / 2.0) * fs))
        a0 = max(0, a - pre)
        b1 = min(n, b + pre + 1)
        vs = np.array([ax[a0 : a + w].sum(), ay[a0 : a + w].sum()]) / fs  # velocity after the start
        ve = -np.array([ax[b - w + 1 : b1].sum(), ay[b - w + 1 : b1].sum()]) / fs  # velocity before the stop
        good_s = np.hypot(*vs) >= s.min_speed
        good_e = np.hypot(*ve) >= s.min_speed
        ls, le = _snap(*vs), _snap(*ve)
        if good_s and good_e:
            agree = ls == le if s.snap else float(np.dot(_unit(*vs), _unit(*ve))) > 0.7
            if agree:
                conf, v = "High", 0.5 * (vs + ve)
            else:
                conf, v = "Low", vs if np.hypot(*vs) >= np.hypot(*ve) else ve
        elif good_s or good_e:
            conf, v = "Medium", vs if good_s else ve
        else:
            out.append(MotionSegment(a, b, kin.t[a], kin.t[b], "Uncertain", "Low", np.nan, np.nan, np.nan))
            continue
        if s.snap:
            label = _snap(*v)
            ux, uy = DIRECTIONS[label]
        else:
            ux, uy = _unit(*v)
            label = f"{np.rad2deg(np.arctan2(uy, ux)):+.0f}° ({_snap(*v)})"
        speed = float(np.mean([np.hypot(*x) for x, g in ((vs, good_s), (ve, good_e)) if g]))
        out.append(MotionSegment(a, b, kin.t[a], kin.t[b], label, conf, speed, float(ux), float(uy)))
    return out


def segment_key(seg: MotionSegment) -> str:
    return f"{seg.t0:.1f}"


def travel_arrays(
    kin: Kinematics, segs: list[MotionSegment], rotations: list, overrides: dict | None = None
) -> dict[str, np.ndarray]:
    """Per-sample travel state, direction (carrier frame) and speed, with manual overrides applied."""
    n = len(kin.t)
    state = np.full(n, STATE_STOPPED, dtype=int)
    ux = np.full(n, np.nan)
    uy = np.full(n, np.nan)
    speed = np.full(n, np.nan)
    label = np.full(n, "", dtype=object)
    conf = np.full(n, "", dtype=object)
    overrides = overrides or {}
    for g in segs:
        sl = slice(g.i0, g.i1 + 1)
        ov = overrides.get(segment_key(g), "Auto")
        if ov == "Stopped":
            continue
        if ov in DIRECTIONS:
            state[sl] = STATE_MOVING
            ux[sl], uy[sl] = DIRECTIONS[ov]
            speed[sl] = g.speed
            label[sl] = ov
            conf[sl] = "Manual"
            continue
        if g.auto_label.startswith("Rotating"):
            continue
        if np.isfinite(g.ux):
            state[sl] = STATE_MOVING
            ux[sl], uy[sl] = g.ux, g.uy
            speed[sl] = g.speed
            label[sl] = g.auto_label
            conf[sl] = g.confidence
        else:
            state[sl] = STATE_UNKNOWN
    for r in rotations:
        if r.kind == "Turntable":
            state[r.i0 : r.i1 + 1] = STATE_ROTATING
    return {"state": state, "ux": ux, "uy": uy, "speed": speed, "label": label, "confidence": conf}


def motion_phase(i: int, travel: dict, fhx: np.ndarray, fhy: np.ndarray, min_g: float = 0.02) -> str:
    """Describe what the carrier is doing at sample i and how the load acts relative to the travel direction."""
    st = int(travel["state"][i])
    ax, ay = float(fhx[i]), float(fhy[i])  # carrier acceleration at the CoM [g]
    mag = float(np.hypot(ax, ay))
    if st == STATE_ROTATING:
        return "Rotating on turntable"
    if st == STATE_STOPPED:
        return "Stopped · impact / push" if mag >= min_g else "Stopped"
    if st == STATE_UNKNOWN:
        return "Moving (direction uncertain)"
    ux, uy = float(travel["ux"][i]), float(travel["uy"][i])
    lab = str(travel["label"][i]).split(" (")[0]
    if travel["confidence"][i] == "Low":
        lab += "?"
    along = ax * ux + ay * uy
    lateral = abs(-ax * uy + ay * ux)
    if mag < min_g:
        return f"Moving {lab} · steady"
    if abs(along) >= lateral:
        return f"Moving {lab} · {'accelerating' if along > 0 else 'braking / stop'}"
    return f"Moving {lab} · lateral load"


def segments_table(segs: list[MotionSegment], overrides: dict | None = None) -> pd.DataFrame:
    overrides = overrides or {}
    return pd.DataFrame(
        [
            {
                "Start [s]": round(g.t0, 1),
                "End [s]": round(g.t1, 1),
                "Duration [s]": round(g.t1 - g.t0, 1),
                "Auto direction": g.auto_label,
                "Confidence": g.confidence,
                "Speed est. [m/s]": round(g.speed, 2) if np.isfinite(g.speed) else np.nan,
                "Direction": overrides.get(segment_key(g), "Auto"),
            }
            for g in segs
        ]
    )


# --------------------------------------------------------------------------- #
# Speed profile along the route
# --------------------------------------------------------------------------- #
def speed_profile(
    kin: Kinematics,
    res: SimulationResult,
    segs: list[MotionSegment],
    overrides: dict | None = None,
    max_reliable_s: float = 60.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Estimated travel speed for the whole record.

    For each motion segment the acceleration along the travel direction is
    integrated from rest to rest. A constant offset is removed so that the
    speed returns to zero at the stop (zero-velocity update); this compensates
    the small DC offsets that MEMS accelerometers show under vibration and
    slight conveyor slopes. The estimate is reliable for segments of up to
    about a minute; longer segments are flagged.

    Returns (speed [m/s], longitudinal acceleration [g], quality code per sample, segment table).
    Quality code: 0 = stopped, 1 = good, 2 = fair / poor (low reliability), 3 = unknown direction.
    """
    fs = kin.fs
    n = len(kin.t)
    v = np.zeros(n)
    a_long = np.full(n, np.nan)
    qual = np.zeros(n, dtype=int)
    overrides = overrides or {}
    pre = int(round(0.5 * fs))
    rows = []
    for k, g in enumerate(segs, start=1):
        ov = overrides.get(segment_key(g), "Auto")
        if ov == "Stopped" or g.auto_label.startswith("Rotating"):
            continue
        if ov in DIRECTIONS:
            ux, uy = DIRECTIONS[ov]
            dlabel, conf = ov, "Manual"
        elif np.isfinite(g.ux):
            ux, uy = g.ux, g.uy
            dlabel, conf = g.auto_label, g.confidence
        else:
            v[g.i0 : g.i1 + 1] = np.nan
            qual[g.i0 : g.i1 + 1] = 3
            rows.append(
                {"Segment": k, "Start [s]": round(g.t0, 1), "End [s]": round(g.t1, 1), "Quality": "Unknown direction"}
            )
            continue
        a = max(0, g.i0 - pre)
        b = min(n - 1, g.i1 + pre)
        al = G0 * (res.fhx_com[a : b + 1] * ux + res.fhy_com[a : b + 1] * uy)
        al = al - al.mean()  # zero-velocity update: speed back to 0 at the stop
        vv = np.cumsum(al) / fs
        vmax, vmin = float(vv.max()), float(vv.min())
        dur = g.t1 - g.t0
        if vmax < 0.05:
            quality = "No significant travel"
            vv[:] = 0.0
        elif vmin < -0.15 * vmax:
            quality = "Poor"
        elif dur > max_reliable_s:
            quality = "Fair"
        else:
            quality = "Good"
        v[a : b + 1] = vv
        a_long[a : b + 1] = al / G0
        qual[a : b + 1] = 1 if quality == "Good" else (0 if quality == "No significant travel" else 2)
        if quality == "No significant travel":
            rows.append(
                {
                    "Segment": k,
                    "Start [s]": round(g.t0, 1),
                    "End [s]": round(g.t1, 1),
                    "Duration [s]": round(dur, 1),
                    "Direction": dlabel,
                    "Quality": quality,
                }
            )
            continue
        vpos = np.clip(vv, 0.0, None)
        cruise = float(np.median(vv[vv > 0.5 * vmax]))
        up = np.where(vv >= 0.9 * cruise)[0]
        i_up = int(up[0]) if len(up) else 0
        i_dn = int(up[-1]) if len(up) else len(vv) - 1
        acc_part = al[: max(i_up, 1) + 1]
        dec_part = al[i_dn:]
        mu_seg = res.mu_req[a : b + 1]
        mu_seg = mu_seg[np.isfinite(mu_seg)]
        rows.append(
            {
                "Segment": k,
                "Start clock": pd.Timestamp(kin.clock[g.i0]).strftime("%H:%M:%S.%f")[:-3],
                "Start [s]": round(g.t0, 1),
                "End [s]": round(g.t1, 1),
                "Duration [s]": round(dur, 1),
                "Direction": dlabel,
                "Direction confidence": conf,
                "Quality": quality,
                "Distance [m]": round(float(np.sum(vpos) / fs), 2),
                "Cruise speed [m/s]": round(cruise, 2),
                "Peak speed [m/s]": round(vmax, 2),
                "Acceleration ramp [s]": round(i_up / fs, 1),
                "Deceleration ramp [s]": round((len(vv) - 1 - i_dn) / fs, 1),
                "Peak acceleration [g]": round(float(acc_part.max()) / G0, 3),
                "Peak deceleration [g]": round(float(-dec_part.min()) / G0, 3),
                "Max required mu": round(float(mu_seg.max()), 3) if len(mu_seg) else np.nan,
            }
        )
    cols = [
        "Segment",
        "Start clock",
        "Start [s]",
        "End [s]",
        "Duration [s]",
        "Direction",
        "Direction confidence",
        "Quality",
        "Distance [m]",
        "Cruise speed [m/s]",
        "Peak speed [m/s]",
        "Acceleration ramp [s]",
        "Deceleration ramp [s]",
        "Peak acceleration [g]",
        "Peak deceleration [g]",
        "Max required mu",
    ]
    return v, a_long, qual, pd.DataFrame(rows, columns=cols)
