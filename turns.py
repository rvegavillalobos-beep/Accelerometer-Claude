"""
turns.py
========

Detection and analysis of carrier rotations (turntables) from the IMU yaw rate.

The gyroscope Z axis measures the yaw rate of the workpiece carrier directly, so
rotations can be detected and their angle obtained by integration with good
accuracy (the WitMotion firmware outputs exactly 0 deg/s at rest, which gives
clean start / end points).

For every rotation the module reports
* start / end time, duration, signed angle and deviation from the nominal angle,
* fast phase and slow (creep / positioning) phase,
* spin-up and braking times and yaw accelerations,
* the friction demand on the cluster before (entry), during and after (exit)
  the rotation, and whether slip is predicted in each phase,
* an estimate of the turntable axis position relative to the sensor
  (least-squares fit of the rigid-body equations to the measured acceleration).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from data_io import G0, Kinematics
from slip_model import SimulationResult


@dataclass(frozen=True)
class TurnSettings:
    rate_threshold_deg_s: float = 0.5  # |yaw rate| above this = rotating (boundaries of a rotation)
    min_turn_deg: float = 45.0  # rotations at least this large are reported as turntable turns
    min_minor_deg: float = 3.0  # smaller rotations are ignored
    merge_gap_s: float = 1.0  # same-direction rotations closer than this are merged (fast + creep phase)
    pad_s: float = 4.0  # entry / exit window analysed around each turn
    nominal_deg: float = 90.0  # nominal turntable angle


@dataclass(frozen=True)
class Rotation:
    kind: str  # "Turntable" | "Minor rotation"
    label: str  # "Turn 1", "Minor 1", ...
    i0: int
    i1: int
    t0: float
    t1: float
    angle_deg: float


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    if len(y) < 2:
        return 0.0
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def detect_rotations(kin: Kinematics, s: TurnSettings = TurnSettings()) -> list[Rotation]:
    """Find carrier rotations in the yaw-rate signal."""
    w = np.rad2deg(kin.wz)
    t = kin.t
    k = max(3, int(round(0.3 * kin.fs)))
    if k % 2 == 0:
        k += 1
    ws = median_filter(w, size=k, mode="nearest")  # suppresses short impact spikes
    active = np.abs(ws) > s.rate_threshold_deg_s
    d = np.diff(np.concatenate([[0], active.astype(int), [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0] - 1

    segs: list[list] = []
    gap = int(round(s.merge_gap_s * kin.fs))
    for a, b in zip(starts, ends):
        ang = _trapz(w[a : b + 1], t[a : b + 1])
        if segs and a - segs[-1][1] <= gap and ang != 0 and np.sign(ang) == np.sign(segs[-1][2]):
            segs[-1][1] = b
            segs[-1][2] = _trapz(w[segs[-1][0] : b + 1], t[segs[-1][0] : b + 1])
        else:
            segs.append([a, b, ang])

    out: list[Rotation] = []
    n_turn = n_minor = 0
    for a, b, ang in segs:
        if abs(ang) < s.min_minor_deg:
            continue
        if abs(ang) >= s.min_turn_deg:
            n_turn += 1
            kind, label = "Turntable", f"Turn {n_turn}"
        else:
            n_minor += 1
            kind, label = "Minor rotation", f"Minor {n_minor}"
        out.append(Rotation(kind, label, int(a), int(b), float(t[a]), float(t[b]), float(ang)))
    return out


def locate(t: float, rotations: list[Rotation], pad_s: float) -> str:
    """Describe where a time instant lies relative to the detected rotations."""
    for r in rotations:
        if r.t0 <= t <= r.t1:
            return f"{r.label} · rotating"
    for r in rotations:
        if r.kind != "Turntable":
            continue
        if r.t0 - pad_s <= t < r.t0:
            return f"{r.label} · entry ({t - r.t0:+.1f} s)"
        if r.t1 < t <= r.t1 + pad_s:
            return f"{r.label} · exit ({t - r.t1:+.1f} s)"
    return "Straight / stationary"


def estimate_axis(kin: Kinematics, idx: np.ndarray) -> dict:
    """Least-squares estimate of the sensor position relative to the rotation axis.

    Rigid body rotating about a fixed axis O:   a_S = alpha z x r - omega^2 r,   r = S - O.
    Returns r (carrier frame) with standard errors and the coefficient of determination.
    """
    w = kin.wz[idx]
    al = kin.alpha[idx]
    ax = kin.fx[idx] * G0
    ay = kin.fy[idx] * G0
    A = np.vstack([np.column_stack([-(w**2), -al]), np.column_stack([al, -(w**2)])])
    y = np.concatenate([ax, ay])
    n = len(y)
    if n < 10 or np.linalg.matrix_rank(A) < 2:
        return {"rx": np.nan, "ry": np.nan, "se": np.nan, "r2": np.nan, "n": n}
    r, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ r
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / sst if sst > 0 else np.nan
    sigma2 = float(np.sum(resid**2)) / max(n - 2, 1)
    try:
        cov = sigma2 * np.linalg.inv(A.T @ A)
        se = float(np.sqrt(max(cov[0, 0], 0) + max(cov[1, 1], 0)))
    except np.linalg.LinAlgError:
        se = np.nan
    return {"rx": float(r[0]), "ry": float(r[1]), "se": se, "r2": r2, "n": n}


def _moving_idx(kin: Kinematics, r: Rotation, min_rate_deg_s: float = 0.5) -> np.ndarray:
    idx = np.arange(r.i0, r.i1 + 1)
    return idx[np.abs(np.rad2deg(kin.wz[idx])) > min_rate_deg_s]


def estimate_axis_all(kin: Kinematics, rotations: list[Rotation]) -> dict:
    """Combined axis estimate over all turntable turns."""
    idx = [_moving_idx(kin, r) for r in rotations if r.kind == "Turntable"]
    if not idx:
        return {"rx": np.nan, "ry": np.nan, "se": np.nan, "r2": np.nan, "n": 0}
    return estimate_axis(kin, np.concatenate(idx))


def _win(t: np.ndarray, a: float, b: float, left_closed=True, right_closed=True) -> np.ndarray:
    m = (t >= a) if left_closed else (t > a)
    m &= (t <= b) if right_closed else (t < b)
    return m


def _nanmax(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(x.max()) if len(x) else np.nan


def analyse_rotations(
    kin: Kinematics, res: SimulationResult, rotations: list[Rotation], s: TurnSettings = TurnSettings()
) -> pd.DataFrame:
    """One row per detected rotation with kinematics, friction demand and slip per phase."""
    t = kin.t
    rows = []
    for r in rotations:
        sl = slice(r.i0, r.i1 + 1)
        tt = t[sl]
        w = np.rad2deg(kin.wz[sl])
        aw = np.abs(w)
        peak = float(aw.max())
        sign = 1.0 if r.angle_deg >= 0 else -1.0
        fast = aw >= 0.3 * peak
        fast_angle = _trapz(np.where(fast, w, 0.0), tt)
        slow_angle = r.angle_deg - fast_angle
        hi = np.where(aw >= 0.9 * peak)[0]
        i_up, i_dn = int(hi[0]), int(hi[-1])
        after = np.where(aw[i_dn:] < 0.3 * peak)[0]
        i_brk_end = i_dn + int(after[0]) if len(after) else len(aw) - 1
        al = np.rad2deg(kin.alpha[sl]) * sign  # + = speeding up in the rotation direction
        spin_acc = float(np.max(al[: i_up + 1])) if i_up >= 0 else np.nan
        brake_dec = float(-np.min(al[i_dn : i_brk_end + 1])) if i_brk_end >= i_dn else np.nan

        m_entry = _win(t, r.t0 - s.pad_s, r.t0, True, False)
        m_rot = _win(t, r.t0, r.t1)
        m_exit = _win(t, r.t1, r.t1 + s.pad_s, False, True)
        m_all = _win(t, r.t0 - s.pad_s, r.t1 + s.pad_s)
        mu_e, mu_r, mu_x = _nanmax(res.mu_req[m_entry]), _nanmax(res.mu_req[m_rot]), _nanmax(res.mu_req[m_exit])
        phases = {"entry": mu_e, "rotation": mu_r, "exit": mu_x}
        worst = max((k for k in phases if np.isfinite(phases[k])), key=lambda k: phases[k], default="-")
        slip_ph = [k for k, m in (("entry", m_entry), ("rotation", m_rot), ("exit", m_exit)) if res.sliding[m].any()]
        ia = np.where(m_all)[0]
        dphi = np.rad2deg(res.phi[ia[-1]] - res.phi[ia[0]]) if len(ia) else np.nan
        dcorner = (
            float(
                np.max(
                    np.hypot(
                        res.corner_xy[ia[-1], :, 0] - res.corner_xy[ia[0], :, 0],
                        res.corner_xy[ia[-1], :, 1] - res.corner_xy[ia[0], :, 1],
                    )
                )
                * 1e3
            )
            if len(ia)
            else np.nan
        )
        ax_est = (
            estimate_axis(kin, _moving_idx(kin, r))
            if r.kind == "Turntable"
            else {"rx": np.nan, "ry": np.nan, "r2": np.nan}
        )
        rows.append(
            {
                "ID": r.label,
                "Type": r.kind,
                "Start clock": pd.Timestamp(kin.clock[r.i0]).strftime("%H:%M:%S.%f")[:-3],
                "Start [s]": round(r.t0, 2),
                "End [s]": round(r.t1, 2),
                "Duration [s]": round(r.t1 - r.t0, 2),
                "Angle [deg]": round(r.angle_deg, 1),
                "Direction": "CCW (left)" if r.angle_deg > 0 else "CW (right)",
                "Deviation from nominal [deg]": round(abs(r.angle_deg) - s.nominal_deg, 1)
                if r.kind == "Turntable"
                else np.nan,
                "Fast phase [deg]": round(fast_angle, 1),
                "Slow / creep phase [deg]": round(slow_angle, 1),
                "Peak rate [deg/s]": round(peak, 1),
                "Spin-up time [s]": round(float(tt[i_up] - tt[0]), 2),
                "Braking time [s]": round(float(tt[i_brk_end] - tt[i_dn]), 2),
                "Peak spin-up accel. [deg/s2]": round(spin_acc, 1),
                "Peak braking decel. [deg/s2]": round(brake_dec, 1),
                "Required mu · entry": round(mu_e, 3) if np.isfinite(mu_e) else np.nan,
                "Required mu · rotation": round(mu_r, 3) if np.isfinite(mu_r) else np.nan,
                "Required mu · exit": round(mu_x, 3) if np.isfinite(mu_x) else np.nan,
                "Most demanding phase": worst,
                "Rotational share (max)": round(_nanmax(res.mu_rot[m_rot]), 3),
                "Slip predicted": ", ".join(slip_ph) if slip_ph else "No",
                "Cluster rotation [deg]": round(float(dphi), 4) if np.isfinite(dphi) else np.nan,
                "Max corner shift [mm]": round(dcorner, 2) if np.isfinite(dcorner) else np.nan,
                "Sensor rel. to axis X [m]": round(ax_est["rx"], 2) if np.isfinite(ax_est["rx"]) else np.nan,
                "Sensor rel. to axis Y [m]": round(ax_est["ry"], 2) if np.isfinite(ax_est["ry"]) else np.nan,
                "Axis fit R²": round(ax_est["r2"], 2) if np.isfinite(ax_est["r2"]) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def turn_insight(table: pd.DataFrame, mu_s: float) -> list[str]:
    """Short, data-driven statements about the turntables."""
    tt = table[table["Type"] == "Turntable"] if len(table) else table
    if tt.empty:
        return ["No turntable rotation detected with the current settings."]
    msgs = []
    angles = ", ".join(f"{a:+.1f}°" for a in tt["Angle [deg]"])
    msgs.append(f"{len(tt)} turntable rotation(s) detected: {angles}.")
    counts = tt["Most demanding phase"].value_counts()
    if len(counts):
        top = counts.index[0]
        msgs.append(
            f"The friction demand around the turntables is highest at the **{top}** phase in {counts.iloc[0]} of {len(tt)} turn(s)."
        )
    mu_rot = tt["Required mu · rotation"].max()
    if np.isfinite(mu_rot):
        msgs.append(
            f"During the rotation itself the required μ is at most {mu_rot:.3f} ({mu_rot / mu_s:.0%} of μs = {mu_s:.3f})."
        )
    slow = tt["Slow / creep phase [deg]"].abs()
    if (slow > 2).any():
        msgs.append(
            f"{int((slow > 2).sum())} turn(s) end with a slow positioning phase (up to {slow.max():.1f}° below 30 % of the peak rate)."
        )
    dev = tt["Deviation from nominal [deg]"].abs()
    if (dev > 3).any():
        msgs.append(
            f"{int((dev > 3).sum())} turn(s) deviate more than 3° from the nominal angle (max {dev.max():.1f}°)."
        )
    slipped = tt[tt["Slip predicted"] != "No"]
    if len(slipped):
        msgs.append(f"Slip is predicted around {', '.join(slipped['ID'])}.")
    return msgs
