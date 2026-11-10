"""
impact.py
=========

Frozen-frame analysis of one moment of a recording (a slip event or a critical moment): what loaded the
carrier, whether it looks like a hit, where the hit acted and how the cluster is predicted to move.

What an IMU on the carrier can tell about a hit
-----------------------------------------------
* Direction and size: the horizontal acceleration of the carrier at the peak is the net horizontal force on it
  divided by its mass. A stopper that brakes a carrier travelling forwards pushes it rearwards.
* Hit or ramp: a hit is a short shock (a few tens of ms); a start or braking ramp of the conveyor lasts several
  tenths of a second. The duration is measured as the width of the pulse at half its peak.
* Line of action: a push that does not pass through the centre of mass also twists the carrier. Over the pulse
  the push changes the speed by Δv and the yaw rate by Δω; with m·Δv = ∫F dt and I·Δω = ∫M dt the push acts along
  the line at the distance h = k²·Δω / |Δv| from the centre (k = radius of gyration of the loaded carrier). Δω is
  read from the gyroscope before and after the pulse, so nothing is differentiated at the shock itself.
* Hit point: a push enters the body where its line of action meets the outline on the side the force comes from.
  The sensor cannot tell where along that line the contact is (a stopper below the carrier acts somewhere on it).

A carrier held by side guides cannot yaw freely, so an off-centre hit twists it less than a free body: the side
is reliable, the position along the side is an indication that tends towards the middle.

Coordinate frame: carrier frame, X = Front, Y = Left, origin at the nominal cluster centre.
"""

from __future__ import annotations

import os as _os

_LOADED_MTIME = _os.path.getmtime(__file__)  # used by app.py to detect stale modules

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data_io import G0, Kinematics
from slip_model import CORNER_NAMES, CORNER_SIGNS, ClusterParams, SimulationResult, is_turning, vertical_factor

GOT_MARGIN = 0.08  # schematic GOT outline: margin around the cluster, fraction of its longer side

KIND_IMPACT = "impact"  # short shock
KIND_SUSTAINED = "sustained"  # start / braking ramp
KIND_TURNTABLE = "turntable"  # load while the carrier rotates on a turntable
KIND_WEAK = "weak"  # hardly any horizontal acceleration (the load is a twist)

_DIR8 = ["Front", "Front-Left", "Left", "Rear-Left", "Rear", "Rear-Right", "Right", "Front-Right"]


@dataclass(frozen=True)
class SnapshotSettings:
    half_level: float = 0.5  # the pulse lasts while the acceleration stays above this share of its peak
    max_half_window_s: float = 0.5  # the pulse is followed at most this far to each side of the peak
    shock_max_s: float = 0.12  # a pulse up to this long is a hit, a longer one a ramp
    min_acc_g: float = 0.02  # below this peak the horizontal acceleration is noise
    search_pad_s: float = 0.1  # the peak is searched this far before and after the event
    corner_tol: float = 0.25  # a point closer to a corner than this share of the shorter side is "the corner"
    centre_tol: float = 0.08  # a point closer to the middle of a side than this share of the side is "centre"


@dataclass
class EventSnapshot:
    # samples
    i_start: int  # first / last sample of the event
    i_end: int
    i_peak: int  # peak horizontal acceleration of the carrier
    i_mu: int  # peak friction demand
    w0: int  # pulse (half-peak) window
    w1: int
    t_peak: float
    clock_peak: pd.Timestamp
    # carrier load
    kind: str
    acc_peak: np.ndarray  # carrier acceleration at the GOT centre at the peak [g] (x, y)
    acc_peak_g: float
    force_dir: np.ndarray  # unit direction of the net horizontal force on the carrier
    pulse_s: float
    shock_limit_s: float
    dv: np.ndarray  # speed change of the carrier over the pulse [m/s]
    dw: float  # yaw-rate change over the pulse [rad/s]
    alpha_mean: float  # yaw acceleration over the pulse [rad/s2]
    fz_min: float  # lowest vertical acceleration around the event [g]
    turning: bool
    # line of action and hit point
    gyration: float  # radius of gyration used [m]
    arm: float  # signed distance of the line of action from the GOT centre [m]
    line_point: np.ndarray  # point of the line of action closest to the centre
    hit_xy: np.ndarray  # where the push enters the GOT outline
    exit_xy: np.ndarray | None  # where its line of action leaves the outline
    hit_outside: bool  # the line of action misses the outline (mostly a twist)
    hit_label: str
    outline: tuple[float, float]  # GOT outline used (length, width) [m]
    # friction at the peak demand
    load_g: np.ndarray  # inertial load on the cluster [g] (x, y) = -acceleration at its centre of mass
    limit_g: float  # friction limit at that instant [g]
    mu_peak: float
    mu_s: float
    # predicted cluster movement during the event
    slipped: bool
    du: np.ndarray  # centre-of-mass shift [m]
    dphi: float  # rotation [rad]
    dcorner: np.ndarray  # (4, 2) corner shifts [m], CORNER_NAMES order
    shift_before_mm: float  # largest corner shift the cluster already had before the event
    pivot: np.ndarray | None = None  # (x, y) relative to the geometric centre [m]
    pivot_label: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def corner_shift_mm(self) -> np.ndarray:
        return np.hypot(self.dcorner[:, 0], self.dcorner[:, 1]) * 1e3

    @property
    def is_hit(self) -> bool:
        return self.kind == KIND_IMPACT


def got_outline(cluster: ClusterParams) -> tuple[float, float]:
    """Length and width of the schematic GOT outline drawn in the top view [m]."""
    m = GOT_MARGIN * max(cluster.length, cluster.width)
    return cluster.length + 2 * m, cluster.width + 2 * m


def default_gyration(cluster: ClusterParams) -> float:
    """Radius of gyration about the vertical axis of a uniform plate with the GOT outline [m]."""
    lg, wg = got_outline(cluster)
    return float(np.sqrt((lg**2 + wg**2) / 12.0))


def carrier_acceleration(kin: Kinematics, sensor, point=(0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal acceleration [g] of a point of the carrier (rigid body), from the sensor signals."""
    rx, ry = point[0] - sensor.x, point[1] - sensor.y
    ax = kin.fx + (kin.alpha * (-ry) - kin.wz**2 * rx) / G0
    ay = kin.fy + (kin.alpha * rx - kin.wz**2 * ry) / G0
    return ax, ay


def direction_name(vx: float, vy: float) -> str:
    """Name of a direction in the carrier frame: Front, Front-Left, Left, ..."""
    if not (np.isfinite(vx) and np.isfinite(vy)) or np.hypot(vx, vy) < 1e-12:
        return "-"
    return _DIR8[int(np.round(np.arctan2(vy, vx) / (np.pi / 4))) % 8]


def _line_box(q0: np.ndarray, d: np.ndarray, hx: float, hy: float) -> tuple[float, float] | None:
    """Parameters (s_in, s_out) for which q0 + s·d lies inside the box |x| <= hx, |y| <= hy; None if it misses."""
    lo, hi = -np.inf, np.inf
    for q, dd, h in ((q0[0], d[0], hx), (q0[1], d[1], hy)):
        if abs(dd) < 1e-12:
            if abs(q) > h:
                return None
            continue
        s1, s2 = (-h - q) / dd, (h - q) / dd
        lo, hi = max(lo, min(s1, s2)), min(hi, max(s1, s2))
    return (lo, hi) if lo <= hi else None


def locate_on_outline(x: float, y: float, lg: float, wg: float, s: SnapshotSettings = SnapshotSettings()) -> str:
    """Corner or side of the outline (length lg, width wg) the point lies on."""
    corners = CORNER_SIGNS * np.array([lg / 2, wg / 2])
    dist = np.hypot(corners[:, 0] - x, corners[:, 1] - y)
    i = int(np.argmin(dist))
    if dist[i] <= s.corner_tol * min(lg, wg):
        return f"{CORNER_NAMES[i]} corner"
    if lg / 2 - abs(x) <= wg / 2 - abs(y):  # on the front or rear face
        side, off, span = ("Front" if x > 0 else "Rear"), y, wg
        toward = "Left" if y > 0 else "Right"
    else:
        side, off, span = ("Left" if y > 0 else "Right"), x, lg
        toward = "Front" if x > 0 else "Rear"
    if abs(off) < s.centre_tol * span:
        return f"{side} side, centre"
    return f"{side} side, {abs(off):.2f} m towards {toward}"


def analyse_event(
    kin: Kinematics,
    res: SimulationResult,
    t_start: float,
    t_end: float,
    gyration: float | None = None,
    pivot: tuple[float, float] | None = None,
    pivot_label: str = "",
    s: SnapshotSettings = SnapshotSettings(),
) -> EventSnapshot:
    """Analyse the load on the carrier and the cluster movement between t_start and t_end [s]."""
    n = len(kin.t)
    fs = float(kin.fs)
    ia = int(np.clip(np.searchsorted(kin.t, t_start - 1e-9), 0, n - 1))
    ib = int(np.clip(np.searchsorted(kin.t, t_end + 1e-9, side="right") - 1, ia, n - 1))
    npad = max(1, int(round(s.search_pad_s * fs)))
    lo, hi = max(ia - npad, 0), min(ib + npad, n - 1)

    # ---- carrier load: peak, pulse, kind
    ax, ay = carrier_acceleration(kin, res.sensor)
    pk = lo + int(np.argmax(np.hypot(ax[lo : hi + 1], ay[lo : hi + 1])))
    acc_peak = np.array([ax[pk], ay[pk]], dtype=float)
    peak = float(np.hypot(*acc_peak))
    d_pk = acc_peak / peak if peak > 1e-9 else np.array([1.0, 0.0])
    along = ax * d_pk[0] + ay * d_pk[1]
    nmax = max(1, int(round(s.max_half_window_s * fs)))
    w0 = pk
    while w0 > max(pk - nmax, 0) and along[w0 - 1] >= s.half_level * peak:
        w0 -= 1
    w1 = pk
    while w1 < min(pk + nmax, n - 1) and along[w1 + 1] >= s.half_level * peak:
        w1 += 1
    pulse_s = (w1 - w0 + 1) / fs
    # impulse of the pulse: speed change and yaw-rate change of the carrier over the same interval
    j0, j1 = max(w0 - 1, 0), min(w1 + 1, n - 1)
    dt_w = float(kin.t[j1] - kin.t[j0])

    def _impulse(a: np.ndarray) -> float:
        seg = a[j0 : j1 + 1]
        return float(np.sum(0.5 * (seg[1:] + seg[:-1])) / fs) if len(seg) > 1 else float(seg[0] / fs)

    dv = G0 * np.array([_impulse(ax), _impulse(ay)])
    dv_mag = float(np.hypot(*dv))
    force_dir = dv / dv_mag if dv_mag > 1e-9 else d_pk
    dw = float(kin.wz[j1] - kin.wz[j0])
    alpha_mean = dw / dt_w if dt_w > 0 else float(kin.alpha[pk])
    turning = bool(is_turning(kin, w0, w1))
    shock_limit = max(s.shock_max_s, 1.5 / max(float(kin.fs_native), 1e-6))
    if peak < s.min_acc_g:
        kind = KIND_WEAK
    elif pulse_s <= shock_limit:
        kind = KIND_IMPACT
    elif turning:
        kind = KIND_TURNTABLE
    else:
        kind = KIND_SUSTAINED

    # ---- line of action of the net horizontal force and the point where it enters the GOT outline
    lg, wg = got_outline(res.cluster)
    k_gyr = float(gyration) if gyration and gyration > 0 else default_gyration(res.cluster)
    arm = k_gyr**2 * dw / max(dv_mag, 1e-6)
    q0 = arm * np.array([force_dir[1], -force_dir[0]])
    hit = _line_box(q0, force_dir, lg / 2, wg / 2)
    if hit is not None:
        hit_xy, exit_xy, outside = q0 + hit[0] * force_dir, q0 + hit[1] * force_dir, False
        hit_label = locate_on_outline(hit_xy[0], hit_xy[1], lg, wg, s)
    else:
        corners = CORNER_SIGNS * np.array([lg / 2, wg / 2])
        gap = np.abs(corners[:, 0] * force_dir[1] - corners[:, 1] * force_dir[0] - arm)
        i = int(np.argmin(gap))
        hit_xy, exit_xy, outside = corners[i], None, True
        hit_label = f"{CORNER_NAMES[i]} corner"

    # ---- friction at the peak demand
    mu_w = np.where(np.isfinite(res.mu_req[lo : hi + 1]), res.mu_req[lo : hi + 1], np.inf)
    i_mu = lo + int(np.argmax(mu_w))
    fzv = vertical_factor(res, kin)

    # ---- predicted movement of the cluster during the event
    a_m, b_m = max(ia - 1, 0), min(ib + 1, n - 1)
    du = np.array([res.ux[b_m] - res.ux[a_m], res.uy[b_m] - res.uy[a_m]], dtype=float)
    dphi = float(res.phi[b_m] - res.phi[a_m])
    dcorner = np.asarray(res.corner_xy[b_m] - res.corner_xy[a_m], dtype=float)
    slipped = bool(res.sliding[a_m : b_m + 1].any())

    notes: list[str] = []
    if kin.fs_native < 50:
        notes.append(
            f"Recorded at {kin.fs_native:.0f} Hz: a shock shorter than {1e3 / kin.fs_native:.0f} ms is one sample, "
            "so its peak and duration are uncertain."
        )
    if outside:
        notes.append(
            "The line of action passes outside the GOT: the carrier was mostly twisted (yaw jolt), "
            "there is no single hit point."
        )
    pv = None
    if pivot is not None and np.all(np.isfinite(pivot)):
        pv = np.array(pivot, dtype=float)

    return EventSnapshot(
        i_start=ia,
        i_end=ib,
        i_peak=pk,
        i_mu=i_mu,
        w0=w0,
        w1=w1,
        t_peak=float(kin.t[pk]),
        clock_peak=pd.Timestamp(kin.clock[pk]),
        kind=kind,
        acc_peak=acc_peak,
        acc_peak_g=peak,
        force_dir=force_dir,
        pulse_s=pulse_s,
        shock_limit_s=shock_limit,
        dv=dv,
        dw=dw,
        alpha_mean=alpha_mean,
        fz_min=float(np.min(kin.fz[lo : hi + 1])),
        turning=turning,
        gyration=k_gyr,
        arm=float(arm),
        line_point=q0,
        hit_xy=np.asarray(hit_xy, dtype=float),
        exit_xy=None if exit_xy is None else np.asarray(exit_xy, dtype=float),
        hit_outside=outside,
        hit_label=hit_label,
        outline=(lg, wg),
        load_g=-np.array([res.fhx_com[i_mu], res.fhy_com[i_mu]], dtype=float),
        limit_g=float(res.contact.mu_s * fzv[i_mu]),
        mu_peak=float(res.mu_req[i_mu]),
        mu_s=float(res.contact.mu_s),
        slipped=slipped,
        du=du,
        dphi=dphi,
        dcorner=dcorner,
        shift_before_mm=float(res.corner_disp[a_m].max() * 1e3),
        pivot=pv,
        pivot_label=pivot_label,
        notes=notes,
    )


def verdict(snap: EventSnapshot) -> str:
    """One line: was it a hit?"""
    ms = snap.pulse_s * 1e3
    if snap.kind == KIND_IMPACT:
        return f"Hit: yes · short shock of {ms:.0f} ms, {snap.acc_peak_g:.2f} g"
    if snap.kind == KIND_TURNTABLE:
        return f"Hit: no · load while rotating on a turntable, {snap.acc_peak_g:.2f} g"
    if snap.kind == KIND_SUSTAINED and snap.hit_outside:
        return f"Hit: no · the carrier is twisted over {ms:.0f} ms (yaw acceleration)"
    if snap.kind == KIND_SUSTAINED:
        return f"Hit: no · start / braking ramp of {ms:.0f} ms, {snap.acc_peak_g:.2f} g"
    return f"Hit: no · hardly any horizontal acceleration ({snap.acc_peak_g:.3f} g)"


def where_line(snap: EventSnapshot) -> str:
    """One line: where the push acted."""
    if snap.kind == KIND_WEAK:
        return "No push to locate"
    if snap.hit_outside:
        return f"Mostly a twist of the carrier, nearest the {snap.hit_label}"
    return ("Estimated hit on: " if snap.is_hit else "Net force passes through: ") + snap.hit_label


def summary_lines(snap: EventSnapshot, context: str = "") -> list[str]:
    """Text of the snapshot panel (simple HTML: <b> only), one entry per line."""
    yaw = np.rad2deg(snap.dw)
    turn = "counter-clockwise" if yaw > 0 else "clockwise"
    mu_txt = f"{snap.mu_peak:.3f}" if np.isfinite(snap.mu_peak) else "∞ (lift-off)"
    out = [
        "<b>Load on the carrier</b>",
        verdict(snap),
        where_line(snap),
    ]
    if snap.kind != KIND_WEAK:
        out.append(f"Carrier pushed towards {direction_name(*snap.force_dir)}")
    out += [
        f"Peak acceleration X {snap.acc_peak[0]:+.2f} g, Y {snap.acc_peak[1]:+.2f} g",
        f"Speed change {np.hypot(*snap.dv):.2f} m/s over the pulse",
        "Yaw jolt " + (f"{yaw:+.2f} °/s ({turn})" if abs(yaw) >= 0.05 else "none"),
        f"Vertical acceleration down to {snap.fz_min:.2f} g",
    ]
    if context:
        out.append(context)
    out += [
        "",
        "<b>Friction</b>",
        f"Required μ {mu_txt}  vs  μs {snap.mu_s:.3f}",
        f"Inertial load {np.hypot(*snap.load_g):.2f} g  vs  limit {snap.limit_g:.2f} g",
        "",
        "<b>Predicted cluster movement</b>",
    ]
    if snap.slipped and snap.corner_shift_mm.max() >= 0.005:
        i = int(np.argmax(snap.corner_shift_mm))
        rot = np.rad2deg(snap.dphi)
        out += [
            f"Centre {np.hypot(*snap.du) * 1e3:.2f} mm towards {direction_name(*snap.du)}",
            f"dX {snap.du[0] * 1e3:+.2f} mm, dY {snap.du[1] * 1e3:+.2f} mm",
            (
                f"Rotation {rot:+.3f}° ({'counter-clockwise' if rot > 0 else 'clockwise'})"
                if abs(rot) >= 0.0005
                else "Rotation: none"
            ),
            f"Largest corner shift: {CORNER_NAMES[i]} {snap.corner_shift_mm[i]:.2f} mm",
        ]
        if snap.pivot is not None and snap.pivot_label:
            out.append(f"Pivot: {snap.pivot_label}")
        elif abs(rot) < 0.0005:
            out.append("Pivot: none, the cluster slides without turning")
    else:
        out.append("No slip: the cluster holds")
    if snap.shift_before_mm >= 0.01:
        out.append(f"Shift already there before: {snap.shift_before_mm:.2f} mm")
    return out
