"""
visuals.py
==========

Plotly figures for the GOT cluster slip simulator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.ndimage import minimum_filter1d
from plotly.subplots import make_subplots

from data_io import Kinematics
from motion import STATE_MOVING, STATE_ROTATING, STATE_UNKNOWN
from slip_model import CORNER_NAMES, SimulationResult

# Categorical palette (fixed order) and reserved status / ink colours
C1 = "#2a78d6"  # blue
C2 = "#eb6834"  # orange
C3 = "#1baf7a"  # aqua
C4 = "#eda100"  # yellow
CRITICAL = "#d03b3b"
MUTED = "#898781"
INK_2 = "#52514e"
CORNER_COLORS = [C1, C2, C3, C4]  # Front-Left, Front-Right, Rear-Right, Rear-Left
SERIES = [C1, C2, C3, C4, "#e87ba4", "#008300", "#4a3aa7", "#e34948"]  # categorical order, never cycled
CORNER_SHORT = ["FL", "FR", "RR", "RL"]

PLOT_CONFIG = {"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2}}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _envelope(x: np.ndarray, y: np.ndarray, max_points: int = 3000) -> tuple[np.ndarray, np.ndarray]:
    """Min/max decimation that keeps peaks visible."""
    n = len(x)
    if n <= max_points:
        return x, y
    buckets = max_points // 2
    edges = np.linspace(0, n, buckets + 1).astype(int)
    xs, ys = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        seg = y[a:b]
        if np.all(~np.isfinite(seg)):
            xs.append(x[a])
            ys.append(np.nan)
            continue
        i0 = a + int(np.nanargmin(seg))
        i1 = a + int(np.nanargmax(seg))
        for i in sorted({i0, i1}):
            xs.append(x[i])
            ys.append(y[i])
    return np.asarray(xs), np.asarray(ys)


def _line(x, y, name, color, width=1.5, showlegend=True, max_points=3000, hover_fmt=".3f"):
    # SVG traces on purpose: browsers allow only 8-16 WebGL contexts per page, and Streamlit renders every tab at
    # once, so WebGL (Scattergl) charts beyond that limit go blank. The min/max decimation keeps SVG fast.
    xd, yd = _envelope(np.asarray(x), np.asarray(y, dtype=float), max_points)
    return go.Scatter(
        x=xd,
        y=yd,
        mode="lines",
        name=name,
        line=dict(color=color, width=width),
        showlegend=showlegend,
        hovertemplate=f"{name}: %{{y:{hover_fmt}}}<extra></extra>",
    )


def slip_intervals(res: SimulationResult, max_n: int = 150) -> list[tuple[float, float]]:
    s = res.sliding
    if not s.any():
        return []
    d = np.diff(np.concatenate([[0], s.astype(int), [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0] - 1
    out = [(float(res.t[max(a - 1, 0)]), float(res.t[b])) for a, b in zip(starts, ends)]
    return out[:max_n]


def _shade_slip(fig: go.Figure, res: SimulationResult, rows: list[int] | None = None, t0=None, t1=None):
    for a, b in slip_intervals(res):
        if t0 is not None and (b < t0 or a > t1):
            continue
        if b - a < 0.02:
            a, b = a - 0.01, b + 0.01
        if rows is None:
            fig.add_vrect(x0=a, x1=b, fillcolor=CRITICAL, opacity=0.15, line_width=0, layer="below")
        else:
            for r in rows:
                fig.add_vrect(x0=a, x1=b, fillcolor=CRITICAL, opacity=0.15, line_width=0, layer="below", row=r, col=1)


def _base_layout(fig: go.Figure, height: int):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    return fig


def _window(res: SimulationResult, t0: float | None, t1: float | None) -> slice:
    if t0 is None or t1 is None:
        return slice(0, len(res.t))
    i0 = int(np.searchsorted(res.t, t0, side="left"))
    i1 = int(np.searchsorted(res.t, t1, side="right"))
    return slice(max(i0, 0), max(i1, i0 + 2))


# --------------------------------------------------------------------------- #
# Time-series figures
# --------------------------------------------------------------------------- #
def fig_kinematics(kin: Kinematics, res: SimulationResult, t0=None, t1=None) -> go.Figure:
    """Carrier kinematics: horizontal / vertical specific force, yaw rate and yaw acceleration."""
    w = _window(res, t0, t1)
    t = res.t[w]
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "Horizontal specific force at cluster CoM [g]",
            "Vertical specific force [g]  (1 g = static; lower values reduce the friction capacity)",
            "Carrier yaw rate [deg/s]",
            "Carrier yaw acceleration [deg/s²]",
        ),
    )
    fig.add_trace(_line(t, res.fhx_com[w], "X (length)", C1), row=1, col=1)
    fig.add_trace(_line(t, res.fhy_com[w], "Y (width)", C2), row=1, col=1)
    fig.add_trace(_line(t, res.fh_com[w], "Magnitude", C3), row=1, col=1)
    fig.add_trace(
        _line(t, res.contact.mu_s * np.clip(kin.fz[w], 0, None), "Dynamic friction limit μs·f_z", CRITICAL, width=1.2),
        row=1,
        col=1,
    )
    fig.add_trace(_line(t, kin.fz[w], "Vertical", C1, showlegend=False), row=2, col=1)
    fig.add_trace(_line(t, np.rad2deg(kin.wz[w]), "Yaw rate", C1, showlegend=False, hover_fmt=".2f"), row=3, col=1)
    fig.add_trace(_line(t, np.rad2deg(kin.alpha[w]), "Yaw accel.", C1, showlegend=False, hover_fmt=".1f"), row=4, col=1)
    _shade_slip(fig, res, rows=[1, 2, 3, 4], t0=t[0], t1=t[-1])
    fig.update_xaxes(title_text="Time from recording start [s]", row=4, col=1)
    return _base_layout(fig, 820)


def fig_friction(res: SimulationResult, t0=None, t1=None) -> go.Figure:
    """Friction coefficient required to keep the cluster in place vs. available friction."""
    w = _window(res, t0, t1)
    t = res.t[w]
    cap = lambda a: np.clip(a, 0, 5.0)
    fig = go.Figure()
    fig.add_trace(_line(t, cap(res.mu_req[w]), "Required μ (combined)", C1, width=2))
    fig.add_trace(_line(t, cap(res.mu_trans[w]), "Translational part", C2, width=1.2))
    fig.add_trace(_line(t, cap(res.mu_rot[w]), "Rotational part", C3, width=1.2))
    fig.add_hline(
        y=res.contact.mu_s,
        line=dict(color=CRITICAL, width=1.5),
        annotation_text=f"Static friction μs = {res.contact.mu_s:.3f}",
        annotation_position="top left",
        annotation_font_color=CRITICAL,
    )
    if abs(res.contact.mu_k - res.contact.mu_s) > 1e-9:
        fig.add_hline(
            y=res.contact.mu_k,
            line=dict(color=MUTED, width=1),
            annotation_text=f"Kinetic μk = {res.contact.mu_k:.3f}",
            annotation_position="bottom left",
        )
    _shade_slip(fig, res, t0=t[0], t1=t[-1])
    ymax = max(res.contact.mu_s * 1.3, float(np.nanmax(cap(res.mu_req[w]))) * 1.1)
    fig.update_yaxes(title_text="Friction coefficient [-]", range=[0, ymax])
    fig.update_xaxes(title_text="Time from recording start [s]")
    fig.update_layout(
        title=dict(text="Friction demand vs. friction capacity (red bands = predicted slip)", x=0, font=dict(size=14))
    )
    return _base_layout(fig, 380)


def fig_motion(res: SimulationResult, tolerance_mm: float, t0=None, t1=None) -> go.Figure:
    """Cluster displacement relative to the GOT."""
    w = _window(res, t0, t1)
    t = res.t[w]
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Cluster CoM displacement on the GOT [mm]",
            "Cluster rotation on the GOT [deg]",
            "Corner displacement [mm]",
        ),
    )
    fig.add_trace(_line(t, res.ux[w] * 1e3, "CoM ΔX", C1, hover_fmt=".2f"), row=1, col=1)
    fig.add_trace(_line(t, res.uy[w] * 1e3, "CoM ΔY", C2, hover_fmt=".2f"), row=1, col=1)
    fig.add_trace(_line(t, np.rad2deg(res.phi[w]), "Rotation", C1, showlegend=False, hover_fmt=".3f"), row=2, col=1)
    for i, name in enumerate(CORNER_NAMES):
        fig.add_trace(_line(t, res.corner_disp[w, i] * 1e3, name, CORNER_COLORS[i], hover_fmt=".2f"), row=3, col=1)
    fig.add_hline(
        y=tolerance_mm,
        line=dict(color=CRITICAL, width=1.2),
        annotation_text=f"Allowed {tolerance_mm:g} mm",
        annotation_position="top left",
        annotation_font_color=CRITICAL,
        row=3,
        col=1,
    )
    _shade_slip(fig, res, rows=[1, 2, 3], t0=t[0], t1=t[-1])
    fig.update_xaxes(title_text="Time from recording start [s]", row=3, col=1)
    return _base_layout(fig, 640)


def fig_corner_paths(res: SimulationResult, t0=None, t1=None) -> go.Figure:
    """Small multiples: path of each corner relative to its start position, laid out like the cluster."""
    w = _window(res, t0, t1)
    xy = res.corner_xy[w]
    ref = xy[0]
    # layout: top row = left side (+Y), right column = front (+X)
    pos = {"Rear-Left": (1, 1), "Front-Left": (1, 2), "Rear-Right": (2, 1), "Front-Right": (2, 2)}
    titles = ["Rear-Left (RL)", "Front-Left (FL)", "Rear-Right (RR)", "Front-Right (FR)"]
    fig = make_subplots(rows=2, cols=2, subplot_titles=titles, horizontal_spacing=0.12, vertical_spacing=0.16)
    dx_all = (xy[:, :, 0] - ref[None, :, 0]) * 1e3
    dy_all = (xy[:, :, 1] - ref[None, :, 1]) * 1e3
    lim = max(0.5, float(np.nanmax(np.abs(np.concatenate([dx_all.ravel(), dy_all.ravel()])))) * 1.15)
    for i, name in enumerate(CORNER_NAMES):
        r, c = pos[name]
        dx, dy = dx_all[:, i], dy_all[:, i]
        step = max(1, len(dx) // 4000)
        fig.add_trace(
            go.Scatter(
                x=dx[::step],
                y=dy[::step],
                mode="lines",
                line=dict(color=C1, width=2),
                name=name,
                showlegend=False,
                hovertemplate="ΔX %{x:.2f} mm<br>ΔY %{y:.2f} mm<extra>" + name + "</extra>",
            ),
            row=r,
            col=c,
        )
        fig.add_trace(
            go.Scatter(
                x=[dx[-1]],
                y=[dy[-1]],
                mode="markers+text",
                marker=dict(color=C1, size=9),
                text=[f"{np.hypot(dx[-1], dy[-1]):.2f} mm"],
                textposition="top center",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=r,
            col=c,
        )
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[0],
                mode="markers",
                marker=dict(color=MUTED, size=7, symbol="circle-open"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=r,
            col=c,
        )
    for i in range(1, 5):
        sx = "" if i == 1 else str(i)
        fig.layout[f"xaxis{sx}"].update(range=[-lim, lim], title_text="ΔX [mm]", zeroline=True)
        fig.layout[f"yaxis{sx}"].update(
            range=[-lim, lim], title_text="ΔY [mm]", zeroline=True, scaleanchor=f"x{sx}", scaleratio=1
        )
    fig.update_layout(height=620, margin=dict(l=10, r=10, t=50, b=10), hovermode="closest")
    return fig


# --------------------------------------------------------------------------- #
# Animated top view
# --------------------------------------------------------------------------- #
def _rect(cx, cy, L, W, ang):
    pts = np.array([[L / 2, W / 2], [L / 2, -W / 2], [-L / 2, -W / 2], [-L / 2, W / 2], [L / 2, W / 2]])
    c, s = np.cos(ang), np.sin(ang)
    x = cx + c * pts[:, 0] - s * pts[:, 1]
    y = cy + s * pts[:, 0] + c * pts[:, 1]
    return x, y


def _rot_pts(x, y, ang):
    c, s = np.cos(ang), np.sin(ang)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    return c * x - s * y, s * x + c * y


def auto_exaggeration(res: SimulationResult, sl: slice) -> float:
    peak = float(np.nanmax(res.corner_disp[sl])) if res.corner_disp[sl].size else 0.0
    if peak <= 1e-6:
        return 1.0
    k = 0.12 * min(res.cluster.length, res.cluster.width) / peak
    return float(np.clip(np.round(k), 1, 2000))


def fig_topview(
    kin: Kinematics,
    res: SimulationResult,
    t0: float,
    t1: float,
    exaggeration: float = 1.0,
    plant_view: bool = False,
    max_frames: int = 600,
    playback_speed: float = 1.0,
    travel: dict | None = None,
) -> go.Figure:
    """Animated top view of the GOT and the cell cluster (optionally with the travel direction)."""
    cl = res.cluster
    L, W = cl.length, cl.width
    margin_got = 0.08 * max(L, W)
    Lg, Wg = L + 2 * margin_got, W + 2 * margin_got
    com0 = res.patch.com
    sl = _window(res, t0, t1)
    idx_all = np.arange(sl.start, min(sl.stop, len(res.t)))
    if len(idx_all) < 2:
        idx_all = np.arange(0, min(2, len(res.t)))

    # frame indices: regular grid + denser sampling while sliding
    dt = res.t[1] - res.t[0]
    n_reg = min(max_frames, len(idx_all))
    reg = idx_all[np.linspace(0, len(idx_all) - 1, n_reg).astype(int)]
    slide_idx = idx_all[res.sliding[idx_all]]
    if len(slide_idx):
        step = max(1, int(round(0.02 / dt)))
        slide_idx = slide_idx[::step][: max_frames // 2]
    frames_idx = np.unique(np.concatenate([reg, slide_idx]))
    # include the sample before each slip start
    frames_idx = (
        np.unique(np.concatenate([frames_idx, np.clip(slide_idx - 1, 0, None)])) if len(slide_idx) else frames_idx
    )

    k = exaggeration
    R = 0.5 * np.hypot(Lg, Wg) + (0.42 if travel is not None else 0.15) * max(L, W)
    t_off, t_len = 0.05 * max(L, W), 0.25 * max(L, W)  # travel arrow: gap to the GOT edge and length
    heading0 = kin.heading[frames_idx[0]]
    arrow_scale = (W / 2) / 0.25  # 0.25 g -> half the width

    def frame_traces(i):
        ang = (kin.heading[i] - heading0) if plant_view else 0.0
        # GOT and nominal footprint
        gx, gy = _rect(0, 0, Lg, Wg, ang)
        nx_, ny_ = _rect(0, 0, L, W, ang)
        # cluster pose (exaggerated)
        ux, uy, ph = k * res.ux[i], k * res.uy[i], k * res.phi[i]
        com = com0 + np.array([ux, uy])
        corners_b = cl.corners() - com0
        c, s = np.cos(ph), np.sin(ph)
        cxs = com[0] + c * corners_b[:, 0] - s * corners_b[:, 1]
        cys = com[1] + s * corners_b[:, 0] + c * corners_b[:, 1]
        poly_x = np.r_[cxs, cxs[0]]
        poly_y = np.r_[cys, cys[0]]
        poly_x, poly_y = _rot_pts(poly_x, poly_y, ang)
        cx_r, cy_r = _rot_pts(cxs, cys, ang)
        com_r = _rot_pts([com[0]], [com[1]], ang)
        # inertial load direction (= -specific force at the CoM)
        ax_, ay_ = -res.fhx_com[i] * arrow_scale, -res.fhy_com[i] * arrow_scale
        ax_r, ay_r = _rot_pts([com[0], com[0] + ax_], [com[1], com[1] + ay_], ang)
        # ICR (body frame -> displayed frame)
        if res.sliding[i] and np.isfinite(res.icr_x[i]):
            ib = np.array([res.icr_x[i], res.icr_y[i]]) - com0
            ix = com[0] + c * ib[0] - s * ib[1]
            iy = com[1] + s * ib[0] + c * ib[1]
            ixr, iyr = _rot_pts([ix], [iy], ang)
            if np.hypot(ixr[0], iyr[0]) > R:
                ixr, iyr = [None], [None]
        else:
            ixr, iyr = [None], [None]
        sliding = bool(res.sliding[i])
        stick_xy = (poly_x, poly_y) if not sliding else ([None], [None])
        slip_xy = (poly_x, poly_y) if sliding else ([None], [None])
        # CoM trail
        j0 = frames_idx[0]
        tr = np.arange(j0, i + 1, max(1, (i + 1 - j0) // 150))
        tx = com0[0] + k * res.ux[tr]
        ty = com0[1] + k * res.uy[tr]
        if plant_view:
            tx, ty = _rot_pts(tx, ty, ang)
        # travel direction arrow, drawn outside the GOT on the side the carrier is moving to
        trav_xy, trav_unc_xy = ([None], [None]), ([None], [None])
        motion_line = ""
        if travel is not None:
            stt = int(travel["state"][i])
            if stt == STATE_MOVING:
                dx, dy = float(travel["ux"][i]), float(travel["uy"][i])
                t_edge = min(
                    (Lg / 2) / abs(dx) if abs(dx) > 1e-9 else np.inf,
                    (Wg / 2) / abs(dy) if abs(dy) > 1e-9 else np.inf,
                )
                p0, p1 = t_edge + t_off, t_edge + t_off + t_len
                xy = _rot_pts([p0 * dx, p1 * dx], [p0 * dy, p1 * dy], ang)
                low = travel["confidence"][i] == "Low"
                if low:
                    trav_unc_xy = xy
                else:
                    trav_xy = xy
                spd = travel["speed"][i]
                lab = str(travel["label"][i])
                motion_line = f"<br>Carrier moving {lab}" + (f" ~{spd:.2f} m/s" if np.isfinite(spd) else "")
                motion_line += " (direction uncertain)" if low else ""
            elif stt == STATE_ROTATING:
                motion_line = "<br>Carrier rotating on turntable"
            elif stt == STATE_UNKNOWN:
                motion_line = "<br>Carrier moving (direction uncertain)"
            else:
                motion_line = "<br>Carrier stopped"
        clock = pd.Timestamp(res.clock[i]).strftime("%H:%M:%S.%f")[:-4]
        mu = res.mu_req[i]
        mu_txt = f"{mu:.3f}" if np.isfinite(mu) else "∞ (lift-off)"
        state = "SLIDING" if sliding else "STICKING"
        status = (
            f"{clock}   t = {res.t[i]:.2f} s<br>"
            f"Required μ {mu_txt}  (μs {res.contact.mu_s:.3f})   {state}<br>"
            f"Max corner shift {res.corner_disp[i].max() * 1e3:.2f} mm   Rotation {np.rad2deg(res.phi[i]):.3f}°"
            f"{motion_line}"
        )
        return [
            go.Scatter(x=gx, y=gy),
            go.Scatter(x=nx_, y=ny_),
            go.Scatter(x=stick_xy[0], y=stick_xy[1]),
            go.Scatter(x=slip_xy[0], y=slip_xy[1]),
            go.Scatter(x=cx_r, y=cy_r),
            go.Scatter(x=ax_r, y=ay_r),
            go.Scatter(x=ixr, y=iyr),
            go.Scatter(x=tx, y=ty),
            go.Scatter(x=com_r[0], y=com_r[1]),
            go.Scatter(x=trav_xy[0], y=trav_xy[1]),
            go.Scatter(x=trav_unc_xy[0], y=trav_unc_xy[1]),
            go.Scatter(x=[-R + 0.03 * R], y=[R - 0.03 * R], text=[status]),
        ]

    first = frame_traces(frames_idx[0])
    styles = [
        dict(
            mode="lines",
            line=dict(color=MUTED, width=2),
            fill="toself",
            fillcolor="rgba(137,135,129,0.08)",
            name="GOT (schematic)",
            hoverinfo="skip",
        ),
        dict(mode="lines", line=dict(color=MUTED, width=1), name="Nominal cluster position", hoverinfo="skip"),
        dict(
            mode="lines",
            line=dict(color=C1, width=2),
            fill="toself",
            fillcolor="rgba(42,120,214,0.30)",
            name="Cluster (sticking)",
            hoverinfo="skip",
        ),
        dict(
            mode="lines",
            line=dict(color=CRITICAL, width=2),
            fill="toself",
            fillcolor="rgba(208,59,59,0.35)",
            name="Cluster (sliding)",
            hoverinfo="skip",
        ),
        dict(
            mode="markers+text",
            marker=dict(color=INK_2, size=7),
            text=CORNER_SHORT,
            textposition="top center",
            name="Corners",
            showlegend=False,
            hoverinfo="skip",
        ),
        dict(
            mode="lines+markers",
            line=dict(color=C2, width=3),
            marker=dict(symbol="arrow", angleref="previous", size=[0, 14], color=C2),
            name="Inertial load direction",
            hoverinfo="skip",
        ),
        dict(
            mode="markers",
            marker=dict(symbol="x", size=13, color=CRITICAL, line=dict(width=1)),
            name="Pivot (instant centre of rotation)",
            hoverinfo="skip",
        ),
        dict(mode="lines", line=dict(color=C3, width=1.5), name="CoM trail", hoverinfo="skip"),
        dict(
            mode="markers",
            marker=dict(color=INK_2, size=6, symbol="circle"),
            name="Cluster CoM",
            showlegend=False,
            hoverinfo="skip",
        ),
        dict(
            mode="lines+markers",
            line=dict(color=SERIES[6], width=4),
            marker=dict(symbol="arrow", angleref="previous", size=[0, 16], color=SERIES[6]),
            name="Travel direction",
            hoverinfo="skip",
            showlegend=travel is not None,
        ),
        dict(
            mode="lines+markers",
            line=dict(color=MUTED, width=3),
            marker=dict(symbol="arrow", angleref="previous", size=[0, 14], color=MUTED),
            name="Travel direction (uncertain)",
            hoverinfo="skip",
            showlegend=travel is not None,
        ),
        dict(mode="text", textposition="bottom right", textfont=dict(size=12), showlegend=False, hoverinfo="skip"),
    ]
    data = []
    for tr, st_ in zip(first, styles):
        tr.update(**st_)
        data.append(tr)

    frames = []
    for i in frames_idx:
        frames.append(go.Frame(data=frame_traces(i), name=f"f{i}", traces=list(range(len(data)))))

    times = res.t[frames_idx]
    # playback: frame duration proportional to real time between frames
    med_gap = float(np.median(np.diff(times))) if len(times) > 1 else dt
    frame_ms = int(np.clip(1000 * med_gap / max(playback_speed, 1e-3), 15, 2000))

    label_every = max(1, len(frames_idx) // 8)
    fig = go.Figure(data=data, frames=frames)
    title_k = f" — displacement magnified ×{k:g}" if k != 1 else ""
    fig.update_layout(
        title=dict(
            text=("Plant view (carrier rotates with measured heading)" if plant_view else "Carrier view (GOT fixed)")
            + title_k,
            x=0,
            font=dict(size=14),
        ),
        height=700,
        margin=dict(l=10, r=10, t=60, b=130),
        xaxis=dict(range=[-R, R], title_text="X [m]  (Front →)", zeroline=False),
        yaxis=dict(range=[-R, R], title_text="Y [m]  (Left ↑)", zeroline=False, scaleanchor="x", scaleratio=1),
        legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02),
        hovermode=False,
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                showactive=False,
                x=0.0,
                y=-0.1,
                xanchor="left",
                yanchor="top",
                pad=dict(t=30, r=10),
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=frame_ms, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0),
                                mode="immediate",
                            ),
                        ],
                    ),
                    dict(
                        label="❚❚ Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(frame=dict(duration=0, redraw=False), transition=dict(duration=0), mode="immediate"),
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                x=0.16,
                len=0.84,
                y=-0.1,
                yanchor="top",
                pad=dict(t=30),
                currentvalue=dict(visible=False),
                ticklen=3,
                minorticklen=0,
                steps=[
                    dict(
                        method="animate",
                        value=f"f{i}",
                        label=(f"{tt:.1f} s" if n % label_every == 0 else ""),
                        args=[
                            [f"f{i}"],
                            dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0)),
                        ],
                    )
                    for n, (i, tt) in enumerate(zip(frames_idx, times))
                ],
            )
        ],
    )
    return fig


# --------------------------------------------------------------------------- #
# Sensitivity and comparison
# --------------------------------------------------------------------------- #
def fig_sweep(
    sw: pd.DataFrame, mu_measured: float, mu_noslip: float, tolerance_mm: float
) -> tuple[go.Figure, go.Figure]:
    f1 = go.Figure()
    f1.add_trace(
        go.Scatter(
            x=sw["mu_s"],
            y=sw["Final max corner shift [mm]"],
            mode="lines+markers",
            name="Final max corner shift",
            line=dict(color=C1, width=2),
            marker=dict(size=9),
            hovertemplate="μs %{x:.3f}<br>%{y:.2f} mm<extra></extra>",
        )
    )
    f1.add_hline(
        y=tolerance_mm,
        line=dict(color=CRITICAL, width=1.2),
        annotation_text=f"Allowed {tolerance_mm:g} mm",
        annotation_position="top right",
        annotation_font_color=CRITICAL,
    )
    f2 = go.Figure()
    f2.add_trace(
        go.Scatter(
            x=sw["mu_s"],
            y=sw["Final rotation [deg]"].abs(),
            mode="lines+markers",
            name="Final rotation",
            line=dict(color=C1, width=2),
            marker=dict(size=9),
            hovertemplate="μs %{x:.3f}<br>%{y:.3f}°<extra></extra>",
        )
    )
    for f in (f1, f2):
        f.add_vline(
            x=mu_measured,
            line=dict(color=INK_2, width=1),
            annotation_text=f"Measured μs {mu_measured:.3f}",
            annotation_position="top",
        )
        if np.isfinite(mu_noslip):
            f.add_vline(
                x=mu_noslip,
                line=dict(color=C3, width=1.5),
                annotation_text=f"No-slip threshold {mu_noslip:.3f}",
                annotation_position="bottom right",
            )
        f.update_xaxes(title_text="Static friction coefficient μs [-]")
        f.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
    f1.update_yaxes(title_text="Final max corner shift [mm]", rangemode="tozero")
    f2.update_yaxes(title_text="Final |rotation| [deg]", rangemode="tozero")
    f1.update_layout(title=dict(text="Corner displacement at the end of the record", x=0, font=dict(size=14)))
    f2.update_layout(title=dict(text="Cluster rotation at the end of the record", x=0, font=dict(size=14)))
    return f1, f2


def fig_compare(df: pd.DataFrame, mu_s: float) -> go.Figure:
    d = df  # keep the given (chronological) order, oldest at the top
    fig = go.Figure(
        go.Bar(
            x=d["Required μ (no slip)"],
            y=d["Recording"],
            orientation="h",
            marker=dict(color=C1),
            hovertemplate="%{y}<br>Required μ %{x:.3f}<extra></extra>",
            name="Required μ",
        )
    )
    fig.add_vline(
        x=mu_s,
        line=dict(color=CRITICAL, width=1.5),
        annotation_text=f"μs {mu_s:.3f}",
        annotation_position="top",
        annotation_font_color=CRITICAL,
    )
    fig.update_layout(
        title=dict(text="Friction coefficient required to avoid any slip", x=0, font=dict(size=14)),
        height=max(260, 70 + 45 * len(d)),
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title="Required μ [-]",
        bargap=0.35,
        showlegend=False,
        yaxis=dict(autorange="reversed"),
    )
    return fig


# --------------------------------------------------------------------------- #
# Turntables
# --------------------------------------------------------------------------- #
def fig_rotation_timeline(kin: Kinematics, res: SimulationResult, rotations, pad_s: float) -> go.Figure:
    """Yaw rate, heading and friction demand with the detected rotations highlighted."""
    t = res.t
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=(
            "Carrier yaw rate [deg/s]  (shaded: detected rotations)",
            "Carrier heading [deg]  (integrated yaw rate)",
            "Required friction coefficient  (light bands: turntable entry / exit windows)",
        ),
    )
    fig.add_trace(_line(t, np.rad2deg(kin.wz), "Yaw rate", C1, showlegend=False, hover_fmt=".2f"), row=1, col=1)
    fig.add_trace(_line(t, np.rad2deg(kin.heading), "Heading", C1, showlegend=False, hover_fmt=".1f"), row=2, col=1)
    fig.add_trace(_line(t, np.clip(res.mu_req, 0, 5), "Required μ", C1, showlegend=False), row=3, col=1)
    fig.add_hline(
        y=res.contact.mu_s,
        line=dict(color=CRITICAL, width=1.2),
        annotation_text=f"μs = {res.contact.mu_s:.3f}",
        annotation_position="top left",
        annotation_font_color=CRITICAL,
        row=3,
        col=1,
    )
    for r in rotations:
        turn = r.kind == "Turntable"
        fill = "rgba(27,175,122,0.22)" if turn else "rgba(137,135,129,0.20)"
        for row in (1, 2, 3):
            fig.add_vrect(x0=r.t0, x1=r.t1, fillcolor=fill, line_width=0, layer="below", row=row, col=1)
        if turn:
            for row in (3,):
                fig.add_vrect(
                    x0=r.t0 - pad_s,
                    x1=r.t0,
                    fillcolor="rgba(235,104,52,0.10)",
                    line_width=0,
                    layer="below",
                    row=row,
                    col=1,
                )
                fig.add_vrect(
                    x0=r.t1,
                    x1=r.t1 + pad_s,
                    fillcolor="rgba(235,104,52,0.10)",
                    line_width=0,
                    layer="below",
                    row=row,
                    col=1,
                )
        fig.add_annotation(
            x=0.5 * (r.t0 + r.t1),
            y=1.0,
            xref="x",
            yref="y domain",
            text=f"{r.label.replace('Turn ', 'T').replace('Minor ', 'm')}<br>{r.angle_deg:+.0f}°",
            showarrow=False,
            yanchor="bottom",
            font=dict(size=11, color=INK_2),
        )
    _shade_slip(fig, res, rows=[3])
    fig.update_xaxes(title_text="Time from recording start [s]", row=3, col=1)
    return _base_layout(fig, 720)


def fig_turn_profiles(kin: Kinematics, rotations, pre_s: float = 1.0, post_s: float = 2.0) -> go.Figure:
    """Overlay of all turntable rotations aligned at their start: |yaw rate| and yaw acceleration."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            "|Yaw rate| [deg/s]",
            "Yaw acceleration in the turning direction [deg/s²]  (+ spin-up, − braking)",
        ),
    )
    turns = [r for r in rotations if r.kind == "Turntable"][: len(SERIES)]
    for n, r in enumerate(turns):
        i0 = max(0, int(np.searchsorted(kin.t, r.t0 - pre_s)))
        i1 = min(len(kin.t), int(np.searchsorted(kin.t, r.t1 + post_s)) + 1)
        x = kin.t[i0:i1] - r.t0
        sign = 1.0 if r.angle_deg >= 0 else -1.0
        name = f"{r.label} ({r.angle_deg:+.1f}°)"
        col = SERIES[n]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=np.abs(np.rad2deg(kin.wz[i0:i1])),
                mode="lines",
                name=name,
                legendgroup=name,
                line=dict(color=col, width=2),
                hovertemplate=name + ": %{y:.2f} °/s<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=sign * np.rad2deg(kin.alpha[i0:i1]),
                mode="lines",
                name=name,
                legendgroup=name,
                showlegend=False,
                line=dict(color=col, width=1.5),
                hovertemplate=name + ": %{y:.1f} °/s²<extra></extra>",
            ),
            row=2,
            col=1,
        )
    fig.update_xaxes(title_text="Time from rotation start [s]", row=2, col=1)
    return _base_layout(fig, 560)


# --------------------------------------------------------------------------- #
# Dynamic friction limit and speed profile
# --------------------------------------------------------------------------- #
def fig_dynamic_limit(kin: Kinematics, res: SimulationResult, t0=None, t1=None) -> go.Figure:
    """Horizontal acceleration at the cluster CoM against the friction limit that moves with the vertical acceleration."""
    w = _window(res, t0, t1)
    t = res.t[w]
    lim = res.contact.mu_s * np.clip(kin.fz[w], 0, None)
    fig = go.Figure()
    fig.add_trace(_line(t, res.fh_com[w], "Horizontal acceleration at cluster CoM", C1, width=1.5))
    fig.add_trace(_line(t, lim, "Dynamic friction limit μs·f_z", CRITICAL, width=1.5))
    fig.add_hline(
        y=res.contact.mu_s,
        line=dict(color=MUTED, width=1),
        annotation_text=f"Static limit at 1 g = μs ({res.contact.mu_s:.3f} g)",
        annotation_position="top left",
    )
    _shade_slip(fig, res, t0=t[0], t1=t[-1])
    fig.update_yaxes(title_text="Acceleration [g]", rangemode="tozero")
    fig.update_xaxes(title_text="Time from recording start [s]")
    fig.update_layout(
        title=dict(
            text="Horizontal acceleration vs. dynamic friction limit (the limit drops when the vertical acceleration drops)",
            x=0,
            font=dict(size=14),
        )
    )
    return _base_layout(fig, 380)


def fig_speed_timeline(kin: Kinematics, res: SimulationResult, speed, a_long, qual, rotations) -> go.Figure:
    """Estimated travel speed, longitudinal acceleration vs. friction limit, and yaw rate for the whole record."""
    t = res.t
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=(
            "Estimated travel speed [m/s]  (grey: low reliability, long segment without a stop)",
            "Acceleration along the travel direction [g]  (+ speeding up, − braking) vs. dynamic friction limit (worst case in 0.5 s)",
            "Carrier yaw rate [deg/s]  (turntables)",
        ),
    )
    v_ok = np.where(qual == 2, np.nan, speed)
    v_poor = np.where(qual == 2, speed, np.nan)
    fig.add_trace(_line(t, v_ok, "Travel speed", C1, width=2, hover_fmt=".2f"), row=1, col=1)
    fig.add_trace(_line(t, v_poor, "Travel speed (low reliability)", MUTED, width=1.5, hover_fmt=".2f"), row=1, col=1)
    lim = minimum_filter1d(res.contact.mu_s * np.clip(kin.fz, 0, None), max(1, int(round(0.5 * kin.fs))))
    fig.add_trace(_line(t, a_long, "Longitudinal acceleration", C1, showlegend=False), row=2, col=1)
    fig.add_trace(_line(t, lim, "Dynamic friction limit μs·f_z", CRITICAL, width=1.2), row=2, col=1)
    fig.add_trace(_line(t, -lim, "−Dynamic friction limit", CRITICAL, width=1.2, showlegend=False), row=2, col=1)
    fig.add_trace(_line(t, np.rad2deg(kin.wz), "Yaw rate", C1, showlegend=False, hover_fmt=".1f"), row=3, col=1)
    for r in rotations:
        if r.kind == "Turntable":
            for row in (1, 2, 3):
                fig.add_vrect(
                    x0=r.t0, x1=r.t1, fillcolor="rgba(27,175,122,0.22)", line_width=0, layer="below", row=row, col=1
                )
            fig.add_annotation(
                x=0.5 * (r.t0 + r.t1),
                y=1.0,
                xref="x",
                yref="y domain",
                text=r.label.replace("Turn ", "T"),
                showarrow=False,
                yanchor="bottom",
                font=dict(size=11, color=INK_2),
            )
    _shade_slip(fig, res, rows=[2])
    fig.update_yaxes(rangemode="tozero", row=1, col=1)
    fig.update_xaxes(title_text="Time from recording start [s]", row=3, col=1)
    return _base_layout(fig, 760)


def fig_speed_ramps(kin: Kinematics, speed, seg_table: pd.DataFrame, window_s: float = 10.0) -> go.Figure:
    """Start and stop ramps of the reliable motion segments, overlaid to compare stations."""
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.06,
        subplot_titles=("Start ramps (aligned at the start)", "Stop ramps (aligned at the stop)"),
    )
    good = seg_table[seg_table["Quality"] == "Good"] if len(seg_table) else seg_table
    for n, (_, row) in enumerate(good.head(len(SERIES)).iterrows()):
        name = f"Seg {int(row['Segment'])} · {str(row['Direction']).split(' (')[0]} · {row['Start clock']}"
        col = SERIES[n]
        t0, t1 = float(row["Start [s]"]), float(row["End [s]"])
        i0 = int(np.searchsorted(kin.t, t0 - 1.0))
        i1 = int(np.searchsorted(kin.t, min(t1, t0 + window_s)))
        fig.add_trace(
            go.Scatter(
                x=kin.t[i0:i1] - t0,
                y=speed[i0:i1],
                mode="lines",
                name=name,
                legendgroup=name,
                line=dict(color=col, width=2),
                hovertemplate=name + ": %{y:.2f} m/s<extra></extra>",
            ),
            row=1,
            col=1,
        )
        j0 = int(np.searchsorted(kin.t, max(t0, t1 - window_s)))
        j1 = int(np.searchsorted(kin.t, t1 + 1.0))
        fig.add_trace(
            go.Scatter(
                x=kin.t[j0:j1] - t1,
                y=speed[j0:j1],
                mode="lines",
                name=name,
                legendgroup=name,
                showlegend=False,
                line=dict(color=col, width=2),
                hovertemplate=name + ": %{y:.2f} m/s<extra></extra>",
            ),
            row=1,
            col=2,
        )
    fig.update_xaxes(title_text="Time from start [s]", row=1, col=1)
    fig.update_xaxes(title_text="Time to stop [s]", row=1, col=2)
    fig.update_yaxes(title_text="Speed [m/s]", rangemode="tozero", row=1, col=1)
    return _base_layout(fig, 420)


# --------------------------------------------------------------------------- #
# Measure tab
# --------------------------------------------------------------------------- #
def fig_measure_overview(kin: Kinematics, res: SimulationResult, rotations, w0: float, w1: float) -> go.Figure:
    """Whole-record strip used to pick the zoom window (drag to select)."""
    t = res.t
    fig = go.Figure()
    fig.add_trace(_line(t, res.fh_com, "Horizontal accel. at CoM [g]", C1, showlegend=False, max_points=2000))
    for r in rotations:
        if r.kind == "Turntable":
            fig.add_vrect(x0=r.t0, x1=r.t1, fillcolor="rgba(27,175,122,0.22)", line_width=0, layer="below")
    fig.add_vrect(x0=w0, x1=w1, fillcolor="rgba(235,104,52,0.25)", line=dict(color=C2, width=1.5), layer="below")
    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=40, b=10),
        title=dict(
            text="Whole record · drag across it to choose the zoom window (orange). Green = turntables.",
            x=0,
            font=dict(size=13),
        ),
        dragmode="select",
        selectdirection="h",
        hovermode="x unified",
        showlegend=False,
    )
    fig.update_yaxes(title_text="g")
    fig.update_xaxes(title_text="Time from recording start [s]", hoverformat=".2f")
    return fig


def fig_measure_detail(
    kin: Kinematics, signals: dict, w0: float, w1: float, a: float, b: float, max_points: int = 20000
) -> go.Figure:
    """Full-resolution view of the zoom window with the measured interval A-B.

    signals: {label: (array over kin.t, number format)}; lines are the simulation-rate signal, markers the
    samples actually recorded by the sensor.
    """
    names = list(signals.keys())
    n = max(1, len(names))
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=names)
    i0 = max(0, int(np.searchsorted(kin.t, w0)) - 1)
    i1 = min(len(kin.t), int(np.searchsorted(kin.t, w1, side="right")) + 1)
    t = kin.t[i0:i1]
    tn = kin.t_native[(kin.t_native >= w0) & (kin.t_native <= w1)] if len(kin.t_native) else np.zeros(0)
    for k, name in enumerate(names, start=1):
        y_all, fmt = signals[name]
        y = np.asarray(y_all, dtype=float)[i0:i1]
        col = SERIES[(k - 1) % len(SERIES)]
        fig.add_trace(
            _line(t, y, name, col, width=1.5, showlegend=False, max_points=max_points, hover_fmt=fmt), row=k, col=1
        )
        if len(tn):
            yn = np.interp(tn, kin.t, np.nan_to_num(np.asarray(y_all, dtype=float), nan=0.0))
            fig.add_trace(
                go.Scatter(
                    x=tn,
                    y=yn,
                    mode="markers",
                    marker=dict(color=col, size=6, line=dict(color="#fcfcfb", width=1)),
                    name="Recorded samples",
                    showlegend=False,
                    hovertemplate="sample %{x:.3f} s<extra></extra>",
                ),
                row=k,
                col=1,
            )
    lo, hi = min(a, b), max(a, b)
    fig.add_vrect(x0=lo, x1=hi, fillcolor="rgba(235,104,52,0.15)", line_width=0, layer="below", row="all", col=1)
    for x, lab in ((lo, "A"), (hi, "B")):
        fig.add_vline(x=x, line=dict(color=C2, width=1.5), row="all", col=1)
    fig.add_annotation(
        x=lo, y=1.0, xref="x", yref="paper", text="A", showarrow=False, yanchor="bottom", font=dict(color=C2, size=13)
    )
    fig.add_annotation(
        x=hi, y=1.0, xref="x", yref="paper", text="B", showarrow=False, yanchor="bottom", font=dict(color=C2, size=13)
    )
    fig.update_xaxes(range=[w0, w1], hoverformat=".3f", showspikes=True, spikemode="across", spikethickness=1)
    fig.update_xaxes(title_text="Time from recording start [s]", row=n, col=1)
    fig.update_layout(
        height=max(320, 190 * n),
        margin=dict(l=10, r=10, t=60, b=10),
        hovermode="x unified",
        dragmode="select",
        selectdirection="h",
        uirevision=f"{w0:.3f}-{w1:.3f}",
        showlegend=False,
    )
    return fig
