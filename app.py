"""
GOT Cluster Slip Simulator
==========================

Streamlit application that simulates the movement of a battery cell cluster
resting on the GOT (lower housing) - held in place by friction only - from
conveyor IMU recordings (WitMotion exports).

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import streamlit as st

from data_io import (
    PreprocessSettings,
    RawRecording,
    kinematics_table,
    native_rate,
    preprocess,
    read_witmotion,
    recording_duration,
)
from slip_model import (
    ClusterParams,
    ContactParams,
    SensorGeometry,
    SimulationResult,
    SolverParams,
    critical_moments,
    friction_sweep,
    results_table,
    simulate,
    summary,
)
from motion import (
    DIRECTIONS,
    TravelSettings,
    detect_motion,
    motion_phase,
    segment_key,
    speed_profile,
    segments_table,
    travel_arrays,
)
from turns import (
    TurnSettings,
    analyse_rotations,
    detect_rotations,
    estimate_axis_all,
    locate,
    turn_insight,
)
from visuals import (
    PLOT_CONFIG,
    auto_exaggeration,
    fig_compare,
    fig_corner_paths,
    fig_friction,
    fig_kinematics,
    fig_motion,
    fig_sweep,
    fig_dynamic_limit,
    fig_rotation_timeline,
    fig_speed_ramps,
    fig_speed_timeline,
    fig_topview,
    fig_turn_profiles,
)

APP_VERSION = "1.0.0"

st.set_page_config(page_title="GOT Cluster Slip Simulator", page_icon="🔋", layout="wide")


# --------------------------------------------------------------------------- #
# Cached computation layer
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, max_entries=32)
def load_raw(data: bytes, name: str) -> RawRecording:
    return read_witmotion(data, name)


@st.cache_data(show_spinner=False, max_entries=32)
def load_kinematics(data: bytes, name: str, pre: dict):
    raw = load_raw(data, name)
    return preprocess(raw, PreprocessSettings(**pre))


@st.cache_data(show_spinner=False, max_entries=12)
def run_simulation(
    data: bytes, name: str, pre: dict, cluster: dict, contact: dict, sensor: dict, solver: dict
) -> SimulationResult:
    kin = load_kinematics(data, name, pre)
    return simulate(
        kin, ClusterParams(**cluster), ContactParams(**contact), SensorGeometry(**sensor), SolverParams(**solver)
    )


@st.cache_data(show_spinner=False, max_entries=8)
def run_sweep(
    data: bytes,
    name: str,
    pre: dict,
    cluster: dict,
    contact: dict,
    sensor: dict,
    solver: dict,
    mus: tuple,
    keep_ratio: bool,
) -> pd.DataFrame:
    kin = load_kinematics(data, name, pre)
    return friction_sweep(
        kin,
        ClusterParams(**cluster),
        ContactParams(**contact),
        SensorGeometry(**sensor),
        list(mus),
        SolverParams(**solver),
        keep_ratio=keep_ratio,
    )


def fmt_clock(ts) -> str:
    return pd.Timestamp(ts).strftime("%H:%M:%S.%f")[:-3]


# --------------------------------------------------------------------------- #
# Landing page
# --------------------------------------------------------------------------- #
def landing():
    st.markdown(
        """
Upload one or more **WitMotion IMU exports** in the sidebar to start.

**What the app does**

1. Reads the raw accelerometer / gyroscope recording of the workpiece carrier and cleans it
   (Bluetooth timestamp jitter, sensor mounting offset, resampling).
2. Transfers the measured carrier motion to the centre of the cell cluster, including turntable rotations.
3. Simulates the cluster on the GOT with a distributed Coulomb friction contact: it predicts **when** the
   cluster slips, **how far** it moves and **how much it rotates**, and **where it pivots**
   (e.g. about one corner).
4. Shows an animated top view, slip-event table, friction sensitivity analysis and a comparison between
   recordings (e.g. before / after a conveyor adjustment).

**Expected file format** (tab-separated, as exported by the WitMotion software):
"""
    )
    st.code(
        "time\tDeviceName\tChipTime()\tAccX(g)\tAccY(g)\tAccZ(g)\tAsX(°/s)\tAsY(°/s)\tAsZ(°/s)\tAngleX(°)\tAngleY(°)\tAngleZ(°)\t...\n"
        "2026-10-02T07:48:45.830\tHC-06(00:0C:BF:19:3F:0D)\tnull\t0.004\t-0.001\t1.001\t0.000\t0.000\t0.000\t-0.44\t-1.41\t64.36\t...",
        language="text",
    )
    st.markdown(
        """
**Coordinate convention** — carrier frame: **X** along the cluster length (Front = +X),
**Y** along the width (Left = +Y), **Z** up. Configure how the sensor is mounted in the sidebar.
"""
    )


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.title("GOT Cluster Slip Simulator")
st.caption(
    "Friction-based stick-slip simulation of a battery cell cluster on the GOT, driven by conveyor IMU recordings."
)

with st.sidebar:
    st.header("1 · Recordings")
    uploads = st.file_uploader(
        "WitMotion export files",
        type=["txt", "csv", "tsv", "log"],
        accept_multiple_files=True,
        help="Tab-separated export of the WitMotion software. Several files can be uploaded and compared.",
    )

if not uploads:
    landing()
    st.stop()

files: dict[str, bytes] = {}
for f in uploads:
    name = f.name
    i = 2
    while name in files:
        name = f"{f.name} ({i})"
        i += 1
    files[name] = f.getvalue()

# Validate files
valid: dict[str, bytes] = {}
for name, data in files.items():
    try:
        load_raw(data, name)
        valid[name] = data
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error(f"{name}: {exc}")
if not valid:
    st.error("None of the uploaded files could be read as a WitMotion export.")
    st.stop()

with st.sidebar:
    active = st.selectbox("Active recording", list(valid.keys()))
    raw = load_raw(valid[active], active)
    dur = recording_duration(raw)
    win = st.slider(
        "Analysis window [s from recording start]",
        min_value=0.0,
        max_value=float(np.floor(dur * 10) / 10),
        value=(0.0, float(np.floor(dur * 10) / 10)),
        step=0.1,
        key=f"win_{active}",
        help="Restrict the analysis to one conveyor section.",
    )
    ts_label = st.selectbox(
        "Timestamp handling",
        ["Smoothed (recommended for Bluetooth)", "As logged (receive time)", "Uniform (sample index)"],
        help=(
            "Bluetooth loggers stamp packets when they arrive at the PC, often in bursts. 'Smoothed' rebuilds "
            "an even sample clock that still follows real dropouts and drift."
        ),
    )
    ts_mode = {"Smoothed": "smoothed", "As logged": "raw", "Uniform": "uniform"}[ts_label.split(" (")[0]]

    st.header("2 · Sensor mounting")
    axis_label = st.selectbox(
        "Sensor X axis points towards",
        ["Carrier +X (Front)", "Carrier +Y (Left)", "Carrier −X (Rear)", "Carrier −Y (Right)"],
        help="Direction of the sensor's printed X axis on the workpiece carrier. Carrier X runs along the cluster length.",
    )
    sensor_yaw = {
        "Carrier +X (Front)": 0.0,
        "Carrier +Y (Left)": 90.0,
        "Carrier −X (Rear)": 180.0,
        "Carrier −Y (Right)": 270.0,
    }[axis_label]
    auto_flip = st.toggle("Auto-detect upside-down mounting", value=True)
    c1, c2 = st.columns(2)
    for _k in ("sensor_x", "sensor_y"):
        if _k not in st.session_state:
            st.session_state[_k] = 0.0
    sx = c1.number_input(
        "Sensor X [m]",
        key="sensor_x",
        step=0.05,
        format="%.3f",
        help="Sensor position on the carrier relative to the nominal cluster centre, along carrier X.",
    )
    sy = c2.number_input(
        "Sensor Y [m]",
        key="sensor_y",
        step=0.05,
        format="%.3f",
        help="Sensor position on the carrier relative to the nominal cluster centre, along carrier Y.",
    )
    st.caption(
        "The sensor offset is used to transfer the measured motion to the cluster centre during turntable "
        "rotations (centripetal and tangential terms). The Turntables tab estimates the sensor position relative "
        "to the turntable axis from the data and can fill these fields."
    )
    bias_on = st.toggle(
        "Remove static offset (rest window)",
        value=True,
        help="Removes sensor tilt / bias using a window where the carrier is at rest.",
    )
    b1, b2 = st.columns(2)
    bias_t0 = b1.number_input("Rest from [s]", value=0.0, min_value=0.0, step=0.5, format="%.1f", disabled=not bias_on)
    bias_t1 = b2.number_input("Rest to [s]", value=3.0, min_value=0.1, step=0.5, format="%.1f", disabled=not bias_on)

    with st.expander("3 · Signal processing"):
        fs_sim = st.selectbox(
            "Simulation sample rate [Hz]",
            [50, 100, 200, 400],
            index=1,
            help="The raw signal is linearly interpolated onto this grid. It does not add information.",
        )
        lp_on = st.toggle("Low-pass filter", value=False)
        fnyq = 0.5 * native_rate(raw)
        lp_fc = st.number_input(
            "Cut-off [Hz]",
            value=float(np.round(min(4.0, 0.8 * fnyq), 1)),
            min_value=0.1,
            max_value=float(max(0.2, np.round(0.95 * fnyq, 1))),
            step=0.5,
            disabled=not lp_on,
        )
        alpha_win = st.number_input(
            "Yaw-acceleration window [s]",
            value=0.3,
            min_value=0.05,
            max_value=3.0,
            step=0.05,
            help="Savitzky-Golay window used to differentiate the yaw rate.",
        )

    with st.expander("4 · Cell cluster", expanded=True):
        mass = st.number_input("Mass [kg]", value=400.0, min_value=1.0, step=10.0)
        c1, c2 = st.columns(2)
        length = c1.number_input("Length [m]", value=2.0, min_value=0.05, step=0.05, format="%.3f")
        width = c2.number_input("Width [m]", value=1.0, min_value=0.05, step=0.05, format="%.3f")
        default_I = mass * (length**2 + width**2) / 12.0
        inertia_on = st.toggle(
            "Override yaw inertia", value=False, help=f"Default (uniform block): {default_I:.1f} kg·m²"
        )
        inertia = st.number_input(
            "Yaw inertia [kg·m²]", value=float(round(default_I, 1)), min_value=0.1, step=5.0, disabled=not inertia_on
        )

    with st.expander("5 · Friction contact", expanded=True):
        c1, c2 = st.columns(2)
        mu_s = c1.number_input("Static μs", value=0.16, min_value=0.01, max_value=2.0, step=0.01, format="%.3f")
        mu_k = c2.number_input(
            "Kinetic μk",
            value=0.16,
            min_value=0.01,
            max_value=2.0,
            step=0.01,
            format="%.3f",
            help="Friction while sliding. Use the static value if unknown.",
        )
        if mu_k > mu_s:
            st.warning("μk is higher than μs; μk will be limited to μs.")
            mu_k = mu_s
        model_label = st.radio(
            "Contact layout",
            ["Full-area contact", "Four corner supports"],
            help=(
                "Full-area: the whole base carries load (linear pressure consistent with the CoM position). "
                "Corner supports: load is carried by four pads; unequal shares reproduce a warped or uneven "
                "support and tend to make the cluster pivot about the most loaded corner."
            ),
        )
        com_off = (0.0, 0.0)
        shares = (25.0, 25.0, 25.0, 25.0)
        inset, pad = 0.10, 0.10
        nx, ny = 16, 8
        if model_label == "Full-area contact":
            c1, c2 = st.columns(2)
            ex = c1.number_input("CoM offset X [m]", value=0.0, step=0.01, format="%.3f")
            ey = c2.number_input("CoM offset Y [m]", value=0.0, step=0.01, format="%.3f")
            com_off = (float(ex), float(ey))
            grid = st.select_slider("Contact grid", ["Coarse", "Standard", "Fine"], value="Standard")
            nx, ny = {"Coarse": (8, 4), "Standard": (16, 8), "Fine": (24, 12)}[grid]
        else:
            st.caption("Normal-load share per corner [%] (normalised to 100 %).")
            c1, c2 = st.columns(2)
            s_rl = c1.number_input("Rear-Left", value=25.0, min_value=0.0, step=5.0)
            s_fl = c2.number_input("Front-Left", value=25.0, min_value=0.0, step=5.0)
            s_rr = c1.number_input("Rear-Right", value=25.0, min_value=0.0, step=5.0)
            s_fr = c2.number_input("Front-Right", value=25.0, min_value=0.0, step=5.0)
            shares = (float(s_fl), float(s_fr), float(s_rr), float(s_rl))
            c1, c2 = st.columns(2)
            inset = c1.number_input("Pad inset [m]", value=0.10, min_value=0.0, step=0.01, format="%.3f")
            pad = c2.number_input("Pad size [m]", value=0.10, min_value=0.005, step=0.01, format="%.3f")
        mu_fac = (1.0, 1.0, 1.0, 1.0)
        nonuni = st.toggle(
            "Non-uniform friction per corner region",
            value=False,
            help=(
                "Multiplies the friction coefficient in each quarter of the footprint (e.g. a dry or rough zone, a "
                "different pad material, contamination). A corner region with higher friction becomes the pivot "
                "when the cluster slips."
            ),
        )
        if nonuni:
            st.caption("Friction factor per region (1.0 = nominal μ).")
            c1, c2 = st.columns(2)
            m_rl = c1.number_input("RL factor", value=1.0, min_value=0.0, max_value=10.0, step=0.1, format="%.2f")
            m_fl = c2.number_input("FL factor", value=1.0, min_value=0.0, max_value=10.0, step=0.1, format="%.2f")
            m_rr = c1.number_input("RR factor", value=1.0, min_value=0.0, max_value=10.0, step=0.1, format="%.2f")
            m_fr = c2.number_input("FR factor", value=1.0, min_value=0.0, max_value=10.0, step=0.1, format="%.2f")
            mu_fac = (float(m_fl), float(m_fr), float(m_rr), float(m_rl))

    with st.expander("6 · Acceptance"):
        tol_mm = st.number_input(
            "Allowed corner displacement [mm]",
            value=5.0,
            min_value=0.01,
            step=0.5,
            help="Displacement of any cluster corner relative to its start position.",
        )

    with st.expander("7 · Turntable detection"):
        tt_rate = st.number_input(
            "Rotation threshold [deg/s]",
            value=0.5,
            min_value=0.05,
            max_value=20.0,
            step=0.1,
            help="Yaw rate above which the carrier is considered to be rotating.",
        )
        tt_min = st.number_input(
            "Minimum turntable angle [deg]",
            value=45.0,
            min_value=5.0,
            max_value=360.0,
            step=5.0,
            help="Rotations at least this large are reported as turntable turns; smaller ones as minor rotations.",
        )
        tt_nom = st.number_input("Nominal turntable angle [deg]", value=90.0, min_value=1.0, max_value=360.0, step=5.0)
        tt_pad = st.number_input(
            "Entry / exit window [s]",
            value=4.0,
            min_value=0.5,
            max_value=30.0,
            step=0.5,
            help="Time analysed before and after each turn (carrier stopping on and leaving the table).",
        )
        tt_merge = st.number_input(
            "Merge gap [s]",
            value=1.0,
            min_value=0.0,
            max_value=10.0,
            step=0.5,
            help="Same-direction rotations separated by less than this are one turn (fast phase + slow positioning).",
        )

    with st.expander("8 · Travel direction"):
        tv_vib = st.number_input(
            "Rolling-vibration threshold [g]",
            value=0.008,
            min_value=0.001,
            max_value=0.2,
            step=0.001,
            format="%.3f",
            help="Vertical vibration (1-s RMS) above which the carrier is considered to be rolling on the conveyor.",
        )
        tv_win = st.number_input(
            "Start / stop window [s]",
            value=2.5,
            min_value=0.5,
            max_value=10.0,
            step=0.5,
            help="The velocity is integrated only over this window after each start and before each stop.",
        )
        tv_snap = st.toggle(
            "Snap to carrier axes", value=True, help="Conveyors move the carrier along one of its own axes (±X or ±Y)."
        )

    with st.expander("Advanced solver settings"):
        dt_sub_ms = st.select_slider("Integration step while sliding [ms]", [0.25, 0.5, 1.0, 2.0], value=1.0)
        n_dirs = st.select_slider("Limit-surface resolution", [1000, 2500, 5000], value=2500)

    st.divider()
    st.caption(f"v{APP_VERSION} · results are model estimates; validate against video and physical measurements.")

# --------------------------------------------------------------------------- #
# Parameter dictionaries (hashable for caching)
# --------------------------------------------------------------------------- #
pre = asdict(
    PreprocessSettings(
        timestamp_mode=ts_mode,
        sensor_yaw_deg=sensor_yaw,
        auto_flip=auto_flip,
        bias_mode="window" if bias_on else "none",
        bias_window_s=(float(bias_t0), float(bias_t1)),
        lowpass_hz=float(lp_fc) if lp_on else None,
        resample_hz=float(fs_sim),
        alpha_window_s=float(alpha_win),
        t_start=float(win[0]),
        t_end=float(win[1]),
    )
)
cluster_d = asdict(
    ClusterParams(
        mass=float(mass), length=float(length), width=float(width), inertia=float(inertia) if inertia_on else None
    )
)
contact_d = asdict(
    ContactParams(
        mu_s=float(mu_s),
        mu_k=float(mu_k),
        model="full" if model_label == "Full-area contact" else "corners",
        com_offset=com_off,
        corner_shares=shares,
        corner_inset=float(inset),
        pad_size=float(pad),
        nx=nx,
        ny=ny,
        corner_mu_factors=mu_fac,
    )
)
sensor_d = asdict(SensorGeometry(x=float(sx), y=float(sy)))
solver_d = asdict(SolverParams(dt_sub=float(dt_sub_ms) / 1000.0, n_dirs=int(n_dirs)))

# --------------------------------------------------------------------------- #
# Computation
# --------------------------------------------------------------------------- #
data = valid[active]
try:
    with st.spinner("Pre-processing and simulating..."):
        kin = load_kinematics(data, active, pre)
        res = run_simulation(data, active, pre, cluster_d, contact_d, sensor_d, solver_d)
except Exception as exc:  # noqa: BLE001
    st.error(f"Simulation failed: {exc}")
    st.stop()

sm = summary(res, kin)
crit = critical_moments(res, kin, n=10)

turn_s = TurnSettings(
    rate_threshold_deg_s=float(tt_rate),
    min_turn_deg=float(tt_min),
    merge_gap_s=float(tt_merge),
    pad_s=float(tt_pad),
    nominal_deg=float(tt_nom),
)
rotations = detect_rotations(kin, turn_s)
turn_tab = analyse_rotations(kin, res, rotations, turn_s)
n_turns = sum(r.kind == "Turntable" for r in rotations)
axis_est = estimate_axis_all(kin, rotations)

# travel direction (auto-detected per motion segment, manual overrides kept in the session)
travel_s = TravelSettings(vib_threshold_g=float(tv_vib), window_s=float(tv_win), snap=bool(tv_snap))
motion_segs = detect_motion(kin, res, rotations, travel_s)
travel_ov_key = f"travel_ov_{active}"
travel_overrides = st.session_state.get(travel_ov_key, {})
travel = travel_arrays(kin, motion_segs, rotations, travel_overrides)
speed, a_long, speed_q, speed_tab = speed_profile(kin, res, motion_segs, travel_overrides)


def _phase_at(t_: float) -> str:
    i_ = int(np.clip(np.searchsorted(kin.t, t_), 0, len(kin.t) - 1))
    return motion_phase(i_, travel, res.fhx_com, res.fhy_com)


# conveyor context for critical moments and slip events
if len(crit):
    crit["Carrier motion"] = [locate(t_, rotations, turn_s.pad_s) for t_ in crit["Time [s]"]]
    crit = crit.rename(columns={"Carrier motion": "Conveyor context"})
    crit.insert(
        crit.columns.get_loc("Conveyor context") + 1, "Motion phase", [_phase_at(t_) for t_ in crit["Time [s]"]]
    )
events_view = res.events.copy()
if len(events_view):
    events_view["Carrier motion"] = [locate(t_, rotations, turn_s.pad_s) for t_ in events_view["Start [s]"]]
    events_view = events_view.rename(columns={"Carrier motion": "Conveyor context"})
    events_view.insert(
        events_view.columns.get_loc("Conveyor context") + 1,
        "Motion phase",
        [_phase_at(t_) for t_ in events_view["Start [s]"]],
    )

# animation window (shared with the Turntables tab)
t_lo, t_hi = round(float(kin.t[0]), 1), round(float(kin.t[-1]), 1)
anim_key = f"anim_{active}_{t_lo:.1f}_{t_hi:.1f}"


def _show_in_animation(a: float, b: float):
    st.session_state[anim_key] = (float(np.clip(round(a, 1), t_lo, t_hi)), float(np.clip(round(b, 1), t_lo, t_hi)))


def _use_axis_as_sensor(x: float, y: float):
    st.session_state["sensor_x"] = round(float(x), 3)
    st.session_state["sensor_y"] = round(float(y), 3)


# --------------------------------------------------------------------------- #
# Header: verdict and KPIs
# --------------------------------------------------------------------------- #
st.subheader(f"{active}")
st.caption(
    f"Device {raw.device} · {fmt_clock(kin.clock[0])} – {fmt_clock(kin.clock[-1])} · "
    f"{kin.duration:.1f} s analysed · native rate {kin.fs_native:.1f} Hz · simulation {kin.fs:.0f} Hz · "
    f"solved in {res.runtime_s:.2f} s"
)

for w_ in kin.warnings + res.patch.notes:
    st.warning(w_)
if len(raw.devices_in_file) > 1:
    st.info(f"The file contains several devices ({', '.join(raw.devices_in_file)}); analysing {raw.device}.")
if res.truncated:
    st.error(
        "The cluster kept sliding for a very long time; the simulation was stopped early. Check the friction values."
    )

if sm["n_events"] == 0:
    st.success(
        f"**No slip predicted.** The highest friction demand in this record is μ = {sm['mu_noslip']:.3f} "
        f"({crit.iloc[0]['Clock'] if len(crit) else '-'}); available μs = {mu_s:.3f} → margin {sm['margin'] - 1:+.0%}."
    )
elif sm["final_corner_mm"] > tol_mm or sm["peak_corner_mm"] > tol_mm:
    st.error(
        f"**Slip predicted — out of tolerance.** {sm['n_events']} slip event(s), max corner shift "
        f"{sm['peak_corner_mm']:.2f} mm (allowed {tol_mm:g} mm), final rotation {sm['final_rot_deg']:.3f}°. "
        f"A friction coefficient of at least {sm['mu_noslip']:.3f} would be needed to avoid any slip."
    )
else:
    st.warning(
        f"**Slip predicted — within tolerance.** {sm['n_events']} slip event(s), max corner shift "
        f"{sm['peak_corner_mm']:.2f} mm (allowed {tol_mm:g} mm), final rotation {sm['final_rot_deg']:.3f}°."
    )

k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Required μ for no slip",
    f"{sm['mu_noslip']:.3f}",
    f"{(sm['margin'] - 1) * 100:+.0f}% margin vs μs",
    help="Highest friction coefficient demanded by the recorded motion (cluster in nominal position).",
)
k2.metric("Slip events", f"{sm['n_events']}", help="Number of predicted stick-slip events.")
k3.metric(
    "Final max corner shift",
    f"{sm['final_corner_mm']:.2f} mm",
    f"{sm['final_corner_mm'] - tol_mm:+.2f} mm vs allowed",
    delta_color="inverse",
)
k4.metric("Final rotation", f"{sm['final_rot_deg']:.3f}°")
k5, k6, k7, k8 = st.columns(4)
k5.metric("Peak horizontal accel. at CoM", f"{sm['peak_fh_g']:.3f} g")
k6.metric(
    "Lowest dynamic friction limit",
    f"{mu_s * max(sm['min_fz_g'], 0.0):.3f} g",
    f"{(max(sm['min_fz_g'], 0.0) - 1) * 100:+.0f}% vs static",
    delta_color="normal",
    help=(
        "Highest horizontal acceleration the cluster can take without sliding = μs × vertical acceleration. "
        f"At rest it equals μs ({mu_s:.3f} g); the minimum vertical acceleration in this record is "
        f"{sm['min_fz_g']:.3f} g."
    ),
)
k7.metric(
    "Turntable rotations",
    f"{n_turns}",
    help="Carrier rotations of at least the minimum turntable angle (see the Turntables tab). "
    f"Peak yaw rate {sm['peak_yaw_rate_deg_s']:.1f} °/s.",
)
k8.metric("Peak yaw accel.", f"{sm['peak_yaw_acc_deg_s2']:.1f} °/s²")

tabs = st.tabs(
    [
        "Overview",
        "Turntables",
        "Speed profile",
        "Top-view animation",
        "Signals",
        "Slip events",
        "Friction sensitivity",
        "Compare recordings",
        "Export",
        "Method",
    ]
)
tab_over, tab_turn, tab_speed, tab_anim, tab_sig, tab_ev, tab_sens, tab_cmp, tab_exp, tab_meth = tabs

# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
with tab_over:
    st.plotly_chart(fig_friction(res), config=PLOT_CONFIG, key="ov_friction")
    st.plotly_chart(fig_dynamic_limit(kin, res), config=PLOT_CONFIG, key="ov_dynlimit")
    st.caption(
        "The friction limit is not constant: when the carrier bumps or drops (vertical acceleration below 1 g) the "
        "cluster is momentarily lighter and the limit falls. Slip happens where the blue curve reaches the red one. "
        "The required-μ chart above already includes this effect and the rotational load."
    )
    st.markdown("##### Most critical moments")
    st.caption(
        "Highest friction demand along the record (slip or not). Use the clock time to find the moment in the video. "
        "Driver tells whether the demand comes mainly from linear acceleration or from yaw acceleration (turntables)."
    )
    st.dataframe(crit, hide_index=True)

# --------------------------------------------------------------------------- #
# Turntables
# --------------------------------------------------------------------------- #
with tab_turn:
    st.caption(
        "Carrier rotations are detected from the gyroscope yaw rate and their angle is its integral. "
        "Entry and exit windows cover the carrier stopping on the table and leaving it."
    )
    if not rotations:
        st.info("No carrier rotation detected with the current settings (sidebar · 7 · Turntable detection).")
    else:
        for msg in turn_insight(turn_tab, mu_s):
            st.markdown(f"- {msg}")
        tt_only = turn_tab[turn_tab["Type"] == "Turntable"]
        mu_cols = ["Required mu · entry", "Required mu · rotation", "Required mu · exit"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Turntable rotations", f"{n_turns}")
        c2.metric("Mean |angle|", f"{tt_only['Angle [deg]'].abs().mean():.1f}°" if len(tt_only) else "-")
        c3.metric(
            "Max deviation from nominal",
            f"{tt_only['Deviation from nominal [deg]'].abs().max():.1f}°" if len(tt_only) else "-",
            help=f"Nominal angle {tt_nom:g}°.",
        )
        mu_near = float(np.nanmax(tt_only[mu_cols].to_numpy(dtype=float))) if len(tt_only) else float("nan")
        c4.metric(
            "Highest μ around turntables",
            f"{mu_near:.3f}" if np.isfinite(mu_near) else "-",
            f"{(mu_s / mu_near - 1) * 100:+.0f}% margin vs μs" if np.isfinite(mu_near) and mu_near > 0 else None,
        )
        st.plotly_chart(fig_rotation_timeline(kin, res, rotations, turn_s.pad_s), config=PLOT_CONFIG, key="tt_timeline")
        st.markdown("##### Detected rotations")
        st.dataframe(turn_tab, hide_index=True)

        st.markdown("##### Turntable axis position")
        rx, ry, se, r2 = axis_est["rx"], axis_est["ry"], axis_est["se"], axis_est["r2"]
        if np.isfinite(rx) and np.isfinite(r2) and r2 >= 0.3 and np.isfinite(se) and se < 0.15:
            st.markdown(
                f"Fitted over all turns, the sensor sits at **X = {rx:+.2f} m, Y = {ry:+.2f} m** from the turntable "
                f"axis ({np.hypot(rx, ry):.2f} m, ± {se:.2f} m, R² = {r2:.2f}). The per-turn values are in the table "
                "above; similar values across turns indicate a reliable estimate."
            )
            st.caption(
                "If the turntable rotates the carrier about the cluster centre, this is the sensor position to use in "
                "the sidebar. Otherwise enter the measured position."
            )
            st.button("Use as sensor position", on_click=_use_axis_as_sensor, args=(rx, ry))
        else:
            st.info(
                "The axis position could not be estimated reliably from this record "
                "(weak angular acceleration or noisy acceleration data)."
            )

        st.markdown("##### Turn profiles")
        st.caption(
            "All turntable rotations aligned at their start. Different ramp shapes or a slow final phase reveal "
            "differences between tables or drives."
        )
        st.plotly_chart(fig_turn_profiles(kin, rotations), config=PLOT_CONFIG, key="tt_profiles")

        st.markdown("##### Turn detail")
        rot_labels = [f"{r.label} · {r.angle_deg:+.1f}° · {fmt_clock(kin.clock[r.i0])}" for r in rotations]
        sel_t = st.selectbox("Rotation", rot_labels, key="tt_sel")
        r_sel = rotations[rot_labels.index(sel_t)]
        w0 = max(float(kin.t[0]), r_sel.t0 - turn_s.pad_s - 1.0)
        w1 = min(float(kin.t[-1]), r_sel.t1 + turn_s.pad_s + 1.0)
        st.button(
            "Show this turn in the Top-view animation",
            on_click=_show_in_animation,
            args=(w0, w1),
            help="Sets the animation window to this turn; then open the Top-view animation tab.",
        )
        st.plotly_chart(fig_friction(res, w0, w1), config=PLOT_CONFIG, key="tt_friction")
        st.plotly_chart(fig_kinematics(kin, res, w0, w1), config=PLOT_CONFIG, key="tt_kin")

# --------------------------------------------------------------------------- #
# Speed profile
# --------------------------------------------------------------------------- #
with tab_speed:
    st.caption(
        "Estimated travel speed of the carrier along the whole route. Each motion segment is integrated from rest to "
        "rest along its travel direction (see Top-view animation → Travel direction per motion segment). The estimate "
        "is good for segments of up to about a minute; longer segments without a stop accumulate error and are shown "
        "in grey. Turntables are shown with their yaw rate."
    )
    good_tab = speed_tab[speed_tab["Quality"] == "Good"] if len(speed_tab) else speed_tab
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Motion segments", f"{len(good_tab)}", help="Segments with a usable speed estimate (Good or Fair).")
    c2.metric(
        "Typical cruise speed",
        f"{good_tab['Cruise speed [m/s]'].median():.2f} m/s" if len(good_tab) else "-",
        help="Median of the cruise speeds of the usable segments.",
    )
    c3.metric(
        "Highest start acceleration",
        f"{good_tab['Peak acceleration [g]'].max():.3f} g" if len(good_tab) else "-",
    )
    c4.metric(
        "Highest braking deceleration",
        f"{good_tab['Peak deceleration [g]'].max():.3f} g" if len(good_tab) else "-",
        help="Includes stopper impacts. Compare with the dynamic friction limit μs·f_z.",
    )
    st.plotly_chart(
        fig_speed_timeline(kin, res, speed, a_long, speed_q, rotations), config=PLOT_CONFIG, key="sp_timeline"
    )
    st.caption(
        "Middle chart: when the acceleration along the travel direction reaches the red dynamic friction limit, the "
        "start or stop ramp alone is enough to make the cluster slide."
    )
    st.markdown("##### Motion segments")
    st.dataframe(speed_tab, hide_index=True)
    st.markdown("##### Start and stop ramps")
    st.caption(
        "Reliable segments overlaid at their start and at their stop. A steeper curve means a harder ramp; "
        "a sudden drop at the stop is a stopper impact."
    )
    if len(good_tab):
        st.plotly_chart(fig_speed_ramps(kin, speed, speed_tab), config=PLOT_CONFIG, key="sp_ramps")
    else:
        st.info("No reliable motion segment in this record.")

# --------------------------------------------------------------------------- #
# Animation
# --------------------------------------------------------------------------- #
with tab_anim:
    if len(res.events):
        ev = res.events.iloc[int(np.argmax(res.events["Max corner shift [mm]"].to_numpy()))]
        centre = 0.5 * (ev["Start [s]"] + ev["End [s]"])
    elif len(crit):
        centre = float(crit.iloc[0]["Time [s]"])
    else:
        centre = t_lo
    d0 = (max(t_lo, centre - 15.0), min(t_hi, centre + 15.0))
    c1, c2, c3, c4 = st.columns([3, 1.3, 1.2, 1.2])
    if anim_key not in st.session_state:
        st.session_state[anim_key] = (
            float(np.clip(round(d0[0], 1), t_lo, t_hi)),
            float(np.clip(round(d0[1], 1), t_lo, t_hi)),
        )
    a_win = c1.slider(
        "Animation window [s]",
        min_value=t_lo,
        max_value=t_hi,
        step=0.1,
        key=anim_key,
        help="Defaults to ±15 s around the largest slip event, or around the most critical moment.",
    )
    view = c2.radio("View", ["Carrier (GOT fixed)", "Plant (carrier rotates)"], key="view")
    speed = c3.selectbox("Playback speed", [0.25, 0.5, 1.0, 2.0, 5.0, 10.0], index=2, format_func=lambda v: f"{v:g}×")
    nfr = c4.selectbox("Max. frames", [300, 600, 1000], index=1)
    sl = slice(int(np.searchsorted(res.t, a_win[0])), int(np.searchsorted(res.t, a_win[1], side="right")))
    k_auto = auto_exaggeration(res, slice(0, len(res.t)))
    c1, c2 = st.columns([1, 3])
    auto_k = c1.toggle(
        "Auto magnification",
        value=True,
        help="Displacements are millimetres on a metre-sized part; they are magnified for visibility.",
    )
    k_val = c2.number_input(
        "Displacement magnification ×", value=float(k_auto), min_value=1.0, max_value=5000.0, step=1.0, disabled=auto_k
    )
    k_use = k_auto if auto_k else float(k_val)
    if a_win[1] - a_win[0] < 0.2:
        st.info("Select a window of at least 0.2 s.")
    else:
        st.plotly_chart(
            fig_topview(
                kin,
                res,
                a_win[0],
                a_win[1],
                exaggeration=k_use,
                plant_view=view.startswith("Plant"),
                max_frames=int(nfr),
                playback_speed=float(speed),
                travel=travel,
            ),
            config=PLOT_CONFIG,
            key="anim",
        )
        st.caption(
            "Blue = cluster sticking, red = sliding. Violet arrow outside the GOT = direction in which the carrier "
            "travels (grey when the direction is uncertain). Orange arrow = direction in which inertia pushes the "
            "cluster relative to the GOT (length ∝ load; 0.25 g reaches half the cluster width): it points backwards "
            "while the carrier speeds up and forwards while it brakes. × = instantaneous pivot while sliding. "
            "The GOT outline is schematic."
        )
        with st.expander("Travel direction per motion segment (auto-detected, editable)"):
            st.caption(
                "Each row is a period in which the carrier rolls on the conveyor. The direction comes from the velocity "
                "after the start and before the stop; 'High' means both agree. If you know the real direction (e.g. from "
                "the video), set it in the Direction column; it is used in the animation and in the Motion phase columns."
            )
            seg_keys = [segment_key(g) for g in motion_segs]
            ed_key = f"travel_ed_{active}_" + hashlib.md5("|".join(seg_keys).encode()).hexdigest()[:10]

            def _save_travel_overrides(ed_key=ed_key, ov_key=travel_ov_key, seg_keys=seg_keys):
                edits = (st.session_state.get(ed_key) or {}).get("edited_rows", {})
                ov = dict(st.session_state.get(ov_key, {}))
                for row, change in edits.items():
                    if "Direction" in change and int(row) < len(seg_keys):
                        ov[seg_keys[int(row)]] = change["Direction"]
                st.session_state[ov_key] = ov

            def _reset_travel_overrides(ed_key=ed_key, ov_key=travel_ov_key):
                st.session_state[ov_key] = {}
                if ed_key in st.session_state:
                    del st.session_state[ed_key]

            seg_df = segments_table(motion_segs, travel_overrides)
            if seg_df.empty:
                st.info(
                    "No motion segment detected. Lower the rolling-vibration threshold in the sidebar (8 · Travel direction)."
                )
            else:
                st.data_editor(
                    seg_df,
                    key=ed_key,
                    on_change=_save_travel_overrides,
                    hide_index=True,
                    disabled=[c for c in seg_df.columns if c != "Direction"],
                    column_config={
                        "Direction": st.column_config.SelectboxColumn(
                            "Direction", options=["Auto", *DIRECTIONS.keys(), "Stopped"], required=True
                        )
                    },
                )
                st.button("Reset manual directions", on_click=_reset_travel_overrides)
        st.plotly_chart(fig_motion(res, tol_mm, a_win[0], a_win[1]), config=PLOT_CONFIG, key="anim_motion")
        st.markdown("##### Corner paths in the animation window")
        st.caption(
            "Each panel sits where the corner is on the cluster (Front to the right, Left at the top). A corner that barely moves while the others travel indicates pivoting about that corner."
        )
        st.plotly_chart(fig_corner_paths(res, a_win[0], a_win[1]), config=PLOT_CONFIG, key="anim_corners")

# --------------------------------------------------------------------------- #
# Signals
# --------------------------------------------------------------------------- #
with tab_sig:
    st.plotly_chart(fig_kinematics(kin, res), config=PLOT_CONFIG, key="sig_kin")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Recording diagnostics")
        st.dataframe(
            pd.DataFrame(
                {
                    "Item": [
                        "Rows in file",
                        "Duplicate packets removed",
                        "Native sample rate [Hz]",
                        "Delivery pauses (> 3× period)",
                        "Axes flipped (upside-down)",
                        "Static offset X [g]",
                        "Static offset Y [g]",
                        "Vertical gain correction",
                        "Yaw-rate offset [deg/s]",
                        "Net heading change [deg]",
                    ],
                    "Value": [
                        f"{raw.n_rows_raw}",
                        f"{raw.n_duplicates_removed}",
                        f"{kin.fs_native:.2f}",
                        f"{len(kin.gaps)}",
                        "Yes" if kin.flipped else "No",
                        f"{kin.bias['fx']:+.4f}",
                        f"{kin.bias['fy']:+.4f}",
                        f"{kin.bias['fz_gain']:.4f}",
                        f"{np.rad2deg(kin.bias['wz']):+.3f}",
                        f"{np.rad2deg(kin.heading[-1]):.1f}",
                    ],
                }
            ),
            hide_index=True,
        )
    with c2:
        st.markdown("##### Delivery pauses")
        if kin.gaps:
            st.dataframe(pd.DataFrame(kin.gaps, columns=["At [s]", "Pause [s]"]).round(3), hide_index=True)
        else:
            st.caption("No delivery pauses detected.")

# --------------------------------------------------------------------------- #
# Slip events
# --------------------------------------------------------------------------- #
with tab_ev:
    if res.events.empty:
        st.info(
            "No slip predicted with the current parameters. The 'Most critical moments' table in the Overview tab "
            "lists where the margin is smallest; the Friction sensitivity tab shows at which μ slip would start."
        )
    else:
        st.dataframe(events_view, hide_index=True)
        labels = [
            f"#{n} · {c} · {d:.3f} s · {pv}"
            for n, c, d, pv in zip(
                res.events["Event"], res.events["Start clock"], res.events["Duration [s]"], res.events["Pivot"]
            )
        ]
        sel = st.selectbox("Event detail", labels)
        ev = res.events.iloc[labels.index(sel)]
        pad_s = max(1.0, 2.0 * float(ev["Duration [s]"]))
        e0 = max(float(kin.t[0]), float(ev["Start [s]"]) - pad_s)
        e1 = min(float(kin.t[-1]), float(ev["End [s]"]) + pad_s)
        st.plotly_chart(fig_friction(res, e0, e1), config=PLOT_CONFIG, key="ev_friction")
        c1, c2 = st.columns([3, 2])
        with c1:
            st.plotly_chart(fig_motion(res, tol_mm, e0, e1), config=PLOT_CONFIG, key="ev_motion")
        with c2:
            st.plotly_chart(fig_corner_paths(res, e0, e1), config=PLOT_CONFIG, key="ev_corners")

    with st.expander("Video synchronisation", expanded=False):
        st.caption(
            "Load the video of the run and enter the recording clock time that corresponds to the first video frame. "
            "The player then jumps to the selected event or critical moment."
        )
        vid = st.file_uploader("Video file", type=["mp4", "mov", "webm", "m4v"], key="video")
        c1, c2 = st.columns(2)
        v0_txt = c1.text_input("Recording clock at video 0:00 (HH:MM:SS.sss)", value=fmt_clock(raw.df["clock"].iloc[0]))
        preroll = c2.number_input("Pre-roll [s]", value=3.0, min_value=0.0, max_value=60.0, step=1.0)
        moments = []
        if not res.events.empty:
            for n, c, t_ in zip(res.events["Event"], res.events["Start clock"], res.events["Start [s]"]):
                moments.append((f"Slip event #{n} · {c}", float(t_)))
        for c, t_, m_ in zip(crit["Clock"], crit["Time [s]"], crit["Required mu"]):
            moments.append((f"Critical moment · {c} · μ {m_:.3f}", float(t_)))
        if moments:
            choice = st.selectbox("Jump to", [m[0] for m in moments])
            t_sel = dict(moments)[choice]
            try:
                day = pd.Timestamp(raw.df["clock"].iloc[0]).normalize()
                v0 = pd.Timestamp(f"{day.date()} {v0_txt.strip()}")
                target = pd.Timestamp(raw.df["clock"].iloc[0]) + pd.to_timedelta(t_sel, unit="s")
                start_s = (target - v0).total_seconds() - preroll
                st.caption(f"Video position: {max(start_s, 0):.1f} s")
                if vid is not None:
                    st.video(vid, start_time=int(max(0, np.floor(start_s))))
            except Exception:  # noqa: BLE001
                st.error("Clock time not understood. Use the format HH:MM:SS or HH:MM:SS.sss.")

# --------------------------------------------------------------------------- #
# Friction sensitivity
# --------------------------------------------------------------------------- #
with tab_sens:
    st.caption(
        "Repeats the simulation for a range of static friction coefficients (same record and settings). "
        "Use it to judge how sensitive the result is to surface condition, contamination or material changes."
    )
    c1, c2, c3, c4 = st.columns(4)
    mu_lo = c1.number_input("μs from", value=0.10, min_value=0.01, max_value=2.0, step=0.01, format="%.2f")
    mu_hi = c2.number_input("μs to", value=0.40, min_value=0.02, max_value=2.0, step=0.01, format="%.2f")
    n_mu = c3.number_input("Steps", value=7, min_value=2, max_value=25, step=1)
    keep_ratio = c4.toggle("Keep μk/μs ratio", value=True)
    if st.button("Run sensitivity analysis", type="primary"):
        if mu_hi <= mu_lo:
            st.error("The upper μ must be larger than the lower μ.")
        else:
            mus = tuple(float(round(v, 4)) for v in np.linspace(mu_lo, mu_hi, int(n_mu)))
            with st.spinner(f"Running {len(mus)} simulations..."):
                st.session_state["sweep"] = {
                    "key": (active, json.dumps([pre, cluster_d, contact_d, sensor_d, solver_d], default=str)),
                    "df": run_sweep(data, active, pre, cluster_d, contact_d, sensor_d, solver_d, mus, keep_ratio),
                }
    sw_state = st.session_state.get("sweep")
    if sw_state is not None:
        if sw_state["key"] != (active, json.dumps([pre, cluster_d, contact_d, sensor_d, solver_d], default=str)):
            st.info("Parameters changed since the last run. Re-run the sensitivity analysis to update.")
        sw = sw_state["df"]
        f1, f2 = fig_sweep(sw, mu_s, sm["mu_noslip"], tol_mm)
        c1, c2 = st.columns(2)
        c1.plotly_chart(f1, config=PLOT_CONFIG, key="sw1")
        c2.plotly_chart(f2, config=PLOT_CONFIG, key="sw2")
        st.dataframe(sw.round(4), hide_index=True)

# --------------------------------------------------------------------------- #
# Compare recordings
# --------------------------------------------------------------------------- #
with tab_cmp:
    if len(valid) < 2:
        st.info("Upload two or more recordings to compare them (e.g. before / after a conveyor adjustment).")
    else:
        st.caption(
            "Every recording is simulated over its full length with the current mounting, cluster and friction settings."
        )
        rows = []
        with st.spinner("Simulating all recordings..."):
            for name, d in valid.items():
                pre_full = dict(pre, t_start=None, t_end=None)
                try:
                    kin_i = load_kinematics(d, name, pre_full)
                    res_i = run_simulation(d, name, pre_full, cluster_d, contact_d, sensor_d, solver_d)
                except Exception as exc:  # noqa: BLE001
                    st.warning(f"{name}: {exc}")
                    continue
                s_i = summary(res_i, kin_i)
                n_turn_i = sum(r.kind == "Turntable" for r in detect_rotations(kin_i, turn_s))
                rows.append(
                    {
                        "Recording": name,
                        "Start": fmt_clock(kin_i.clock[0]),
                        "Duration [s]": round(kin_i.duration, 1),
                        "Native rate [Hz]": round(kin_i.fs_native, 1),
                        "Required μ (no slip)": round(s_i["mu_noslip"], 3),
                        "Margin vs μs": f"{(s_i['margin'] - 1) * 100:+.0f}%",
                        "Slip events": s_i["n_events"],
                        "Slip time [s]": round(s_i["slip_time_s"], 2),
                        "Final max corner shift [mm]": round(s_i["final_corner_mm"], 2),
                        "Final rotation [deg]": round(s_i["final_rot_deg"], 3),
                        "Peak horiz. accel. [g]": round(s_i["peak_fh_g"], 3),
                        "Min. vertical [g]": round(s_i["min_fz_g"], 3),
                        "Peak yaw accel. [deg/s2]": round(s_i["peak_yaw_acc_deg_s2"], 1),
                        "Turntable rotations": n_turn_i,
                    }
                )
        if rows:
            cmp_df = pd.DataFrame(rows)
            st.plotly_chart(fig_compare(cmp_df, mu_s), config=PLOT_CONFIG, key="cmp")
            st.dataframe(cmp_df, hide_index=True)
            st.download_button(
                "Download comparison (CSV)", cmp_df.to_csv(index=False).encode(), "comparison.csv", "text/csv"
            )

# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
with tab_exp:
    stem = active.rsplit(".", 1)[0]
    params = {
        "app_version": APP_VERSION,
        "recording": active,
        "preprocessing": pre,
        "cluster": cluster_d,
        "contact": contact_d,
        "sensor": sensor_d,
        "solver": solver_d,
        "allowed_corner_displacement_mm": tol_mm,
        "turntable_detection": asdict(turn_s),
        "travel_direction": asdict(travel_s),
        "travel_direction_overrides": travel_overrides,
        "summary": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in sm.items()},
    }
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Simulation time series (CSV)",
            results_table(res).to_csv(index=False).encode(),
            f"{stem}_simulation.csv",
            "text/csv",
        )
        st.download_button(
            "Slip events (CSV)",
            events_view.to_csv(index=False).encode(),
            f"{stem}_slip_events.csv",
            "text/csv",
            disabled=res.events.empty,
        )
        st.download_button(
            "Critical moments (CSV)", crit.to_csv(index=False).encode(), f"{stem}_critical_moments.csv", "text/csv"
        )
        st.download_button(
            "Turntable rotations (CSV)",
            turn_tab.to_csv(index=False).encode(),
            f"{stem}_turntables.csv",
            "text/csv",
            disabled=turn_tab.empty,
        )
        st.download_button(
            "Speed profile per segment (CSV)",
            speed_tab.to_csv(index=False).encode(),
            f"{stem}_speed_segments.csv",
            "text/csv",
            disabled=speed_tab.empty,
        )
        st.download_button(
            "Motion segments / travel direction (CSV)",
            segments_table(motion_segs, travel_overrides).to_csv(index=False).encode(),
            f"{stem}_motion_segments.csv",
            "text/csv",
            disabled=not motion_segs,
        )
    with c2:
        st.download_button(
            "Carrier kinematics (CSV)",
            kinematics_table(kin).to_csv(index=False).encode(),
            f"{stem}_kinematics.csv",
            "text/csv",
        )
        st.download_button(
            "Parameters and summary (JSON)",
            json.dumps(params, indent=2, default=str).encode(),
            f"{stem}_parameters.json",
            "application/json",
        )
    st.caption("Time series are exported at the simulation sample rate. Distances in mm, angles in degrees.")

# --------------------------------------------------------------------------- #
# Method
# --------------------------------------------------------------------------- #
with tab_meth:
    I_used = ClusterParams(**cluster_d).yaw_inertia()
    patch = res.patch
    st.markdown(
        f"""
#### Physical model

**Carrier motion.** The IMU on the workpiece carrier measures the *specific force* **f** (acceleration minus
gravity, in g) and the yaw rate ω. The yaw acceleration α is obtained by differentiating ω. For a sensor
parallel to the GOT support surface, the specific force is exactly what loads the friction contact, so gravity
does not need to be removed: a tilted carrier and a horizontal acceleration act in the same way.

**Transfer to the cluster.** The carrier is rigid, so the specific force at the cluster centre of mass P is

&nbsp;&nbsp;&nbsp;&nbsp;**f**ₚ = **f**ₛ + α ẑ × **r** − ω² **r**,  with **r** = P − S (sensor → CoM).

This adds the tangential and centripetal terms that appear on turntables when the sensor is not at the cluster centre.

**Friction demand.** To follow the carrier, the cluster needs a friction force **F** = m g₀ **f**ₚ and a friction
moment M = I α (I = {I_used:.1f} kg·m²). The normal load is N = m g₀ f_z, so vertical bumps
(f_z < 1 g) momentarily reduce the friction capacity.

**Contact and limit surface.** The base of the cluster is represented by {len(patch.w)} contact patches with
normal-load fractions wᵢ (layout: *{"full-area" if res.contact.model == "full" else "four corner supports"}*).
The set of friction wrenches (Fx, Fy, M) the contact can transmit is a convex *limit surface*. The app evaluates
its gauge exactly through the support function, which gives the **required friction coefficient**
μ_req = gauge(**F**, M) / N. Pure translation needs μ_req = |**F**|/N; pure rotation about the CoM needs
|M| / (N · r̄) with r̄ = Σ wᵢ |ρᵢ| = {patch.r_m:.3f} m.

**Stick–slip.** The cluster sticks while μ_req ≤ μs. When μ_req exceeds μs, the relative twist
(vx, vy, Ω) is integrated with a linearly implicit scheme; every patch receives a kinetic friction force
μk N wᵢ opposite to its own sliding velocity. Rotation and the pivot location (instantaneous centre of
rotation) therefore emerge from the contact mechanics: an uneven load distribution makes the cluster pivot
about the most loaded corner. Coriolis and centrifugal terms of the rotating carrier frame are included.
The cluster re-sticks when its relative velocity vanishes and μ_req ≤ μs again.

**Why a cluster pivots about a corner.** With a uniform contact the inertial load acts through the friction
centroid, so a linear acceleration produces almost pure sliding. Rotation appears when (a) the carrier
rotates (turntables, yaw acceleration), (b) the load is concentrated on some supports, or (c) the friction is
not uniform — a region with higher friction stays put and the rest of the cluster swings around it. Options
(b) and (c) can be set in the sidebar to reproduce the observed behaviour.

#### Turntable detection

The gyroscope measures the carrier yaw rate directly. A rotation starts and ends where the (median-filtered)
yaw rate crosses the rotation threshold; same-direction rotations separated by less than the merge gap are
one turn, so a fast phase followed by a slow positioning phase counts as a single turn. The angle is the
integral of the yaw rate. Each turn is split into an **entry** window (carrier stopping on the table), the
**rotation** and an **exit** window (carrier leaving), and the friction demand on the cluster is evaluated in
each phase.

**Axis position.** For a carrier rotating about a fixed axis O the sensor S measures
**a**ₛ = α ẑ × **r** − ω² **r** with **r** = S − O. A least-squares fit over the rotation samples gives **r**.
The fit relies mainly on the spin-up and braking (α) and is therefore limited by the sample rate.

#### Travel direction

The inertial load alone does not show where the carrier is going: it points against the acceleration, so it
vanishes at constant speed. The carrier is considered to be rolling when the vertical vibration exceeds a
threshold. For each rolling period the velocity is integrated only over a short window after the start
(from rest) and before the stop (back to rest), which avoids the drift caused by conveyor slopes and sensor
offsets. When both windows give the same direction the confidence is *High*. The direction is snapped to the
carrier axes (±X, ±Y) and can be overridden manually. The **Motion phase** column then tells whether a load
occurred while the carrier was accelerating, braking / stopping, or as a lateral load.

#### Dynamic friction limit

The friction force available is μs·N and the normal load N = m·g₀·f_z follows the vertical acceleration. The
largest horizontal acceleration the cluster can take is therefore μs·f_z (in g): at rest it equals μs, during a
bump to 0.7 g it is 30 % lower. The column *Z effect on demand* in the critical moments shows how much the
vertical acceleration raised the required μ at that instant. The **kinetic** friction coefficient μk cannot be
derived from the carrier sensor: it only acts while the cluster slides. Measure it with a pull test (steady
force while sliding ÷ weight) or with a second sensor on the cluster.

#### Speed profile

For every motion segment the acceleration along the travel direction is integrated from rest to rest. A constant
offset is removed so that the speed returns to zero at the stop (zero-velocity update). This compensates the DC
offset that MEMS accelerometers show under vibration and slight conveyor slopes. Segments longer than about a
minute without a stop accumulate error and are flagged *Fair*; segments where the speed turns clearly negative
are flagged *Poor*.

#### Assumptions and limits

* The GOT is rigid and fixed to the carrier (aligned by pins); only the cluster moves.
* The cluster is rigid and stays in contact (no tipping). Contact is lost only if f_z ≤ 0.
* Coulomb friction with constant μs / μk; no adhesive, sealant or geometric stops are modelled.
* Only planar motion (X, Y, yaw) is simulated. Roll / pitch rates are not used.
* The sample rate of the recording limits what can be resolved. At about 10 Hz, impacts at stoppers or
  transfers lasting 10–50 ms are under-sampled, so peaks and slip are likely underestimated.
  Interpolation to the simulation rate does not recover lost information.
* The integrated heading (plant view) drifts slowly; it is used for visualisation only.

#### Validation (built-in physics checks)

* Linear step above the friction limit: displacement within 2 % of the closed-form solution.
* Pure yaw acceleration: the rotation threshold matches μ N r̄ / I and the rotation within 2 %.
* Limit surface: wrenches generated by arbitrary sliding twists lie on the evaluated limit surface (error < 0.5 %).
"""
    )
