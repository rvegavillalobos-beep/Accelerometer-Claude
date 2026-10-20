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
from contextlib import contextmanager
import dataclasses
from dataclasses import asdict
from pathlib import Path

import importlib
import os
import pickle
import sys
import threading
import time
import uuid

import numpy as np
import pandas as pd
import streamlit as st

# Local modules in dependency order. After a git push, Streamlit Cloud can rerun the new app.py while older
# versions of these modules are still loaded in the Python process (ImportError on new names). Every module
# stores the mtime of its source file when it is loaded; if any file on disk is newer, all are reloaded.
_LOCAL_MODULES = (
    "data_io",
    "slip_model",
    "turns",
    "motion",
    "line_layout_data",
    "line_layout",
    "whole_route",
    "visuals",
)


def _local_modules_stale() -> bool:
    for _name in _LOCAL_MODULES:
        _mod = sys.modules.get(_name)
        _file = getattr(_mod, "__file__", None) if _mod is not None else None
        if not _file:
            continue
        try:
            if getattr(_mod, "_LOADED_MTIME", None) != os.path.getmtime(_file):
                return True
        except OSError:
            continue
    return False


def _reload_stale_local_modules() -> None:
    if not _local_modules_stale():
        return
    # One process serves every open browser session: only one of them reloads, the others wait for it.
    lock = sys.__dict__.setdefault("_got_slip_reload_lock", threading.Lock())
    with lock:
        if _local_modules_stale():
            for _name in _LOCAL_MODULES:
                if _name in sys.modules:
                    importlib.reload(sys.modules[_name])


_reload_stale_local_modules()

from data_io import (
    PreprocessSettings,
    RawRecording,
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
    exceedance_peaks,
    friction_sweep,
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
from line_layout import background_uri as layout_background_uri
from line_layout import flow_arrows as layout_arrows
from line_layout import image_size as layout_image_size
from line_layout import known_stations, route_order, section_geometry, section_keys, section_label, sections_table
from line_layout import station_display
from line_layout import network as layout_network
from line_layout import stations_table as layout_stations
from whole_route import accumulate, effective_mu, totals
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
    friction_demand_g,
    fig_friction,
    fig_kinematics,
    fig_line_map,
    fig_route_progress,
    fig_route_sweep,
    fig_motion,
    fig_sweep,
    dynamic_limit_g,
    fig_dynamic_limit,
    fig_measure_detail,
    fig_measure_overview,
    fig_rotation_timeline,
    fig_turn_compare,
    fig_turn_detail,
    fig_speed_ramps,
    fig_speed_timeline,
    fig_topview,
    fig_turn_profiles,
)

APP_VERSION = "1.4.0"

# Cached results are keyed on a fingerprint of the calculation modules, so a code update never reuses
# objects computed by an older version of the app (Streamlit Cloud keeps its cache across redeploys).
_CALC_MODULES = ("data_io.py", "slip_model.py", "turns.py", "motion.py")
_CACHE_FORMAT = b"plain-records-v2"  # bump when the layout of cached values changes
CACHE_V = hashlib.md5(
    _CACHE_FORMAT
    + b"".join(Path(__file__).with_name(f).read_bytes() for f in _CALC_MODULES if Path(__file__).with_name(f).exists())
).hexdigest()[:12]

st.set_page_config(page_title="GOT Cluster Slip Simulator", page_icon="🔋", layout="wide")


# --------------------------------------------------------------------------- #
# Cached computation layer
# --------------------------------------------------------------------------- #
# Cache sizes are bounded and entries expire, so memory does not grow without limit on Streamlit Cloud
# (a container that runs out of memory is restarted and every open browser loses its connection).
#
# st.cache_data pickles what it stores, and pickle refers to classes by module and name. After a redeploy the
# app's own modules are loaded again, so a result built with the previous class object can no longer be
# pickled ("it's not the same object as slip_model.ClusterParams"). Instances of the app's own classes are
# therefore stored as plain records (module, class name, attribute values) and rebuilt with the classes that
# are loaded when they are read. Arrays and tables are stored unchanged.
_PLAIN_TAG = "__got_local_object__"


def _to_plain(obj):
    t = type(obj)
    if t.__module__ in _LOCAL_MODULES and hasattr(obj, "__dict__"):
        return {_PLAIN_TAG: (t.__module__, t.__qualname__), "state": {k: _to_plain(v) for k, v in vars(obj).items()}}
    if t is list:
        return [_to_plain(v) for v in obj]
    if t is tuple:
        return tuple(_to_plain(v) for v in obj)
    if t is dict:
        return {k: _to_plain(v) for k, v in obj.items()}
    return obj


def _from_plain(obj):
    t = type(obj)
    if t is dict:
        if _PLAIN_TAG in obj:
            mod_name, qualname = obj[_PLAIN_TAG]
            cls = getattr(sys.modules.get(mod_name) or importlib.import_module(mod_name), qualname)
            inst = cls.__new__(cls)
            state = {k: _from_plain(v) for k, v in obj["state"].items()}
            for f in getattr(cls, "__dataclass_fields__", {}).values():  # fields added by a newer version
                if f.name not in state:
                    if f.default is not dataclasses.MISSING:
                        state[f.name] = f.default
                    elif f.default_factory is not dataclasses.MISSING:
                        state[f.name] = f.default_factory()
            for k, v in state.items():
                object.__setattr__(inst, k, v)  # also works for frozen dataclasses
            return inst
        return {k: _from_plain(v) for k, v in obj.items()}
    if t is list:
        return [_from_plain(v) for v in obj]
    if t is tuple:
        return tuple(_from_plain(v) for v in obj)
    return obj


_CACHE_ERROR_NAMES = {
    "CacheError",
    "CacheKeyNotFoundError",
    "UnserializableReturnValueError",
    "UnhashableParamError",
    "UnhashableTypeError",
}


def _is_cache_failure(exc: BaseException) -> bool:
    e, depth = exc, 0
    while e is not None and depth < 6:
        if isinstance(e, (pickle.PicklingError, pickle.UnpicklingError)):
            return True
        if any(c.__name__ in _CACHE_ERROR_NAMES for c in type(e).__mro__):
            return True
        e, depth = e.__cause__ or e.__context__, depth + 1
    return False


def _cached_call(cached_fn, direct_fn, *args):
    """Read through the cache; if the cache itself fails, compute directly so the page still works."""
    try:
        out = cached_fn(CACHE_V, *args)
    except Exception as exc:  # noqa: BLE001
        if not _is_cache_failure(exc):
            raise
        out = direct_fn(*args)
    return _from_plain(out)


def _compute_raw(data: bytes, name: str):
    return _to_plain(read_witmotion(data, name))


@st.cache_data(show_spinner=False, max_entries=64, ttl=6 * 3600)
def _load_raw_cached(cache_v: str, data: bytes, name: str):
    return _compute_raw(data, name)


def load_raw(data: bytes, name: str) -> RawRecording:
    return _cached_call(_load_raw_cached, _compute_raw, data, name)


def _compute_kinematics(data: bytes, name: str, pre: dict):
    raw = load_raw(data, name)
    return _to_plain(preprocess(raw, PreprocessSettings(**pre)))


@st.cache_data(show_spinner=False, max_entries=16, ttl=2 * 3600)
def _load_kinematics_cached(cache_v: str, data: bytes, name: str, pre: dict):
    return _compute_kinematics(data, name, pre)


def load_kinematics(data: bytes, name: str, pre: dict):
    return _cached_call(_load_kinematics_cached, _compute_kinematics, data, name, pre)


def _compute_simulation(data, name, pre, cluster, contact, sensor, solver):
    kin = load_kinematics(data, name, pre)
    return _to_plain(
        simulate(
            kin, ClusterParams(**cluster), ContactParams(**contact), SensorGeometry(**sensor), SolverParams(**solver)
        )
    )


@st.cache_data(show_spinner=False, max_entries=8, ttl=2 * 3600)
def _run_simulation_cached(
    cache_v: str, data: bytes, name: str, pre: dict, cluster: dict, contact: dict, sensor: dict, solver: dict
):
    return _compute_simulation(data, name, pre, cluster, contact, sensor, solver)


def run_simulation(
    data: bytes, name: str, pre: dict, cluster: dict, contact: dict, sensor: dict, solver: dict
) -> SimulationResult:
    return _cached_call(_run_simulation_cached, _compute_simulation, data, name, pre, cluster, contact, sensor, solver)


def _compute_sweep(data, name, pre, cluster, contact, sensor, solver, mus, keep_ratio) -> pd.DataFrame:
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


@st.cache_data(show_spinner=False, max_entries=8, ttl=2 * 3600)
def _run_sweep_cached(
    cache_v: str,
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
    return _compute_sweep(data, name, pre, cluster, contact, sensor, solver, mus, keep_ratio)


def run_sweep(data, name, pre, cluster, contact, sensor, solver, mus, keep_ratio) -> pd.DataFrame:
    return _cached_call(
        _run_sweep_cached, _compute_sweep, data, name, pre, cluster, contact, sensor, solver, mus, keep_ratio
    )


@contextmanager
def tab_guard(tab_name: str):
    """Keep an error in one tab from stopping the rest of the app (st.stop / st.rerun pass through)."""
    try:
        yield
    except Exception as exc:  # noqa: BLE001
        st.error(f"The {tab_name} tab could not be displayed: {type(exc).__name__}: {exc}")
        with st.expander("Technical details"):
            st.exception(exc)


def fmt_clock(ts) -> str:
    return pd.Timestamp(ts).strftime("%H:%M:%S.%f")[:-3]


def _corners_rel_com(cluster: dict, res_i) -> list:
    """Corner positions relative to the cluster CoM [m] (to add up shifts over several recordings)."""
    return (ClusterParams(**cluster).corners() - np.asarray(res_i.patch.com, dtype=float)).tolist()


def _compute_compare_row(data, name, pre, cluster, contact, sensor, solver, turn_d, mu_s):
    """Summary of one recording for the Compare, Line map and whole-route views. Only small results are cached (a few
    kB), not the full simulation, so many recordings can be handled without filling the memory or recomputing on
    every click. Returns (row, exceedance peaks, extra) with extra = final shift, slip events and corner geometry."""
    kin_i = preprocess(load_raw(data, name), PreprocessSettings(**pre))
    res_i = simulate(
        kin_i, ClusterParams(**cluster), ContactParams(**contact), SensorGeometry(**sensor), SolverParams(**solver)
    )
    s_i = summary(res_i, kin_i)
    turn_s_i = TurnSettings(**turn_d)
    rots_i = detect_rotations(kin_i, turn_s_i)
    n_turn_i = sum(r.kind == "Turntable" for r in rots_i)
    pk_i = exceedance_peaks(res_i, kin_i, mu_s)
    row = {
        "Recording": name,
        "Start": fmt_clock(kin_i.clock[0]),
        "Duration [s]": round(kin_i.duration, 1),
        "Native rate [Hz]": round(kin_i.fs_native, 1),
        "Required μ (no slip)": round(s_i["mu_noslip"], 3),
        "Margin vs μs": f"{(s_i['margin'] - 1) * 100:+.0f}%",
        "Exceedances of μs": len(pk_i),
        "Exceedance times": ", ".join(pk_i["Clock"].tolist()[:12]) + (" …" if len(pk_i) > 12 else ""),
        "Slip events": s_i["n_events"],
        "Slip time [s]": round(s_i["slip_time_s"], 2),
        "Final max corner shift [mm]": round(s_i["final_corner_mm"], 2),
        "Final rotation [deg]": round(s_i["final_rot_deg"], 3),
        "Peak horiz. accel. [g]": round(s_i["peak_fh_g"], 3),
        "Min. vertical [g]": round(s_i["min_fz_g"], 3),
        "Peak yaw accel. [deg/s2]": round(s_i["peak_yaw_acc_deg_s2"], 1),
        "Turntable rotations": n_turn_i,
    }
    events = res_i.events.copy()
    if len(events):
        events.insert(
            events.columns.get_loc("Carrier motion") + 1,
            "Conveyor context",
            [locate(t_, rots_i, turn_s_i.pad_s) for t_ in events["Start [s]"]],
        )
        events = events.drop(columns=["Carrier motion"])
    extra = {
        "dx_mm": s_i["final_dx_mm"],
        "dy_mm": s_i["final_dy_mm"],
        "rot_deg": s_i["final_rot_deg"],
        "corner_mm": s_i["final_corner_mm"],
        "n_events": int(s_i["n_events"]),
        "slip_time_s": float(s_i["slip_time_s"]),
        "mu_noslip": float(s_i["mu_noslip"]),
        "rel": _corners_rel_com(cluster, res_i),
        "events": events,
    }
    return row, pk_i, extra


@st.cache_data(show_spinner=False, max_entries=400, ttl=6 * 3600)
def _compare_row_cached(
    cache_v: str,
    data: bytes,
    name: str,
    pre: dict,
    cluster: dict,
    contact: dict,
    sensor: dict,
    solver: dict,
    turn_d: dict,
    mu_s: float,
):
    return _compute_compare_row(data, name, pre, cluster, contact, sensor, solver, turn_d, mu_s)


def compare_row(data, name, pre, cluster, contact, sensor, solver, turn_d, mu_s) -> tuple[dict, pd.DataFrame, dict]:
    return _cached_call(
        _compare_row_cached, _compute_compare_row, data, name, pre, cluster, contact, sensor, solver, turn_d, float(mu_s)
    )


def _compute_route_point(data, name, pre, cluster, contact, sensor, solver) -> dict:
    """Final shift of one recording for one friction setting (whole-route friction sensitivity)."""
    kin_i = preprocess(load_raw(data, name), PreprocessSettings(**pre))
    res_i = simulate(
        kin_i,
        ClusterParams(**cluster),
        ContactParams(**contact),
        SensorGeometry(**sensor),
        SolverParams(**solver),
        max_slip_seconds=120.0,
    )
    s_i = summary(res_i, kin_i)
    return {
        "dx_mm": s_i["final_dx_mm"],
        "dy_mm": s_i["final_dy_mm"],
        "rot_deg": s_i["final_rot_deg"],
        "corner_mm": s_i["final_corner_mm"],
        "n_events": int(s_i["n_events"]),
        "slip_time_s": float(s_i["slip_time_s"]),
        "mu_noslip": float(s_i["mu_noslip"]),
        "truncated": bool(res_i.truncated),
        "rel": _corners_rel_com(cluster, res_i),
    }


@st.cache_data(show_spinner=False, max_entries=4000, ttl=6 * 3600)
def _route_point_cached(cache_v, data, name, pre, cluster, contact, sensor, solver) -> dict:
    return _compute_route_point(data, name, pre, cluster, contact, sensor, solver)


def route_point(data, name, pre, cluster, contact, sensor, solver) -> dict:
    return _cached_call(_route_point_cached, _compute_route_point, data, name, pre, cluster, contact, sensor, solver)


# --------------------------------------------------------------------------- #
# Uploaded files survive a page reload
# --------------------------------------------------------------------------- #
# A browser that loses its connection (idle tab, sleep, network change) gets a new, empty session when the page
# is reloaded. The uploaded files are therefore also kept in server memory under a random id that lives in the
# page URL (?ws=...), and restored after a reload. Entries expire after 12 h and the total size is capped.
_WS_TTL_S = 12 * 3600
_WS_MAX_BYTES = 300_000_000


@st.cache_resource
def _upload_store() -> dict:
    return {}


def _store_prune(store: dict, keep: str | None = None) -> None:
    now = time.time()
    for k in [k for k, v in list(store.items()) if now - v["t"] > _WS_TTL_S]:
        store.pop(k, None)
    total = sum(sum(len(b) for b in v["files"].values()) for v in store.values())
    for k, _v in sorted(store.items(), key=lambda kv: kv[1]["t"]):
        if total <= _WS_MAX_BYTES:
            break
        if k != keep:
            total -= sum(len(b) for b in store[k]["files"].values())
            store.pop(k, None)


def _clear_restored_files(ws: str) -> None:
    _upload_store().pop(ws, None)
    st.session_state["_restored_base"] = {}


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

ws_id = st.query_params.get("ws")
if not ws_id:
    ws_id = uuid.uuid4().hex[:16]
    st.query_params["ws"] = ws_id
_store = _upload_store()
_store_prune(_store, keep=ws_id)

files: dict[str, bytes] = {}
if "_restored_base" not in st.session_state:
    # new browser session (first visit or page reload): restore the files kept for this URL, if any
    _entry = _store.get(ws_id)
    st.session_state["_restored_base"] = dict(_entry["files"]) if _entry else {}
files.update(st.session_state["_restored_base"])
for f in uploads or []:
    name = f.name
    i = 2
    while name in files and name not in st.session_state["_restored_base"]:
        name = f"{f.name} ({i})"
        i += 1
    files[name] = f.getvalue()
if uploads or files:
    _store[ws_id] = {"files": dict(files), "t": time.time()}
if st.session_state.get("_had_uploads") and not uploads and not st.session_state["_restored_base"]:
    _store.pop(ws_id, None)  # the user removed every uploaded file
    files = {}
st.session_state["_had_uploads"] = bool(uploads)

if st.session_state["_restored_base"]:
    with st.sidebar:
        st.info(
            f"{len(st.session_state['_restored_base'])} file(s) restored after the page was reloaded. "
            "Upload more files to add them, or clear the restored ones."
        )
        st.button("Clear restored files", on_click=_clear_restored_files, args=(ws_id,))

if not files:
    landing()
    st.stop()

# Validate files
valid: dict[str, bytes] = {}
rec_start: dict[str, pd.Timestamp] = {}
rec_dur: dict[str, float] = {}
for name, data in files.items():
    try:
        _raw = load_raw(data, name)
        valid[name] = data
        rec_start[name] = pd.Timestamp(_raw.df["clock"].iloc[0])
        rec_dur[name] = recording_duration(_raw)
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error(f"{name}: {exc}")
if not valid:
    st.error("None of the uploaded files could be read as a WitMotion export.")
    st.stop()

# oldest -> newest, using the first timestamp inside each file
valid = dict(sorted(valid.items(), key=lambda kv: (rec_start[kv[0]], kv[0])))
_multi_day = len({ts.date() for ts in rec_start.values()}) > 1


def _recording_label(name: str) -> str:
    ts = rec_start[name]
    when = ts.strftime("%Y-%m-%d %H:%M:%S") if _multi_day else ts.strftime("%H:%M:%S")
    return f"{name} · {when} · {rec_dur[name] / 60:.1f} min"


with st.sidebar:
    active = st.selectbox(
        "Active recording",
        list(valid.keys()),
        format_func=_recording_label,
        help="Sorted by the first timestamp inside each file (oldest first).",
    )
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
        st.markdown("**Spike filter** (transmission glitches)")
        spike_mode_lbl = st.selectbox(
            "Remove isolated spikes",
            ["Auto (from 50 Hz)", "On", "Off"],
            index=0,
            help="A glitch is a sample that jumps far away from its neighbours and comes straight back (typically one "
            "sample, up to several g). It is replaced by interpolation. Real impacts last several samples at "
            "high sample rates and are kept. Auto: on when the native rate is 50 Hz or more; at 10 Hz a real "
            "impact can be a single sample.",
        )
        spike_mode = {"Auto (from 50 Hz)": "auto", "On": "on", "Off": "off"}[spike_mode_lbl]
        sp1, sp2 = st.columns(2)
        spike_acc = sp1.number_input(
            "Jump [g]",
            value=0.5,
            min_value=0.1,
            max_value=10.0,
            step=0.1,
            disabled=spike_mode == "off",
            help="Acceleration sample this far from the median of its neighbours = glitch, if the neighbours sit at the "
            "normal level of the signal (around the peak of a real impact they are raised too, so it is kept).",
        )
        spike_gyro = sp2.number_input(
            "Jump [deg/s]",
            value=15.0,
            min_value=1.0,
            max_value=1000.0,
            step=5.0,
            disabled=spike_mode == "off",
            help="Same for the yaw rate.",
        )
        sp3, sp4 = st.columns(2)
        spike_width = sp3.select_slider(
            "Max. width [samples]",
            [1, 2, 3],
            value=1,
            disabled=spike_mode == "off",
            help="Wider events are kept as real. At 200 Hz one sample is 5 ms.",
        )
        spike_limit = sp4.number_input(
            "Hard limit [g]",
            value=0.0,
            min_value=0.0,
            max_value=20.0,
            step=0.5,
            disabled=spike_mode == "off",
            help="Optional (0 = off): any acceleration sample farther than this from its typical value (≈0 g "
            "horizontal, 1 g vertical) is removed, however long. Use it only above the largest real impact.",
        )
        spike_min_total = st.number_input(
            "Dropout: vertical below [g]",
            value=0.3,
            min_value=0.0,
            max_value=0.9,
            step=0.05,
            disabled=spike_mode == "off",
            help="A carrier cannot fall faster than gravity: the vertical acceleration stays near 1 g (real bumps "
            "reach about 0.7 g). A sample with a vertical reading below this value (zero, dropped or inverted data "
            "packet) is removed, however long. 0 = off.",
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
        spike_mode=spike_mode,
        spike_acc_g=float(spike_acc),
        spike_gyro_dps=float(spike_gyro),
        spike_max_width=int(spike_width),
        spike_limit_g=float(spike_limit),
        spike_min_vertical_g=float(spike_min_total),
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
travel_speed, a_long, speed_q, speed_tab = speed_profile(kin, res, motion_segs, travel_overrides)


# --------------------------------------------------------------------------- #
# Measure mode: drag across a time chart to measure an interval
# --------------------------------------------------------------------------- #
def _selected_range(event) -> tuple[float, float] | None:
    """x-range of a box selection returned by st.plotly_chart(on_select=...)."""
    if event is None:
        return None
    try:
        sel = event["selection"]
    except (KeyError, TypeError):
        sel = getattr(event, "selection", None)
    if not sel:
        return None
    xs: list[float] = []
    for box in sel.get("box") or []:
        x = box.get("x") if isinstance(box, dict) else None
        if x is not None and len(x) >= 2:
            xs.extend([float(x[0]), float(x[-1])])
    if not xs:
        for p in sel.get("points") or []:
            if isinstance(p, dict) and p.get("x") is not None:
                xs.append(float(p["x"]))
    if len(xs) < 2 or max(xs) - min(xs) <= 0:
        return None
    return min(xs), max(xs)


def _show_measurement(a: float, b: float):
    a, b = max(a, float(kin.t[0])), min(b, float(kin.t[-1]))
    i0 = int(np.searchsorted(kin.t, a))
    i1 = max(i0, int(np.searchsorted(kin.t, b, side="right")) - 1)
    sl = slice(i0, i1 + 1)
    dt = b - a
    sp = np.nan_to_num(travel_speed[sl], nan=0.0)
    dist = float(np.sum(np.clip(sp, 0.0, None)) / kin.fs)
    mu_i = res.mu_req[sl]
    mu_i = mu_i[np.isfinite(mu_i)]
    mu_max = float(mu_i.max()) if len(mu_i) else float("nan")
    d_head = float(np.rad2deg(kin.heading[i1] - kin.heading[i0]))
    d_corner = (
        float(
            np.max(
                np.hypot(
                    res.corner_xy[i1, :, 0] - res.corner_xy[i0, :, 0],
                    res.corner_xy[i1, :, 1] - res.corner_xy[i0, :, 1],
                )
            )
            * 1e3
        )
        if i1 > i0
        else 0.0
    )
    with st.container(border=True):
        st.markdown(
            f"**Measured interval:** {a:.2f} → {b:.2f} s  ·  {fmt_clock(kin.clock[i0])} → {fmt_clock(kin.clock[i1])}"
        )
        m = st.columns(6)
        m[0].metric("Δt", f"{dt:.3f} s")
        m[1].metric("Distance ≈", f"{dist:.2f} m", help="From the estimated travel speed (Speed profile tab).")
        m[2].metric("Mean speed ≈", f"{dist / dt:.2f} m/s" if dt > 0 else "-")
        m[3].metric("Carrier rotation", f"{d_head:+.1f}°")
        m[4].metric(
            "Max required μ",
            f"{mu_max:.3f}" if np.isfinite(mu_max) else "-",
            f"{(mu_s / mu_max - 1) * 100:+.0f}% margin" if np.isfinite(mu_max) and mu_max > 0 else None,
        )
        m[5].metric("Max horiz. accel.", f"{float(res.fh_com[sl].max()):.3f} g")
        st.caption(
            f"Slip predicted in the interval: {'yes' if res.sliding[sl].any() else 'no'} · "
            f"cluster corner shift in the interval: {d_corner:.2f} mm · "
            f"lowest vertical acceleration: {float(kin.fz[sl].min()):.3f} g · "
            f"from {locate(a, rotations, turn_s.pad_s)} to {locate(b, rotations, turn_s.pad_s)}."
        )


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
        "Measure",
        "Turntables",
        "Speed profile",
        "Top-view animation",
        "Signals",
        "Slip events",
        "Friction sensitivity",
        "Compare recordings",
        "Line map",
        "Method",
    ]
)
tab_over, tab_meas, tab_turn, tab_speed, tab_anim, tab_sig, tab_ev, tab_sens, tab_cmp, tab_map, tab_meth = tabs


def all_recording_rows() -> tuple[list[dict], dict[str, pd.DataFrame], list[str], dict[str, dict]]:
    """Summary, exceedances of μs and final shift of every uploaded recording, simulated over its full length
    (cached: the Compare, Line map, Slip events and Friction sensitivity tabs share the same results)."""
    rows, peaks, errors, extras = [], {}, [], {}
    pre_full = dict(pre, t_start=None, t_end=None)
    for name, d in valid.items():
        try:
            row_i, pk_i, ex_i = compare_row(
                d, name, pre_full, cluster_d, contact_d, sensor_d, solver_d, asdict(turn_s), mu_s
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {exc}")
            continue
        peaks[name] = pk_i
        rows.append(row_i)
        extras[name] = ex_i
    return rows, peaks, errors, extras


# --------------------------------------------------------------------------- #
# Whole route: all recordings in line order, as if one part had travelled them
# --------------------------------------------------------------------------- #
SCOPE_ACTIVE = "Active recording"
SCOPE_ROUTE = "All uploaded recordings (whole route)"
_SHIFT_KEYS = ("dx_mm", "dy_mm", "rot_deg", "corner_mm", "n_events", "slip_time_s")


def effective_sections(recordings: list[str]) -> pd.DataFrame:
    """Sections from the file names, with the corrections made in the Line map tab."""
    base = sections_table(recordings)
    key = "line_sections_" + hashlib.md5(base.to_json().encode()).hexdigest()[:10]
    state = st.session_state.get(key)
    edits = state.get("edited_rows") if isinstance(state, dict) else None
    for i, changes in (edits or {}).items():
        for col, val in (changes or {}).items():
            if col in base.columns and col != "Recording" and int(i) < len(base):
                base.at[int(i), col] = val
    return base.fillna("")


def route_rows(recordings: list[str]) -> list[tuple[str, pd.Series]]:
    """(recording, section row) in line order (ST 130 -> buffer); names without a known station come last."""
    sec = effective_sections(recordings)
    return [(r["Recording"], r) for r in sorted((r for _, r in sec.iterrows()), key=route_order)]


def route_exclusion_picker(where: str, names: list[str]) -> list[str]:
    """Recordings left out of the whole route; one choice shared by the Slip events and Friction sensitivity tabs."""
    key = f"route_out_{where}"
    st.session_state[key] = [n for n in st.session_state.get("route_out", []) if n in names]

    def _sync():
        st.session_state["route_out"] = list(st.session_state.get(key, []))

    return st.multiselect(
        "Leave out of the whole route",
        names,
        key=key,
        on_change=_sync,
        help="E.g. a repeated measurement of the same section, or a test. The choice applies to the Slip events and "
        "Friction sensitivity tabs.",
    )


def route_duplicates(incl: list[tuple[str, pd.Series]]) -> list[str]:
    groups: dict[tuple, list[str]] = {}
    for name, row in incl:
        a, b, via = section_keys(row)
        if not a:
            continue
        if b is None and not via:  # process inside a station: same station and same kind of process
            words = str(row.get("Note") or "").strip().lower().split()
            key = ("process", a, words[0] if words else "")
        else:
            key = ("transport", a, b, tuple(via))
        groups.setdefault(key, []).append(name)
    return [", ".join(v) for v in groups.values() if len(v) > 1]


def route_parts(incl: list[tuple[str, pd.Series]], values: dict[str, dict]) -> list[dict]:
    return [
        dict(name=n, label=section_label(r), **{k: values[n][k] for k in _SHIFT_KEYS}) for n, r in incl if n in values
    ]


def route_selection(where: str):
    """All recordings simulated (cached), put in line order; returns (included, extras) or None."""
    with st.spinner("Simulating all recordings..."):
        rows_r, _pk, errs, extras = all_recording_rows()
    for msg in errs:
        st.warning(msg)
    if not extras:
        st.info("None of the uploaded recordings could be simulated.")
        return None
    ordered = route_rows([r["Recording"] for r in rows_r])
    out = route_exclusion_picker(where, [n for n, _ in ordered])
    incl = [(n, r) for n, r in ordered if n not in out]
    if not incl:
        st.info("Every recording is left out.")
        return None
    for d in route_duplicates(incl):
        st.warning(
            f"Same section measured more than once: {d}. If these are repeated measurements, leave all but one out "
            "above, otherwise their shift is added up twice."
        )
    unknown = [n for n, r in incl if not section_keys(r)[0]]
    if unknown:
        st.caption(
            "Not placed on the line (no station in the file name), added at the end: " + ", ".join(unknown) + "."
        )
    return incl, extras


def render_route_slip_events() -> None:
    st.caption(
        "Every uploaded recording is simulated over its full length with the current settings and put in line order "
        "(ST 130 → buffer, from the file names), as if one part had travelled all the measured segments. The shifts "
        "of the segments add up; shifts in opposite directions cancel. Compare the shift at the end with what you "
        "measure on the part."
    )
    sel = route_selection("ev")
    if sel is None:
        return
    incl, extras = sel
    parts = route_parts(incl, extras)
    rel = extras[incl[0][0]]["rel"]
    prog = accumulate(parts, rel)
    tot = totals(parts, rel)
    c = st.columns(5)
    c[0].metric("Recordings", len(incl))
    c[1].metric("Slip events", tot["n_events"])
    c[2].metric("Slip time", f"{tot['slip_time_s']:.1f} s")
    c[3].metric(
        "Corner shift at the end",
        f"{tot['corner_mm']:.2f} mm",
        help=f"If every segment had shifted the cluster the same way: {tot['sum_segment_corner_mm']:.2f} mm.",
    )
    c[4].metric("Rotation at the end", f"{tot['rot_deg']:.3f}°")
    st.plotly_chart(fig_route_progress(prog, tol_mm, mu_s), config=PLOT_CONFIG, key="route_progress")
    st.caption(
        "Bars: shift caused by each segment on its own (red = it slips). Line: shift of the cluster accumulated "
        "from the start of the line up to that segment."
    )
    st.markdown("##### Slip events of all recordings")
    frames = []
    for n, r in incl:
        e = extras[n].get("events")
        if e is None or not len(e):
            continue
        e = e.copy()
        e.insert(0, "Section", section_label(r))
        e.insert(0, "Recording", n)
        frames.append(e)
    if frames:
        allev = pd.concat(frames, ignore_index=True)
        cols = [
            c_
            for c_ in [
                "Recording",
                "Section",
                "Event",
                "Start clock",
                "Duration [s]",
                "Conveyor context",
                "Driver",
                "Peak mu required",
                "dX [mm]",
                "dY [mm]",
                "dRotation [deg]",
                "Max corner shift [mm]",
                "Pivot",
            ]
            if c_ in allev.columns
        ]
        st.dataframe(allev[cols], hide_index=True)
        st.download_button(
            "Download slip events of all recordings (CSV)",
            allev[cols].to_csv(index=False).encode(),
            "slip_events_whole_route.csv",
            "text/csv",
        )
    else:
        st.info(f"No slip predicted on any of the recordings with μs = {mu_s:.3f}.")
    with st.expander("Shift per recording"):
        st.dataframe(prog, hide_index=True)


def render_route_sweep(mu_lo: float, mu_hi: float, n_mu: int, keep_ratio: bool) -> None:
    sel = route_selection("fs")
    if sel is None:
        return
    incl, extras = sel
    names = [n for n, _ in incl]
    c1, c2 = st.columns(2)
    meas_c = c1.number_input(
        "Measured at the end: max corner shift [mm]",
        value=0.0,
        min_value=0.0,
        step=0.1,
        format="%.2f",
        key="route_meas_corner",
        help="Shift of the cluster on the GOT measured on the part at the end of the line (0 = not measured). "
        "The μs at which the model predicts the same shift is the effective friction of the line.",
    )
    meas_r = c2.number_input(
        "Measured at the end: rotation [deg]",
        value=0.0,
        min_value=0.0,
        step=0.01,
        format="%.3f",
        key="route_meas_rot",
        help="Rotation of the cluster on the GOT measured at the end of the line (0 = not measured).",
    )
    mus = tuple(float(round(v, 4)) for v in np.linspace(mu_lo, mu_hi, int(n_mu)))
    run_key = json.dumps([names, pre, cluster_d, contact_d, sensor_d, solver_d, mus, keep_ratio], default=str)
    st.caption(
        f"{len(names)} recordings × {len(mus)} values of μs = {len(names) * len(mus)} simulations. Results are kept: "
        "running again, or adding a recording, only computes what is new."
    )
    if st.button("Run whole-route sensitivity", type="primary", disabled=mu_hi <= mu_lo):
        ratio = contact_d["mu_k"] / contact_d["mu_s"] if contact_d["mu_s"] > 0 else 1.0
        pre_full = dict(pre, t_start=None, t_end=None)
        bar = st.progress(0.0, text="Simulating...")
        per, rel, k = [], None, 0
        for mu in mus:
            c_ = dict(contact_d, mu_s=mu, mu_k=mu * ratio if keep_ratio else min(contact_d["mu_k"], mu))
            for n in names:
                p_ = route_point(valid[n], n, pre_full, cluster_d, c_, sensor_d, solver_d)
                rel = rel or p_["rel"]
                per.append({"Recording": n, "mu_s": mu, **{kk: p_[kk] for kk in _SHIFT_KEYS}, "truncated": p_["truncated"]})
                k += 1
                bar.progress(k / (len(mus) * len(names)), text=f"μs {mu:.3f} · {n}")
        bar.empty()
        st.session_state["route_sweep"] = {"key": run_key, "per": pd.DataFrame(per), "rel": rel, "names": names}
    state = st.session_state.get("route_sweep")
    if state is None:
        return
    if state["key"] != run_key:
        st.info("Settings or recordings changed since the last run. Run the whole-route sensitivity again to update.")
    per, rel = state["per"], state["rel"]
    tot_rows = []
    for mu, g in per.groupby("mu_s"):
        t_ = totals(g.to_dict("records"), rel)
        tot_rows.append(
            {
                "mu_s": mu,
                "Slip events": t_["n_events"],
                "Slip time [s]": round(t_["slip_time_s"], 2),
                "dX at the end [mm]": round(t_["dx_mm"], 2),
                "dY at the end [mm]": round(t_["dy_mm"], 2),
                "Rotation at the end [deg]": round(t_["rot_deg"], 3),
                "Shift at the end [mm]": round(t_["corner_mm"], 2),
                "Sum of segment shifts [mm]": round(t_["sum_segment_corner_mm"], 2),
                "Runs stopped (long slip)": int(g["truncated"].sum()),
            }
        )
    tot = pd.DataFrame(tot_rows).sort_values("mu_s").reset_index(drop=True)
    mu_noslip = max(extras[n]["mu_noslip"] for n in state["names"] if n in extras) if extras else float("nan")
    mu_eff_c, note_c = effective_mu(tot["mu_s"], tot["Shift at the end [mm]"], meas_c)
    mu_eff_r, note_r = effective_mu(tot["mu_s"], tot["Rotation at the end [deg]"].abs(), meas_r)
    f1, f2 = fig_route_sweep(tot, mu_s, mu_noslip, tol_mm, meas_c, meas_r, mu_eff_c, mu_eff_r)
    g1, g2 = st.columns(2)
    g1.plotly_chart(f1, config=PLOT_CONFIG, key="route_sw1")
    g2.plotly_chart(f2, config=PLOT_CONFIG, key="route_sw2")
    for label, measured, mu_eff, note, unit in (
        ("corner shift", meas_c, mu_eff_c, note_c, "mm"),
        ("rotation", meas_r, mu_eff_r, note_r, "°"),
    ):
        if measured <= 0:
            continue
        if mu_eff is not None:
            st.success(
                f"Measured {label} {measured:g} {unit} → the model predicts it with an effective μs ≈ {mu_eff:.3f} "
                f"on this route (setting: {mu_s:.3f})."
            )
        else:
            st.info(f"Measured {label} {measured:g} {unit}: {note}.")
    st.caption(
        f"Green line: from this μs on (the highest μ any recording requires) nothing slips on the route. Dashed grey: "
        "the shift if every segment had moved the cluster the same way; the blue line is lower where shifts cancel."
    )
    st.dataframe(tot, hide_index=True)
    with st.expander("Corner shift per recording and μs [mm]"):
        order = {n: i for i, n in enumerate(state["names"])}
        piv = per.pivot_table(index="Recording", columns="mu_s", values="corner_mm").round(2)
        piv = piv.loc[sorted(piv.index, key=lambda n: order.get(n, 1e9))]
        piv.columns = [f"μs {c_:.3f}" for c_ in piv.columns]
        st.dataframe(piv)

# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
with tab_over, tab_guard("Overview"):
    st.plotly_chart(fig_friction(res), config=PLOT_CONFIG, key="ov_friction")
    st.plotly_chart(fig_dynamic_limit(kin, res), config=PLOT_CONFIG, key="ov_dynlimit")
    st.caption(
        "The friction limit is not constant: when the carrier bumps or drops (vertical acceleration below 1 g) the "
        "cluster is momentarily lighter and the limit falls. The blue curve is the friction the cluster needs "
        "(required μ × vertical acceleration, including turntable rotation). Slip happens where it reaches the red one."
    )
    st.markdown("##### Most critical moments")
    st.caption(
        "Highest friction demand along the record (slip or not). Use the clock time to find the moment in the video. "
        "Driver tells whether the demand comes mainly from linear acceleration or from yaw acceleration (turntables)."
    )
    st.dataframe(crit, hide_index=True)

# --------------------------------------------------------------------------- #
# Measure
# --------------------------------------------------------------------------- #
with tab_meas, tab_guard("Measure"):
    st.caption(
        "Measure precise time intervals. 1) Drag across the whole-record strip to choose a zoom window. "
        "2) In the zoomed chart, drag from point A to point B, or type A and B below (to the millisecond). "
        "Dots are the samples actually recorded by the sensor."
    )
    t_min, t_max = float(kin.t[0]), float(kin.t[-1])
    k_w0, k_w1, k_a, k_b = (f"meas_{x}_{active}_{t_min:.1f}_{t_max:.1f}" for x in ("w0", "w1", "a", "b"))
    if k_w0 not in st.session_state:
        c_ = float(crit.iloc[0]["Time [s]"]) if len(crit) else t_min
        st.session_state[k_w0] = round(max(t_min, c_ - 5.0), 2)
        st.session_state[k_w1] = round(min(t_max, c_ + 5.0), 2)
        st.session_state[k_a] = round(float(np.clip(c_ - 0.5, t_min, t_max)), 3)
        st.session_state[k_b] = round(float(np.clip(c_ + 0.5, t_min, t_max)), 3)

    # 1) zoom window from the overview strip
    w0_, w1_ = sorted((float(st.session_state[k_w0]), float(st.session_state[k_w1])))
    ev_ov = st.plotly_chart(
        fig_measure_overview(kin, res, rotations, w0_, w1_),
        config=PLOT_CONFIG,
        key=f"meas_overview_{active}",
        on_select="rerun",
        selection_mode="box",
    )
    rng_ov = _selected_range(ev_ov)
    if rng_ov is not None and st.session_state.get("meas_overview_last") != rng_ov:
        st.session_state["meas_overview_last"] = rng_ov
        st.session_state[k_w0] = round(max(t_min, rng_ov[0]), 2)
        st.session_state[k_w1] = round(min(t_max, rng_ov[1]), 2)
        st.rerun()

    c1, c2, c3 = st.columns([1, 1, 3])
    c1.number_input("Zoom window from [s]", key=k_w0, step=0.5, format="%.2f")
    c2.number_input("Zoom window to [s]", key=k_w1, step=0.5, format="%.2f")
    sig_options = {
        "Horizontal accel. at CoM [g]": (res.fh_com, ".3f"),
        "Vertical accel. [g]": (kin.fz, ".3f"),
        "Accel. along travel [g]": (a_long, ".3f"),
        "Travel speed [m/s]": (travel_speed, ".3f"),
        "Yaw rate [deg/s]": (np.rad2deg(kin.wz), ".2f"),
        "Heading [deg]": (np.rad2deg(kin.heading), ".2f"),
        "Required μ": (np.clip(res.mu_req, 0, 5), ".3f"),
    }
    sig_sel = c3.multiselect(
        "Signals",
        list(sig_options.keys()),
        default=["Horizontal accel. at CoM [g]", "Vertical accel. [g]", "Yaw rate [deg/s]", "Travel speed [m/s]"],
        key="meas_signals",
    )
    w0_, w1_ = sorted((float(st.session_state[k_w0]), float(st.session_state[k_w1])))
    w0_, w1_ = max(t_min, w0_), min(t_max, w1_)
    if w1_ - w0_ < 0.2:
        w1_ = min(t_max, w0_ + 0.2)

    # 2) A-B interval in the full-resolution zoom
    a_, b_ = float(st.session_state[k_a]), float(st.session_state[k_b])
    if sig_sel:
        ev_det = st.plotly_chart(
            fig_measure_detail(kin, {k_: sig_options[k_] for k_ in sig_sel}, w0_, w1_, a_, b_),
            config=PLOT_CONFIG,
            key=f"meas_detail_{active}",
            on_select="rerun",
            selection_mode="box",
        )
        rng_det = _selected_range(ev_det)
        if rng_det is not None and st.session_state.get("meas_detail_last") != rng_det:
            st.session_state["meas_detail_last"] = rng_det
            st.session_state[k_a] = round(rng_det[0], 3)
            st.session_state[k_b] = round(rng_det[1], 3)
            st.rerun()
    else:
        st.info("Select at least one signal.")

    c1, c2, c3, c4 = st.columns(4)
    a_ = c1.number_input("A [s]", key=k_a, step=0.01, format="%.3f")
    b_ = c2.number_input("B [s]", key=k_b, step=0.01, format="%.3f")
    c3.metric("Δt = B − A", f"{abs(b_ - a_):.3f} s")
    dt_native = 1.0 / kin.fs_native if kin.fs_native > 0 else float("nan")
    c4.metric(
        "Sensor sample spacing",
        f"{dt_native:.3f} s",
        help="The sensor records one sample every this many seconds. Intervals cannot be resolved more finely "
        "than about one sample spacing; record at a higher rate to measure shorter times.",
    )
    if abs(b_ - a_) < 2 * dt_native:
        st.warning(
            f"The interval is shorter than two sensor samples ({2 * dt_native:.2f} s): the result is limited by the "
            "sample rate of the recording."
        )
    if abs(b_ - a_) > 0:
        _show_measurement(min(a_, b_), max(a_, b_))

# --------------------------------------------------------------------------- #
# Turntables
# --------------------------------------------------------------------------- #
with tab_turn, tab_guard("Turntables"):
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
        st.plotly_chart(fig_turn_detail(kin, res, w0, w1), config=PLOT_CONFIG, key="tt_detail")
        st.caption(
            "Friction demand = required μ × vertical acceleration: the friction the cluster needs, including the "
            "turntable rotational load, expressed in g. Dynamic friction limit = μs × vertical acceleration: the "
            "friction available, which drops on bumps. Where the blue curve reaches the red one, the cluster slips."
        )

# --------------------------------------------------------------------------- #
# Speed profile
# --------------------------------------------------------------------------- #
with tab_speed, tab_guard("Speed profile"):
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
        fig_speed_timeline(kin, res, travel_speed, speed_q, rotations), config=PLOT_CONFIG, key="sp_timeline"
    )
    st.caption(
        "Middle chart: same friction demand as in the Turntables tab (translation + rotation, in g) against the "
        "dynamic friction limit μs·f_z, which drops when the vertical acceleration drops. Where the demand reaches "
        "the limit the cluster slides; read it together with the speed above to see whether a start ramp, a stop, "
        "a stopper impact or a turntable causes it."
    )
    st.markdown("##### Motion segments")
    st.dataframe(speed_tab, hide_index=True)
    st.markdown("##### Start and stop ramps")
    st.caption(
        "Reliable segments overlaid at their start and at their stop. A steeper curve means a harder ramp; "
        "a sudden drop at the stop is a stopper impact."
    )
    if len(good_tab):
        st.plotly_chart(fig_speed_ramps(kin, travel_speed, speed_tab), config=PLOT_CONFIG, key="sp_ramps")
    else:
        st.info("No reliable motion segment in this record.")

# --------------------------------------------------------------------------- #
# Animation
# --------------------------------------------------------------------------- #
with tab_anim, tab_guard("Top-view animation"):
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
    play_speed = c3.selectbox(
        "Playback speed", [0.25, 0.5, 1.0, 2.0, 5.0, 10.0], index=2, format_func=lambda v: f"{v:g}×"
    )
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
                playback_speed=float(play_speed),
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
with tab_sig, tab_guard("Signals"):
    st.plotly_chart(fig_kinematics(kin, res), config=PLOT_CONFIG, key="sig_kin")
    spikes_ = getattr(kin, "spikes", []) or []
    if spikes_:
        with st.expander(f"Spikes removed by the spike filter ({len(spikes_)})", expanded=False):
            c0 = pd.Timestamp(raw.df["clock"].iloc[0])
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Time [s]": round(t_, 3),
                            "Clock": fmt_clock(c0 + pd.to_timedelta(t_, unit="s")),
                            "Channel (sensor axes)": ch_,
                            "Value": v_,
                            "Neighbours": n_,
                            "Unit": "deg/s" if ch_ == "Yaw rate" else "g",
                        }
                        for t_, ch_, v_, n_ in spikes_
                    ]
                ),
                hide_index=True,
            )
            st.caption(
                "Each of these samples was replaced (all channels) by interpolation between its neighbours before the "
                "simulation. Change the thresholds under Signal processing → Spike filter in the sidebar."
            )
    elif spike_mode == "on" or (spike_mode == "auto" and kin.fs_native >= 50):
        st.caption("Spike filter on: no isolated spikes found in this recording.")
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
with tab_ev, tab_guard("Slip events"):
    ev_scope = st.radio("Show the slip events of", [SCOPE_ACTIVE, SCOPE_ROUTE], horizontal=True, key="ev_scope")
    if ev_scope == SCOPE_ROUTE:
        render_route_slip_events()
    else:
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
with tab_sens, tab_guard("Friction sensitivity"):
    fs_scope = st.radio("Run for", [SCOPE_ACTIVE, SCOPE_ROUTE], horizontal=True, key="fs_scope")
    if fs_scope == SCOPE_ROUTE:
        st.caption(
            "Repeats the simulation of every uploaded recording for a range of static friction coefficients and adds "
            "the shifts up along the line (ST 130 → buffer), as if one part had travelled all the measured segments. "
            "Enter what you measure on the part at the end of the line to find the friction that explains it."
        )
    else:
        st.caption(
            "Repeats the simulation for a range of static friction coefficients (same record and settings). "
            "Use it to judge how sensitive the result is to surface condition, contamination or material changes."
        )
    c1, c2, c3, c4 = st.columns(4)
    mu_lo = c1.number_input("μs from", value=0.10, min_value=0.01, max_value=2.0, step=0.01, format="%.2f")
    mu_hi = c2.number_input("μs to", value=0.40, min_value=0.02, max_value=2.0, step=0.01, format="%.2f")
    n_mu = c3.number_input("Steps", value=7, min_value=2, max_value=25, step=1)
    keep_ratio = c4.toggle("Keep μk/μs ratio", value=True)
    if fs_scope == SCOPE_ROUTE:
        render_route_sweep(float(mu_lo), float(mu_hi), int(n_mu), bool(keep_ratio))
    elif st.button("Run sensitivity analysis", type="primary"):
        if mu_hi <= mu_lo:
            st.error("The upper μ must be larger than the lower μ.")
        else:
            mus = tuple(float(round(v, 4)) for v in np.linspace(mu_lo, mu_hi, int(n_mu)))
            with st.spinner(f"Running {len(mus)} simulations..."):
                st.session_state["sweep"] = {
                    "key": (active, json.dumps([pre, cluster_d, contact_d, sensor_d, solver_d], default=str)),
                    "df": run_sweep(data, active, pre, cluster_d, contact_d, sensor_d, solver_d, mus, keep_ratio),
                }
    sw_state = st.session_state.get("sweep") if fs_scope == SCOPE_ACTIVE else None
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
# --------------------------------------------------------------------------- #
# Line layout map (Compare tab)
# --------------------------------------------------------------------------- #
def render_line_map(cmp_df: pd.DataFrame, peaks_map: dict, mu_s: float) -> None:
    """Every recording drawn along the conveyor path its file name describes, coloured by exceedances of μs."""
    recordings = list(cmp_df["Recording"])
    st.caption(
        "Every uploaded recording is simulated over its full length and drawn where its file name says it was "
        "measured. **St430-440**: transport from ST 430 to ST 440, drawn along the conveyors in the flow direction "
        "(ST 130 → buffer, grey arrows). **St989**, **St540 deindex**, **St540Loop540**: one station only, i.e. a "
        "process inside the station (indexing, deindexing, ...), drawn as a ring. **St981-Buffer**: transport to "
        "the buffer. Red = μs exceeded, the badge is the number of exceedances. The map shows in which section, "
        "not where inside it."
    )
    map_box = st.container()
    with st.expander("Sections from the file names", expanded=False):
        st.caption(
            "Read from the file names. Correct a row here if a name does not follow the convention. **Via**: "
            "stations the carrier passes, separated by commas, when the shortest way along the conveyors is not "
            "the one it takes. Stations on the layout: "
            + ", ".join(station_display(k) for k in known_stations())
            + "."
        )
        base = sections_table(recordings)
        ed_key = "line_sections_" + hashlib.md5(base.to_json().encode()).hexdigest()[:10]
        sections = st.data_editor(base, key=ed_key, hide_index=True, disabled=["Recording"])
    sections = pd.DataFrame(sections).fillna("")

    row_of = {r["Recording"]: r for _, r in cmp_df.iterrows()}
    groups: dict[tuple, dict] = {}
    unmapped: list[tuple[str, dict]] = []
    for _, row in sections.iterrows():
        rec = row["Recording"]
        geo = section_geometry(row)
        pk = peaks_map.get(rec)
        n = 0 if pk is None else len(pk)
        times = [] if pk is None or not len(pk) else [str(c) for c in pk["Clock"].tolist()]
        mu_max = float(row_of[rec]["Required μ (no slip)"]) if rec in row_of else float("nan")
        if not geo["path"]:
            unmapped.append((rec, geo))
            key = ("unmapped", rec)
        else:
            key = (geo["kind"], tuple(geo["keys"]))
        g = groups.setdefault(
            key,
            {
                "label": section_label(row),
                "kind": geo["kind"],
                "path": geo["path"],
                "keys": geo["keys"],
                "geo": geo,
                "n_exc": 0,
                "mu_max": float("nan"),
                "recordings": [],
                "n_per_rec": [],
                "times": [],
                "notes": set(),
            },
        )
        g["n_exc"] += n
        if np.isfinite(mu_max):
            g["mu_max"] = mu_max if not np.isfinite(g["mu_max"]) else max(g["mu_max"], mu_max)
        g["recordings"].append(rec)
        g["n_per_rec"].append(n)
        g["times"].append(times)
        g["notes"].add(str(row.get("Note") or ""))
        if len(g["notes"]) > 1:  # several recordings with different notes on the same section
            g["label"] = section_label(dict(row, Note=""))
    secs = list(groups.values())

    with map_box:
        W, H = layout_image_size()
        st.plotly_chart(
            fig_line_map(
                layout_background_uri(),
                W,
                H,
                layout_network(),
                [g for g in secs if g["path"]],
                layout_stations(),
                mu_s,
                arrows=layout_arrows(),
            ),
            config=PLOT_CONFIG,
            key="line_map",
        )
        if unmapped:
            parts = []
            for rec, geo in unmapped:
                if geo["kind"] == "missing":
                    parts.append(f"{rec} ({', '.join(station_display(k) for k in geo['missing'])} not on the layout)")
                elif geo["kind"] == "no_path":
                    parts.append(f"{rec} (no conveyor connection between its stations)")
                else:
                    parts.append(f"{rec} (no station in the file name)")
            st.warning(
                f"{len(unmapped)} recording(s) not on the map: " + "; ".join(parts) + ". Correct them under "
                "**Sections from the file names**."
            )
        drawn = [g for g in secs if g["path"]]
        n_bad = sum(1 for g in drawn if g["n_exc"] > 0)
        st.caption(
            f"{n_bad} of {len(drawn)} sections exceed μs = {mu_s:.3f}. Grey = conveyor path, red = μs exceeded "
            "(the line gets thicker with more exceedances), blue = measured without exceedance, ring = process inside "
            "a station. "
            "Hover a badge for the recordings and times; the camera icon saves the map as an image."
        )
        tab = pd.DataFrame(
            [
                {
                    "Section": g["label"],
                    "Recordings": ", ".join(g["recordings"]),
                    "Exceedances of μs": g["n_exc"],
                    "Highest required μ": round(g["mu_max"], 3) if np.isfinite(g["mu_max"]) else None,
                    "Exceedance times": ", ".join(t for ts in g["times"] for t in ts),
                    "On the map": "Yes" if g["path"] else "No",
                }
                for g in sorted(secs, key=lambda g: -g["n_exc"])
            ]
        )
        st.dataframe(tab, hide_index=True)


with tab_cmp, tab_guard("Compare recordings"):
    if len(valid) < 2:
        st.info("Upload two or more recordings to compare them (e.g. before / after a conveyor adjustment).")
    else:
        st.caption(
            "Every recording is simulated over its full length with the current mounting, cluster and friction "
            "settings. Recordings are listed in time order (oldest first)."
        )
        with st.spinner("Simulating all recordings..."):
            rows, peaks_map, row_errors, _extras = all_recording_rows()
        for msg in row_errors:
            st.warning(msg)
        if rows:
            cmp_df = pd.DataFrame(rows)
            st.plotly_chart(fig_compare(cmp_df, mu_s, peaks_map), config=PLOT_CONFIG, key="cmp")
            st.caption(
                "Bar = highest friction coefficient required in the recording. Orange marks = every distinct moment "
                "in which the required μ exceeds μs, at its peak value (exceedances less than 1 s apart count as "
                "one). The number at the end of each bar is how many times μs is exceeded."
            )
            st.dataframe(cmp_df, hide_index=True)
            st.download_button(
                "Download comparison (CSV)", cmp_df.to_csv(index=False).encode(), "comparison.csv", "text/csv"
            )

    # ------------------------------------------------------------------ #
    # Compare turntable rotations between recordings
    # ------------------------------------------------------------------ #
    st.divider()
    st.markdown("##### Compare turntable rotations")
    st.caption(
        "Pick a turn in one recording and a turn in another (or in the same) recording. The curves are aligned "
        "at the chosen point and overlaid: yaw rate, friction demand against the dynamic friction limit, and yaw "
        "acceleration."
    )
    rec_names = list(valid.keys())
    pre_full = dict(pre, t_start=None, t_end=None)

    def _turn_data(rec: str):
        d_ = valid[rec]
        kin_x = load_kinematics(d_, rec, pre_full)
        res_x = run_simulation(d_, rec, pre_full, cluster_d, contact_d, sensor_d, solver_d)
        rots_x = [r for r in detect_rotations(kin_x, turn_s) if r.kind == "Turntable"]
        return kin_x, res_x, rots_x

    def _turn_label(r) -> str:
        return f"{r.label} · {r.angle_deg:+.1f}° · {r.t1 - r.t0:.1f} s"

    cA, cB, cC = st.columns([2, 2, 1.3])
    picks = []
    for col, tag, default_idx in ((cA, "A", 0), (cB, "B", min(1, len(rec_names) - 1))):
        with col:
            rec = st.selectbox(
                f"Recording {tag}", rec_names, index=default_idx, format_func=_recording_label, key=f"tc_rec_{tag}"
            )
            kin_x, res_x, rots_x = _turn_data(rec)
            if not rots_x:
                st.info("No turntable rotation detected in this recording.")
                continue
            same_as_a = tag == "B" and picks and picks[0][1] == rec
            r_idx = st.selectbox(
                f"Turn {tag}",
                list(range(len(rots_x))),
                index=1 if same_as_a and len(rots_x) > 1 else 0,
                format_func=lambda i, rots_x=rots_x: _turn_label(rots_x[i]),
                key=f"tc_turn_{tag}_{rec}",
            )
            picks.append((tag, rec, kin_x, res_x, rots_x[r_idx]))
    with cC:
        align = st.radio("Align at", ["Turn start", "Turn end"], key="tc_align")
        tc_pad = st.number_input("Show before / after [s]", value=float(turn_s.pad_s), min_value=0.5, step=0.5)

    if len(picks) == 2:
        series, table_rows = [], []
        for tag, rec, kin_x, res_x, r in picks:
            ref = r.t0 if align == "Turn start" else r.t1
            i0 = int(np.searchsorted(kin_x.t, r.t0 - tc_pad))
            i1 = int(np.searchsorted(kin_x.t, r.t1 + tc_pad, side="right"))
            sl_ = slice(i0, max(i1, i0 + 2))
            sign_ = 1.0 if r.angle_deg >= 0 else -1.0
            demand = friction_demand_g(res_x, kin_x)
            limit = dynamic_limit_g(res_x, kin_x)
            series.append(
                {
                    "label": f"{tag}: {rec} · {r.label}",
                    "t": kin_x.t[sl_] - ref,
                    "yaw_rate": np.abs(np.rad2deg(kin_x.wz[sl_])),
                    "demand_g": demand[sl_],
                    "limit_g": limit[sl_],
                    "yaw_acc": sign_ * np.rad2deg(kin_x.alpha[sl_]),
                }
            )
            tt_x = analyse_rotations(kin_x, res_x, [r], turn_s).iloc[0]
            rot_m = (kin_x.t >= r.t0) & (kin_x.t <= r.t1)
            around = (kin_x.t >= r.t0 - tc_pad) & (kin_x.t <= r.t1 + tc_pad)
            table_rows.append(
                {
                    "": tag,
                    "Recording": rec,
                    "Turn": r.label,
                    "Start clock": tt_x["Start clock"],
                    "Angle [deg]": tt_x["Angle [deg]"],
                    "Duration [s]": tt_x["Duration [s]"],
                    "Peak rate [deg/s]": tt_x["Peak rate [deg/s]"],
                    "Spin-up time [s]": tt_x["Spin-up time [s]"],
                    "Braking time [s]": tt_x["Braking time [s]"],
                    "Peak spin-up accel. [deg/s2]": tt_x["Peak spin-up accel. [deg/s2]"],
                    "Peak braking decel. [deg/s2]": tt_x["Peak braking decel. [deg/s2]"],
                    "Slow / creep phase [deg]": tt_x["Slow / creep phase [deg]"],
                    "Max demand · rotation [g]": round(float(np.nanmax(demand[rot_m])), 3) if rot_m.any() else np.nan,
                    "Max demand · window [g]": round(float(np.nanmax(demand[around])), 3) if around.any() else np.nan,
                    "Lowest dynamic limit [g]": round(float(np.nanmin(limit[around])), 3) if around.any() else np.nan,
                    "Required mu · entry": tt_x["Required mu · entry"],
                    "Required mu · rotation": tt_x["Required mu · rotation"],
                    "Required mu · exit": tt_x["Required mu · exit"],
                    "Slip predicted": tt_x["Slip predicted"],
                }
            )
        st.plotly_chart(fig_turn_compare(series), config=PLOT_CONFIG, key="turn_compare")
        st.dataframe(pd.DataFrame(table_rows).set_index("").T, use_container_width=True)
        st.caption(
            "Friction demand = required μ × vertical acceleration (translation and turntable rotation), in g. "
            "Dynamic limit = μs × vertical acceleration. Where a demand curve reaches its limit, that cluster slips."
        )

# --------------------------------------------------------------------------- #
# Line map
# --------------------------------------------------------------------------- #
with tab_map, tab_guard("Line map"):
    with st.spinner("Simulating all recordings..."):
        map_rows, map_peaks, map_errors, _map_extras = all_recording_rows()
    for msg in map_errors:
        st.warning(msg)
    if map_rows:
        render_line_map(pd.DataFrame(map_rows), map_peaks, mu_s)
    else:
        st.info("None of the uploaded recordings could be simulated.")

# --------------------------------------------------------------------------- #
# Method
# --------------------------------------------------------------------------- #
with tab_meth, tab_guard("Method"):
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

#### Spike filter

Bluetooth transmission errors produce isolated samples far away from the real signal (a single sample of several g).
A sample is treated as a glitch when it differs from the median of its neighbours (window of 2·w + 3 samples) by more
than the jump threshold (0.5 g for accelerations, 15 deg/s for the yaw rate by default), the deviation lasts at most
w samples (1 by default; 5 ms at 200 Hz) and the neighbours sit at the normal level of the signal (median over about
0.2 s). Around the peak of a real impact the neighbouring samples are raised as well, so the peak is kept. A sample
whose vertical acceleration is below 0.3 g is a dropped, zero or inverted data packet (a carrier cannot fall faster
than gravity; real bumps reach about 0.7 g) and is removed whatever its width. Each glitch sample is replaced by linear interpolation in all channels,
because the data packet is corrupt. The optional hard limit removes every acceleration sample farther than the limit
from its typical value, whatever its duration. In Auto mode the filter runs only from 50 Hz on: at 10 Hz a real impact
can be a single sample.

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

#### Line layout map (Compare tab)

The plant layout, the conveyor path and the station positions are embedded in the app (generated once from the
layout with the path drawn in red). The conveyors are directed: carriers flow from ST 130 to the buffer; side
branches that leave and rejoin the main line run in the same direction, and the loop through ST 540 returns
upstream (20 → 21 → ST 540 → 23 → 18). Every recording is drawn as the shortest way in the flow direction between
the stations in its file name (`St<from>-<to>`; a name written against the flow is drawn on the same stretch).
A name with one station only (`St989`, `St540 deindex`, `St540Loop540`) is a process inside the station (indexing,
deindexing, ...), not a transport, and is drawn as a ring at the station. Where the carrier takes another way, the
stations it passes can be added as *Via* in the sections table. The map tells in which section μs is exceeded and
how often, not where inside the section.

#### Whole route (Slip events and Friction sensitivity tabs)

Every recording is simulated on its own over its full length, starting with the cluster in its nominal position, and
the recordings are put in line order (ST 130 → buffer) from their file names. The cluster shift of each recording is
a translation of the centre of mass and a rotation in the carrier (GOT) frame, the same for every recording, so the
shifts add up: U = Σ u_i, φ = Σ φ_i, and the shift of a corner is U + (R(φ) − I)·r_c. This is valid for shifts of
millimetres; shifts in opposite directions cancel. Repeated measurements of the same section should be left out,
otherwise they are added twice. With the shift measured on the part at the end of the line, the friction sweep gives
the effective μs at which the model predicts the same shift (linear interpolation between the computed μ values).

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
