"""
data_io.py
==========

Reading and pre-processing of WitMotion IMU exports (tab-separated .txt files
written by the WitMotion PC / mobile application).

The functions in this module turn a raw export into a clean, uniformly sampled
set of *carrier kinematics* expressed in the carrier (GOT) frame:

    X  : along the cluster length  ("Front" = +X)
    Y  : along the cluster width   ("Left"  = +Y)
    Z  : up

All accelerations are kept as **specific force** in g (what an accelerometer
actually measures: acceleration minus gravity). For a sensor mounted parallel
to the GOT support surface this is exactly the quantity that loads the
friction contact, so no explicit gravity removal is needed:

    tangential friction demand = m * g0 * f_xy
    normal contact force       = m * g0 * f_z
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import butter, savgol_filter, sosfiltfilt

G0 = 9.80665  # standard gravity [m/s^2]

# Canonical column names -> list of accepted header prefixes in WitMotion exports
_COLUMN_MAP = {
    "time": ["time"],
    "device": ["DeviceName"],
    "ax": ["AccX"],
    "ay": ["AccY"],
    "az": ["AccZ"],
    "wx": ["AsX"],
    "wy": ["AsY"],
    "wz": ["AsZ"],
    "roll": ["AngleX"],
    "pitch": ["AngleY"],
    "yaw": ["AngleZ"],
    "temp": ["Temperature"],
    "battery": ["Battery"],
}
_REQUIRED = ["time", "ax", "ay", "az", "wz"]


# --------------------------------------------------------------------------- #
# Raw file parsing
# --------------------------------------------------------------------------- #
@dataclass
class RawRecording:
    """A parsed WitMotion export (one device)."""

    name: str
    df: pd.DataFrame  # columns: clock (datetime64), ax, ay, az, wx, wy, wz, roll, pitch, yaw, ...
    device: str
    devices_in_file: list[str]
    n_rows_raw: int
    n_duplicates_removed: int


def _match_column(columns: list[str], prefixes: list[str]) -> str | None:
    for c in columns:
        base = c.strip()
        for p in prefixes:
            if base.lower().startswith(p.lower()):
                return c
    return None


def read_witmotion(data: bytes | str, name: str = "recording", device: str | None = None) -> RawRecording:
    """Parse a WitMotion tab-separated export.

    Parameters
    ----------
    data : bytes or str
        File content (as uploaded) or a path to the file.
    name : str
        Display name of the recording.
    device : str, optional
        Device to keep if the file contains several sensors. Defaults to the
        device with the most rows.
    """
    if isinstance(data, (bytes, bytearray)):
        text = None
        for enc in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                text = data.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        buf = io.StringIO(text)
    else:
        buf = data

    df = pd.read_csv(buf, sep="\t", na_values=["null", "NULL", "NaN", ""])
    df.columns = [str(c).strip() for c in df.columns]
    # Drop trailing empty columns created by trailing tabs
    df = df.loc[:, [c for c in df.columns if c and not c.startswith("Unnamed")]]

    cols = list(df.columns)
    rename = {}
    for key, prefixes in _COLUMN_MAP.items():
        col = _match_column(cols, prefixes)
        if col is not None:
            rename[col] = key
    missing = [k for k in _REQUIRED if k not in rename.values()]
    if missing:
        raise ValueError(
            "The file does not look like a WitMotion export. Missing columns: "
            + ", ".join(missing)
            + f". Found: {', '.join(cols[:12])}..."
        )
    df = df.rename(columns=rename)
    n_raw = len(df)

    # Device selection
    if "device" in df.columns:
        counts = df["device"].astype(str).value_counts()
        devices = counts.index.tolist()
        chosen = device if device in devices else devices[0]
        df = df[df["device"].astype(str) == chosen]
    else:
        devices, chosen = ["(unnamed)"], "(unnamed)"

    df = df.copy()
    df["clock"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["clock", "ax", "ay", "az", "wz"])
    num_cols = [
        c for c in ["ax", "ay", "az", "wx", "wy", "wz", "roll", "pitch", "yaw", "temp", "battery"] if c in df.columns
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("clock", kind="stable")

    # Exact duplicate packets (same timestamp and same values) are removed.
    before = len(df)
    df = df.drop_duplicates(subset=["clock"] + num_cols, keep="first")
    n_dup = before - len(df)

    keep = ["clock"] + num_cols
    df = df[keep].reset_index(drop=True)
    if len(df) < 20:
        raise ValueError("The recording contains fewer than 20 valid samples.")

    return RawRecording(
        name=name,
        df=df,
        device=chosen,
        devices_in_file=devices,
        n_rows_raw=n_raw,
        n_duplicates_removed=n_dup,
    )


# --------------------------------------------------------------------------- #
# Pre-processing
# --------------------------------------------------------------------------- #
@dataclass
class PreprocessSettings:
    """User settings controlling how raw data becomes carrier kinematics."""

    timestamp_mode: str = "smoothed"  # "smoothed" | "raw" | "uniform"
    sensor_yaw_deg: float = 0.0  # angle from carrier +X to sensor +X (about +Z, CCW)
    auto_flip: bool = True  # flip axes if the sensor is mounted upside down
    bias_mode: str = "window"  # "window" | "none"
    bias_window_s: tuple[float, float] = (0.0, 3.0)  # relative to recording start
    lowpass_hz: float | None = None  # zero-phase Butterworth cut-off (None = off)
    resample_hz: float = 100.0
    alpha_window_s: float = 0.3  # Savitzky-Golay window for d(omega)/dt
    gap_factor: float = 3.0  # a gap is flagged when dt > gap_factor * nominal period
    t_start: float | None = None  # analysis window [s from recording start]
    t_end: float | None = None


@dataclass
class Kinematics:
    """Uniformly sampled carrier kinematics in the carrier frame."""

    t: np.ndarray  # [s] from recording start
    clock: np.ndarray  # datetime64[ns]
    fx: np.ndarray  # specific force [g] at the sensor, carrier X
    fy: np.ndarray  # [g] carrier Y
    fz: np.ndarray  # [g] carrier Z (1 g at rest)
    wz: np.ndarray  # yaw rate [rad/s]
    alpha: np.ndarray  # yaw acceleration [rad/s^2]
    heading: np.ndarray  # integrated yaw [rad] (relative to start of window)
    fs: float  # simulation sample rate [Hz]
    fs_native: float  # median native sample rate [Hz]
    gaps: list[tuple[float, float]] = field(default_factory=list)  # (t, duration) of gaps
    flipped: bool = False
    bias: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def dt(self) -> float:
        return 1.0 / self.fs

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) else 0.0


def _make_increasing(t: np.ndarray, min_step: float) -> np.ndarray:
    """Return a strictly increasing copy of t (non-positive steps -> min_step)."""
    d = np.diff(t)
    d = np.where(d <= 0, min_step, d)
    return np.concatenate([[t[0]], t[0] + np.cumsum(d)])


def _retime(clock: pd.Series, mode: str) -> np.ndarray:
    """Return sample times [s] from the first sample.

    Bluetooth loggers stamp packets when they *arrive* at the PC, so packets
    are often delivered in bursts (several samples within a few ms followed by
    a long pause). The sensor itself samples at a fixed rate, therefore:

    * "raw"      : use the receive time as logged (de-duplicated).
    * "uniform"  : sample index x mean period (assumes no lost packets).
    * "smoothed" : sample index x mean period + slowly varying correction
                   (rolling median of the receive-time residual). Removes the
                   burst jitter while following real dropouts and clock drift.
    """
    t_raw = (clock - clock.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    n = len(t_raw)
    idx = np.arange(n, dtype=float)
    if mode == "raw":
        return _make_increasing(t_raw, 1e-4)
    period = (t_raw[-1] - t_raw[0]) / max(n - 1, 1)
    t_uniform = idx * period
    if mode == "uniform":
        return t_uniform
    # smoothed
    resid = t_raw - t_uniform
    win = int(max(5, min(201, round(5.0 / max(period, 1e-3)))))  # ~5 s window
    if win % 2 == 0:
        win += 1
    corr = pd.Series(resid).rolling(win, center=True, min_periods=1).median().to_numpy()
    t = t_uniform + corr
    t -= t[0]
    return _make_increasing(t, 0.25 * period)


def native_rate(raw: RawRecording) -> float:
    t = (raw.df["clock"] - raw.df["clock"].iloc[0]).dt.total_seconds().to_numpy()
    if len(t) < 2:
        return float("nan")
    return (len(t) - 1) / max(t[-1] - t[0], 1e-9)


def recording_duration(raw: RawRecording) -> float:
    c = raw.df["clock"]
    return float((c.iloc[-1] - c.iloc[0]).total_seconds())


def preprocess(raw: RawRecording, s: PreprocessSettings) -> Kinematics:
    """Convert a raw recording into uniformly sampled carrier kinematics."""
    df = raw.df
    warnings: list[str] = []

    t_native = _retime(df["clock"], s.timestamp_mode)
    clock0 = df["clock"].iloc[0]

    ax = df["ax"].to_numpy(float)
    ay = df["ay"].to_numpy(float)
    az = df["az"].to_numpy(float)
    wz = np.deg2rad(df["wz"].to_numpy(float))

    # Fill isolated NaNs
    for arr in (ax, ay, az, wz):
        bad = ~np.isfinite(arr)
        if bad.any():
            arr[bad] = np.interp(t_native[bad], t_native[~bad], arr[~bad])

    # --- Mounting: upside-down detection (rotation of 180 deg about sensor X)
    flipped = False
    if s.auto_flip and np.nanmedian(az) < 0:
        ay, az, wz = -ay, -az, -wz
        flipped = True
        warnings.append("Sensor Z axis points down: axes were flipped (180 deg about sensor X).")

    # --- Mounting: yaw rotation sensor -> carrier
    th = np.deg2rad(s.sensor_yaw_deg)
    c, sn = np.cos(th), np.sin(th)
    fx = c * ax - sn * ay
    fy = sn * ax + c * ay
    fz = az

    # --- Native-rate diagnostics
    dt_native = np.diff(t_native)
    period = float(np.median(dt_native)) if len(dt_native) else float("nan")
    fs_native = 1.0 / period if period > 0 else float("nan")
    # Delivery pauses are evaluated on the receive time as logged
    t_recv = (df["clock"] - clock0).dt.total_seconds().to_numpy(float)
    dt_recv = np.diff(t_recv)
    gaps = []
    if np.isfinite(period):
        for i in np.where(dt_recv > s.gap_factor * period)[0]:
            gaps.append((float(t_recv[i]), float(dt_recv[i])))
    if fs_native < 50:
        warnings.append(
            f"Native sample rate is about {fs_native:.0f} Hz. Short impacts (10-50 ms) at stoppers and "
            "transfers are under-sampled; peak accelerations and slip are likely underestimated. "
            "Record at 100 Hz or more if the sensor allows it."
        )

    # --- Bias removal (sensor mounting tilt / offset), using a stationary window
    bias = {"fx": 0.0, "fy": 0.0, "fz_gain": 1.0, "wz": 0.0}
    if s.bias_mode == "window":
        b0, b1 = s.bias_window_s
        m = (t_native >= b0) & (t_native <= b1)
        if m.sum() >= 3:
            bias["fx"] = float(np.mean(fx[m]))
            bias["fy"] = float(np.mean(fy[m]))
            mz = float(np.mean(fz[m]))
            bias["fz_gain"] = 1.0 / mz if mz > 0.5 else 1.0
            bias["wz"] = float(np.mean(wz[m]))
            fx = fx - bias["fx"]
            fy = fy - bias["fy"]
            fz = fz * bias["fz_gain"]
            wz = wz - bias["wz"]
            spread = max(np.std(fx[m] + bias["fx"]), np.std(fy[m] + bias["fy"]))
            if spread > 0.01 or np.max(np.abs(np.rad2deg(wz[m]))) > 1.0:
                warnings.append(
                    "The bias reference window does not look stationary (signal spread above 0.01 g "
                    "or rotation above 1 deg/s). Choose a window where the carrier is at rest."
                )
        else:
            warnings.append("Bias reference window contains fewer than 3 samples; bias removal skipped.")

    # --- Analysis window
    t0 = s.t_start if s.t_start is not None else t_native[0]
    t1 = s.t_end if s.t_end is not None else t_native[-1]
    t0 = max(t0, t_native[0])
    t1 = min(t1, t_native[-1])
    if t1 - t0 < 1.0:
        raise ValueError("The analysis window must be at least 1 s long.")

    # --- Resample to a uniform grid (linear interpolation)
    fs = float(s.resample_hz)
    n = int(np.floor((t1 - t0) * fs)) + 1
    t = t0 + np.arange(n) / fs
    fx_u = np.interp(t, t_native, fx)
    fy_u = np.interp(t, t_native, fy)
    fz_u = np.interp(t, t_native, fz)
    wz_u = np.interp(t, t_native, wz)

    # --- Optional zero-phase low-pass filter
    if s.lowpass_hz:
        nyq_native = 0.5 * fs_native
        fc = min(float(s.lowpass_hz), 0.95 * nyq_native, 0.45 * fs)
        if fc < s.lowpass_hz:
            warnings.append(f"Low-pass cut-off limited to {fc:.2f} Hz (below the native Nyquist frequency).")
        sos = butter(2, fc, btype="low", fs=fs, output="sos")
        padlen = min(3 * 6, n - 1)
        fx_u = sosfiltfilt(sos, fx_u, padlen=padlen)
        fy_u = sosfiltfilt(sos, fy_u, padlen=padlen)
        fz_u = sosfiltfilt(sos, fz_u, padlen=padlen)
        wz_u = sosfiltfilt(sos, wz_u, padlen=padlen)

    # --- Yaw acceleration (Savitzky-Golay derivative)
    win = int(round(s.alpha_window_s * fs))
    win = max(win, 5)
    if win % 2 == 0:
        win += 1
    win = min(win, n if n % 2 == 1 else n - 1)
    if win >= 5:
        alpha = savgol_filter(wz_u, win, 2, deriv=1, delta=1.0 / fs)
    else:
        alpha = np.gradient(wz_u, 1.0 / fs)

    heading = np.concatenate([[0.0], np.cumsum(0.5 * (wz_u[1:] + wz_u[:-1]) / fs)])

    clock = (clock0 + pd.to_timedelta(t, unit="s")).to_numpy()

    return Kinematics(
        t=t,
        clock=clock,
        fx=fx_u,
        fy=fy_u,
        fz=fz_u,
        wz=wz_u,
        alpha=alpha,
        heading=heading,
        fs=fs,
        fs_native=fs_native,
        gaps=gaps,
        flipped=flipped,
        bias=bias,
        warnings=warnings,
    )


def kinematics_table(kin: Kinematics) -> pd.DataFrame:
    """Kinematics as a DataFrame (for export / inspection)."""
    return pd.DataFrame(
        {
            "time_s": kin.t,
            "clock": kin.clock,
            "fx_g": kin.fx,
            "fy_g": kin.fy,
            "fz_g": kin.fz,
            "yaw_rate_deg_s": np.rad2deg(kin.wz),
            "yaw_accel_deg_s2": np.rad2deg(kin.alpha),
            "heading_deg": np.rad2deg(kin.heading),
        }
    )
