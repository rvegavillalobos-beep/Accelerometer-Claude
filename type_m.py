"""Predicted corner deviations of a Type M module, in the coordinates of the ST020 corner measurement.

The nominal corner positions are the Type M values of the ST020 corner-deviation app (RICKHARD X, repo
rvegavillalobos-beep/AI, ``rickhard/constants.py``; same values as RICKHARDV). They are in the CAD reference system:
X grows towards the front of the module, the left corners carry negative Y.

The simulation gives the shift of the cluster centre of mass (ux, uy) and its rotation phi in the carrier frame
(+X front, +Y left, rotation counter-clockwise seen from above). A corner at r from the centre of mass moves by
U + (R(phi) - I) r. With the module front towards carrier +X, CAD X = carrier X and CAD Y = -carrier Y; with the
module turned by 180 deg on the carrier both signs flip. The deviations, the centroid and the rotation angle are
then computed exactly as the ST020 app does, so the numbers can be compared with a measured module.
"""

from __future__ import annotations

import os as _os

_LOADED_MTIME = _os.path.getmtime(__file__)  # used by app.py to detect stale modules

import numpy as np
import pandas as pd

CORNERS = ("FL", "FR", "RR", "RL")  # drawing order around the outline
CORNER_LABELS = {"FL": "Front Left", "FR": "Front Right", "RR": "Rear Right", "RL": "Rear Left"}

# Nominal corner coordinates [mm], CAD reference system (ST020 / RICKHARD X, Type M).
TYPE_M_NOMINAL_MM: dict[str, tuple[float, float]] = {
    "FL": (2290.48, -559.4),
    "FR": (2290.48, 558.9),
    "RL": (609.31, -583.3),
    "RR": (609.31, 535.0),
}
TYPE_M_SPEC_MM = 3.0  # default ± tolerance per corner and axis in the ST020 app

_N = TYPE_M_NOMINAL_MM
TYPE_M_LENGTH_M = ((_N["FL"][0] + _N["FR"][0]) - (_N["RL"][0] + _N["RR"][0])) / 2.0 / 1000.0
TYPE_M_WIDTH_M = ((_N["FR"][1] - _N["FL"][1]) + (_N["RR"][1] - _N["RL"][1])) / 2.0 / 1000.0


def nominal_centre_mm() -> tuple[float, float]:
    """Mean of the four nominal corners [mm] (the ST020 app's module centre)."""
    return (
        float(np.mean([_N[c][0] for c in CORNERS])),
        float(np.mean([_N[c][1] for c in CORNERS])),
    )


def _cad_from_carrier(x, y, front_plus_x: bool):
    s = 1.0 if front_plus_x else -1.0
    return s * np.asarray(x, dtype=float), -s * np.asarray(y, dtype=float)


def corner_deviations(
    dx_mm: float,
    dy_mm: float,
    rot_deg: float,
    com_offset_m=(0.0, 0.0),
    front_plus_x: bool = True,
) -> pd.DataFrame:
    """Deviation [mm] of each Type M corner (CAD X / Y) for a CoM shift (dx, dy) and a rotation, carrier frame.

    com_offset_m: centre of mass relative to the geometric centre of the contact footprint, carrier frame [m]
    (the simulation's ``patch.com``); it is placed relative to the Type M module centre."""
    cx0, cy0 = nominal_centre_mm()
    ox, oy = _cad_from_carrier(com_offset_m[0] * 1000.0, com_offset_m[1] * 1000.0, front_plus_x)
    com_cad = np.array([cx0 + float(ox), cy0 + float(oy)])
    nom = np.array([_N[c] for c in CORNERS], dtype=float)
    r_cad = nom - com_cad
    # carrier frame <-> CAD is its own inverse (sign flips only)
    rx, ry = _cad_from_carrier(r_cad[:, 0], r_cad[:, 1], front_plus_x)
    ph = np.deg2rad(float(rot_deg))
    c, s = np.cos(ph), np.sin(ph)
    dxc = float(dx_mm) + (c - 1.0) * rx - s * ry
    dyc = float(dy_mm) + s * rx + (c - 1.0) * ry
    dX, dY = _cad_from_carrier(dxc, dyc, front_plus_x)
    return pd.DataFrame(
        {
            "Corner": list(CORNERS),
            "Nominal X [mm]": nom[:, 0],
            "Nominal Y [mm]": nom[:, 1],
            "X [mm]": dX,
            "Y [mm]": dY,
        }
    )


def placement(dev: pd.DataFrame) -> dict:
    """Centroid shift and rotation angle with the ST020 app's definitions.

    Centroid = mean of the four corner deviations; rotation angle = change of the angle of the line from the
    middle of the rear corners to the middle of the front corners [deg]."""
    d = dev.set_index("Corner")

    def ang(x, y):
        fx, fy = (x["FL"] + x["FR"]) / 2.0, (y["FL"] + y["FR"]) / 2.0
        rx, ry = (x["RL"] + x["RR"]) / 2.0, (y["RL"] + y["RR"]) / 2.0
        return np.degrees(np.arctan2(fy - ry, fx - rx))

    nx, ny = d["Nominal X [mm]"], d["Nominal Y [mm]"]
    rot = (ang(nx + d["X [mm]"], ny + d["Y [mm]"]) - ang(nx, ny) + 180.0) % 360.0 - 180.0
    return {
        "centroid_x_mm": float(d["X [mm]"].mean()),
        "centroid_y_mm": float(d["Y [mm]"].mean()),
        "rotation_deg": float(rot),
        "max_abs_mm": float(np.max(np.abs(d[["X [mm]", "Y [mm]"]].to_numpy()))),
    }


def auto_magnification(max_dev_mm: float, tol_mm: float, choices=(10, 20, 40, 80, 160, 320)) -> int:
    """Largest factor that keeps the larger of the deviation and the tolerance under ~250 mm on the drawing."""
    ref = max(float(max_dev_mm), float(tol_mm), 1e-6)
    ok = [k for k in choices if k * ref <= 250.0]
    return int(max(ok) if ok else min(choices))
