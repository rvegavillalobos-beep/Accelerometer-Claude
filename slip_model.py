"""
slip_model.py
=============

Planar stick-slip model of a rigid cell cluster resting on the GOT (lower
housing) and held in place by friction only.

Model summary
-------------
* The carrier (workpiece carrier + GOT) is a rigid body whose motion is known
  from the IMU: specific force f_S at the sensor, yaw rate w and yaw
  acceleration a (all in the carrier frame).
* The cluster is a rigid rectangle (length L, width W, mass m, yaw inertia I)
  with three relative degrees of freedom on the GOT: displacement (ux, uy) of
  its centre of mass and rotation phi.
* The contact is a set of discrete patches i carrying a fraction w_i of the
  normal load N = m * g0 * f_z (Coulomb friction, mu_s static / mu_k kinetic).
  Two layouts are available: full-area contact with a linear pressure
  distribution consistent with the CoM position, or four corner support pads
  with user-defined load shares.
* STICK: the friction wrench required to make the cluster follow the carrier
      F_req = m * f_P            (f_P = specific force at the cluster CoM)
      M_req = I * a
  is checked against the exact friction limit surface of the distributed
  contact (gauge function evaluated through its support function). The
  cluster sticks while  mu_req = gauge(F_req, M_req) / N  <=  mu_s.
* SLIP: the relative twist (vx, vy, Omega) is integrated with a linearly
  implicit (backward Euler) scheme. Every contact patch produces a kinetic
  friction force opposite to its own sliding velocity, so translation,
  rotation and pivoting about a heavily loaded corner emerge from the contact
  mechanics instead of being prescribed. A tiny velocity regularisation
  (eps_v) is used only while sliding; re-sticking is detected explicitly.

Coordinate frame: X along the cluster length ("Front" = +X), Y along the width
("Left" = +Y), Z up. Origin at the nominal cluster centre on the GOT.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data_io import G0, Kinematics

CORNER_NAMES = ["Front-Left", "Front-Right", "Rear-Right", "Rear-Left"]
CORNER_SIGNS = np.array([(+1, +1), (+1, -1), (-1, -1), (-1, +1)], dtype=float)


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ClusterParams:
    mass: float = 400.0  # [kg]
    length: float = 2.0  # [m] along carrier X
    width: float = 1.0  # [m] along carrier Y
    inertia: float | None = None  # yaw inertia about the CoM [kg m^2]; None -> uniform plate

    def yaw_inertia(self) -> float:
        if self.inertia and self.inertia > 0:
            return float(self.inertia)
        return self.mass * (self.length**2 + self.width**2) / 12.0

    def corners(self) -> np.ndarray:
        """Corner coordinates relative to the geometric centre (body frame), in CORNER_NAMES order."""
        return CORNER_SIGNS * np.array([self.length / 2, self.width / 2])


@dataclass(frozen=True)
class ContactParams:
    mu_s: float = 0.28
    mu_k: float = 0.28
    model: str = "full"  # "full" | "corners"
    com_offset: tuple[float, float] = (0.0, 0.0)  # full-area model: CoM offset from geometric centre [m]
    corner_shares: tuple[float, float, float, float] = (25.0, 25.0, 25.0, 25.0)  # FL, FR, RR, RL [%]
    corner_inset: float = 0.10  # corner model: pad centre inset from both edges [m]
    pad_size: float = 0.10  # corner model: square pad side [m]
    nx: int = 16  # full-area model: contact grid along X
    ny: int = 8  # full-area model: contact grid along Y
    corner_mu_factors: tuple[float, float, float, float] = (
        1.0,
        1.0,
        1.0,
        1.0,
    )  # friction multiplier per corner region FL, FR, RR, RL


@dataclass(frozen=True)
class SensorGeometry:
    """Sensor position on the carrier, relative to the nominal cluster centre [m] (carrier frame)."""

    x: float = 0.0
    y: float = 0.0


@dataclass(frozen=True)
class SolverParams:
    dt_sub: float = 1e-3  # integration step while sliding [s]
    eps_v: float = 1e-4  # friction regularisation velocity [m/s] (sliding phase only)
    v_stick: float = 1e-3  # re-stick velocity threshold [m/s]
    n_dirs: int = 2500  # directions used to evaluate the friction limit surface
    block: int = 1000  # samples scanned per vectorised stick check


# --------------------------------------------------------------------------- #
# Contact patch and friction limit surface
# --------------------------------------------------------------------------- #
@dataclass
class ContactPatch:
    g: np.ndarray  # (n, 2) patch positions relative to the geometric centre (body frame) [m]
    w: np.ndarray  # (n,) normal-load fractions (sum = 1)
    wf: np.ndarray  # (n,) friction-capacity weights = load fraction x regional friction factor
    com: np.ndarray  # (2,) CoM relative to the geometric centre [m]
    rho: np.ndarray  # (n, 2) patch positions relative to the CoM [m]
    ell: float  # characteristic length (radius of gyration of the footprint) [m]
    r_m: float  # friction-weighted mean lever arm sum(wf |rho|) [m]
    dirs: np.ndarray  # (K, 3) unit directions in (Fx, Fy, M/ell) space
    h: np.ndarray  # (K,) support function of the limit surface for mu = 1, N = 1
    notes: list[str] = field(default_factory=list)


def _fibonacci_hemisphere(k: int) -> np.ndarray:
    i = np.arange(k, dtype=float)
    z = 1.0 - (i + 0.5) / k  # (0, 1]
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    phi = i * np.pi * (3.0 - np.sqrt(5.0))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    # explicit equator ring (pure-force directions) for accuracy
    a = np.linspace(0, np.pi, 181)[:-1]
    ring = np.column_stack([np.cos(a), np.sin(a), np.zeros_like(a)])
    return np.vstack([pts, ring, [[0.0, 0.0, 1.0]]])


def build_patch(cluster: ClusterParams, contact: ContactParams, n_dirs: int = 2500) -> ContactPatch:
    L, W = cluster.length, cluster.width
    notes: list[str] = []
    if contact.model == "corners":
        inset = float(np.clip(contact.corner_inset, 0.0, min(L, W) / 2))
        half = 0.5 * float(np.clip(contact.pad_size, 1e-3, min(L, W)))
        shares = np.clip(np.asarray(contact.corner_shares, dtype=float), 0.0, None)
        if shares.sum() <= 0:
            shares = np.ones(4)
        shares = shares / shares.sum()
        centres = CORNER_SIGNS * np.array([L / 2 - inset, W / 2 - inset])
        offs = np.linspace(-half * 2 / 3, half * 2 / 3, 3)
        ox, oy = np.meshgrid(offs, offs)
        sub = np.column_stack([ox.ravel(), oy.ravel()])
        g = np.vstack([c + sub for c in centres])
        w = np.repeat(shares / len(sub), len(sub))
    else:
        nx, ny = max(int(contact.nx), 2), max(int(contact.ny), 2)
        xs = -L / 2 + (np.arange(nx) + 0.5) * L / nx
        ys = -W / 2 + (np.arange(ny) + 0.5) * W / ny
        gx, gy = np.meshgrid(xs, ys)
        g = np.column_stack([gx.ravel(), gy.ravel()])
        ex, ey = contact.com_offset
        var_x = np.mean(g[:, 0] ** 2)
        var_y = np.mean(g[:, 1] ** 2)
        p = 1.0 + ex * g[:, 0] / var_x + ey * g[:, 1] / var_y
        if p.min() < 0:
            notes.append(
                "The CoM offset lies outside the middle third of the footprint: part of the base loses "
                "contact. The pressure distribution was clipped at zero."
            )
            p = np.clip(p, 0.0, None)
        w = p / p.sum()

    com = (w[:, None] * g).sum(axis=0)
    rho = g - com
    ell = float(np.sqrt((L**2 + W**2) / 12.0))

    # regional friction factors (corner regions = quadrants of the footprint)
    fac = np.clip(np.asarray(contact.corner_mu_factors, dtype=float), 0.0, None)
    quad = np.where(g[:, 0] >= 0, np.where(g[:, 1] >= 0, 0, 1), np.where(g[:, 1] < 0, 2, 3))  # FL, FR, RR, RL
    wf = w * fac[quad]
    if not np.allclose(fac, 1.0):
        notes.append(
            "Non-uniform friction: corner-region factors "
            + ", ".join(f"{CORNER_NAMES[i]} x{fac[i]:g}" for i in range(4))
            + ". The required-μ values refer to the nominal μs."
        )
    if wf.sum() <= 0:
        raise ValueError("All friction factors are zero.")
    r_m = float(np.sum(wf * np.hypot(rho[:, 0], rho[:, 1])))

    dirs = _fibonacci_hemisphere(int(n_dirs))
    dm = dirs[:, 2:3] / ell
    vx = dirs[:, 0:1] - dm * rho[None, :, 1]
    vy = dirs[:, 1:2] + dm * rho[None, :, 0]
    h = (np.hypot(vx, vy) * wf[None, :]).sum(axis=1)

    return ContactPatch(g=g, w=w, wf=wf, com=com, rho=rho, ell=ell, r_m=r_m, dirs=dirs, h=h, notes=notes)


def gauge(patch: ContactPatch, fx: np.ndarray, fy: np.ndarray, m: np.ndarray, chunk: int = 2000) -> np.ndarray:
    """Gauge of the friction wrench (fx, fy, m) w.r.t. the unit limit surface (mu = 1, N = 1).

    Returns the friction coefficient required to transmit the wrench per unit
    normal load. Inputs are body-frame forces [N] and moment about the CoM [N m]
    already divided by the normal load.
    """
    fx = np.atleast_1d(np.asarray(fx, float))
    fy = np.atleast_1d(np.asarray(fy, float))
    m = np.atleast_1d(np.asarray(m, float))
    wr = np.column_stack([fx, fy, m / patch.ell]).astype(np.float32)
    out = np.empty(len(wr))
    # directions pre-scaled by 1/h: max_k |d_k . w| / h_k
    dT = (patch.dirs / patch.h[:, None]).T.astype(np.float32)
    for s in range(0, len(wr), chunk):
        blk = wr[s : s + chunk] @ dT
        out[s : s + chunk] = np.maximum(blk.max(axis=1), -blk.min(axis=1))
    return out


def boundary_wrench(patch: ContactPatch, twist: np.ndarray) -> np.ndarray:
    """Exact friction wrench (mu = 1, N = 1) opposing a unit relative twist (vx, vy, Omega)."""
    vx, vy, om = twist
    ux = vx - om * patch.rho[:, 1]
    uy = vy + om * patch.rho[:, 0]
    s = np.hypot(ux, uy)
    s = np.where(s < 1e-15, 1e-15, s)
    fx = -patch.wf * ux / s
    fy = -patch.wf * uy / s
    return np.array([fx.sum(), fy.sum(), np.sum(patch.rho[:, 0] * fy - patch.rho[:, 1] * fx)])


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
@dataclass
class SimulationResult:
    t: np.ndarray
    clock: np.ndarray
    ux: np.ndarray  # CoM displacement relative to the GOT [m], carrier frame
    uy: np.ndarray
    phi: np.ndarray  # cluster rotation relative to the GOT [rad]
    vx: np.ndarray
    vy: np.ndarray
    omega_rel: np.ndarray  # [rad/s]
    sliding: np.ndarray  # bool
    mu_req: np.ndarray  # friction coefficient required to stick
    mu_trans: np.ndarray  # translational part |F| / N
    mu_rot: np.ndarray  # rotational part |M| / (N r_m)
    fh_com: np.ndarray  # horizontal specific force at the cluster CoM [g]
    fhx_com: np.ndarray
    fhy_com: np.ndarray
    normal: np.ndarray  # normal load [N]
    icr_x: (
        np.ndarray
    )  # instantaneous centre of rotation, body frame rel. to geometric centre [m] (NaN when not rotating)
    icr_y: np.ndarray
    corner_disp: np.ndarray  # (T, 4) corner displacement magnitudes [m]
    corner_xy: np.ndarray  # (T, 4, 2) corner positions in the carrier frame [m]
    mu_req_initial: np.ndarray  # mu_req evaluated with the cluster in its nominal position
    patch: ContactPatch
    cluster: ClusterParams
    contact: ContactParams
    sensor: SensorGeometry
    events: pd.DataFrame
    runtime_s: float = 0.0
    truncated: bool = False

    @property
    def max_corner_disp(self) -> np.ndarray:
        return self.corner_disp.max(axis=1)


def _rot(phi: float) -> np.ndarray:
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s], [s, c]])


def simulate(
    kin: Kinematics,
    cluster: ClusterParams,
    contact: ContactParams,
    sensor: SensorGeometry,
    solver: SolverParams = SolverParams(),
    max_slip_seconds: float = 300.0,
) -> SimulationResult:
    """Run the stick-slip simulation over the whole kinematics record."""
    import time

    t_start_wall = time.perf_counter()
    patch = build_patch(cluster, contact, solver.n_dirs)
    m = cluster.mass
    inertia = cluster.yaw_inertia()
    mu_s, mu_k = float(contact.mu_s), float(contact.mu_k)
    S = np.array([sensor.x, sensor.y], dtype=float)
    com0 = patch.com.copy()

    t = kin.t
    T = len(t)
    fx, fy, fz, wz, al = kin.fx, kin.fy, kin.fz, kin.wz, kin.alpha
    normal = m * G0 * np.clip(fz, 0.0, None)

    # outputs
    ux = np.zeros(T)
    uy = np.zeros(T)
    phi = np.zeros(T)
    vx = np.zeros(T)
    vy = np.zeros(T)
    om = np.zeros(T)
    sliding = np.zeros(T, dtype=bool)
    mu_req = np.zeros(T)
    mu_tr = np.zeros(T)
    mu_rt = np.zeros(T)
    fhx = np.zeros(T)
    fhy = np.zeros(T)
    icr_x = np.full(T, np.nan)
    icr_y = np.full(T, np.nan)

    def demand(idx_or_vals, u_vec, ph):
        """Friction demand (body frame) for given inputs and cluster pose."""
        fxs, fys, wzs, als, Ns = idx_or_vals
        P = com0 + u_vec
        r = P - S
        apx = G0 * fxs + als * (-r[1]) - wzs**2 * r[0]
        apy = G0 * fys + als * (r[0]) - wzs**2 * r[1]
        c, s = np.cos(ph), np.sin(ph)
        Fbx = m * (c * apx + s * apy)
        Fby = m * (-s * apx + c * apy)
        Mreq = inertia * als
        with np.errstate(divide="ignore", invalid="ignore"):
            invN = np.where(Ns > 1e-9, 1.0 / np.maximum(Ns, 1e-9), np.inf)
        return apx, apy, Fbx, Fby, Mreq, invN

    def mu_parts(Fbx, Fby, Mreq, invN):
        with np.errstate(invalid="ignore"):
            g = gauge(
                patch,
                Fbx * np.where(np.isfinite(invN), invN, 0.0),
                Fby * np.where(np.isfinite(invN), invN, 0.0),
                Mreq * np.where(np.isfinite(invN), invN, 0.0),
            )
            g = np.where(np.isfinite(invN), g, np.inf)
            mt = np.hypot(Fbx, Fby) * invN
            mr = np.abs(Mreq) * invN / patch.r_m
        return g, mt, mr

    # mu_req with the cluster in nominal position for the whole record (diagnostic)
    apx0, apy0, Fbx0, Fby0, M0, invN0 = demand((fx, fy, wz, al, normal), np.zeros(2), 0.0)
    mu_req_initial, _, _ = mu_parts(Fbx0, Fby0, M0, invN0)

    # state
    u = np.zeros(2)
    ph = 0.0
    xi = np.zeros(3)  # vx, vy, Omega (relative, carrier frame)
    is_sliding = False
    no_restick_before = -np.inf
    slip_time = 0.0
    truncated = False

    nsub = max(1, int(round(kin.dt / solver.dt_sub)))
    dts = kin.dt / nsub
    Mdiag = np.array([m, m, inertia])
    eps2 = solver.eps_v**2
    rho_b = patch.rho
    wts = patch.wf
    ell = patch.ell

    k = 0
    block = solver.block
    while k < T:
        if not is_sliding:
            # ---- vectorised stick scan over a block (short right after a slip, then growing)
            k1 = min(T, k + block)
            sl = slice(k, k1)
            apx, apy, Fbx, Fby, Mr, invN = demand((fx[sl], fy[sl], wz[sl], al[sl], normal[sl]), u, ph)
            g, mt, mr = mu_parts(Fbx, Fby, Mr, invN)
            exceed = np.where(g > mu_s)[0]
            stop = k1 if len(exceed) == 0 else k + int(exceed[0])
            n_ok = stop - k
            if n_ok > 0:
                ux[k:stop], uy[k:stop], phi[k:stop] = u[0], u[1], ph
                mu_req[k:stop], mu_tr[k:stop], mu_rt[k:stop] = g[:n_ok], mt[:n_ok], mr[:n_ok]
                fhx[k:stop], fhy[k:stop] = apx[:n_ok] / G0, apy[:n_ok] / G0
            if len(exceed) == 0:
                k = k1
                block = min(2 * block, 4 * solver.block)
                continue
            # slip onset between samples stop-1 and stop
            is_sliding = True
            no_restick_before = t[stop]
            k = max(stop - 1, 0)
            if stop == 0:
                # record the very first sample and integrate from it
                ux[0], uy[0], phi[0] = u[0], u[1], ph
                mu_req[0], mu_tr[0], mu_rt[0] = g[0], mt[0], mr[0]
                fhx[0], fhy[0] = apx[0] / G0, apy[0] / G0
            continue

        # ---- sliding: integrate from sample k to k+1
        if k >= T - 1:
            break
        if slip_time > max_slip_seconds:
            truncated = True
            # freeze the remaining record
            ux[k:], uy[k:], phi[k:] = u[0], u[1], ph
            sliding[k:] = True
            break
        restuck = False
        for j in range(nsub):
            s_frac = (j + 1) / nsub
            fxs = fx[k] + s_frac * (fx[k + 1] - fx[k])
            fys = fy[k] + s_frac * (fy[k + 1] - fy[k])
            wzs = wz[k] + s_frac * (wz[k + 1] - wz[k])
            als = al[k] + s_frac * (al[k + 1] - al[k])
            Ns = normal[k] + s_frac * (normal[k + 1] - normal[k])
            ts = t[k] + s_frac * kin.dt

            c, s = np.cos(ph), np.sin(ph)
            rx = c * rho_b[:, 0] - s * rho_b[:, 1]
            ry = s * rho_b[:, 0] + c * rho_b[:, 1]
            P = com0 + u
            r = P - S
            apx = G0 * fxs + als * (-r[1]) - wzs**2 * r[0]
            apy = G0 * fys + als * (r[0]) - wzs**2 * r[1]

            vxs, vys, oms = xi
            pux = vxs - oms * ry
            puy = vys + oms * rx
            s2 = pux * pux + puy * puy + eps2
            sn = np.sqrt(s2)
            muN = mu_k * Ns
            coef = muN * wts / sn
            Ffx = -np.sum(coef * pux)
            Ffy = -np.sum(coef * puy)
            Mf = -np.sum(coef * (rx * puy - ry * pux))
            q = coef / s2  # = mu N w / s^3
            d11 = q * (s2 - pux * pux)
            d12 = -q * pux * puy
            d22 = q * (s2 - puy * puy)
            A11 = d11.sum()
            A12 = d12.sum()
            A22 = d22.sum()
            A13 = np.sum(-ry * d11 + rx * d12)
            A23 = np.sum(-ry * d12 + rx * d22)
            A33 = np.sum(ry * ry * d11 - 2.0 * rx * ry * d12 + rx * rx * d22)
            Amat = np.array(
                [
                    [Mdiag[0] + dts * A11, dts * A12, dts * A13],
                    [dts * A12, Mdiag[1] + dts * A22, dts * A23],
                    [dts * A13, dts * A23, Mdiag[2] + dts * A33],
                ]
            )
            Fex = -m * apx + 2.0 * m * wzs * vys  # Coriolis: -2 m w z x v
            Fey = -m * apy - 2.0 * m * wzs * vxs
            Me = -inertia * als
            rhs = dts * np.array([Ffx + Fex, Ffy + Fey, Mf + Me])
            xi = xi + np.linalg.solve(Amat, rhs)
            u = u + dts * xi[:2]
            ph = ph + dts * xi[2]
            slip_time += dts

            # re-stick check
            if (
                ts >= no_restick_before
                and np.hypot(xi[0], xi[1]) < solver.v_stick
                and abs(xi[2]) * ell < solver.v_stick
            ):
                _, _, Fbx, Fby, Mr, invN = demand((fxs, fys, wzs, als, Ns), u, ph)
                gs, _, _ = mu_parts(np.atleast_1d(Fbx), np.atleast_1d(Fby), np.atleast_1d(Mr), np.atleast_1d(invN))
                if gs[0] <= mu_s:
                    xi = np.zeros(3)
                    restuck = True
                    break

        # store sample k+1
        kk = k + 1
        ux[kk], uy[kk], phi[kk] = u[0], u[1], ph
        vx[kk], vy[kk], om[kk] = xi
        apx, apy, Fbx, Fby, Mr, invN = demand((fx[kk], fy[kk], wz[kk], al[kk], normal[kk]), u, ph)
        gk, mtk, mrk = mu_parts(np.atleast_1d(Fbx), np.atleast_1d(Fby), np.atleast_1d(Mr), np.atleast_1d(invN))
        mu_req[kk], mu_tr[kk], mu_rt[kk] = gk[0], mtk[0], mrk[0]
        fhx[kk], fhy[kk] = apx / G0, apy / G0
        if not restuck:
            sliding[kk] = True
            if abs(xi[2]) > 1e-6:
                # ICR in carrier frame, then body frame relative to geometric centre
                icr_c = (com0 + u) + np.array([-xi[1] / xi[2], xi[0] / xi[2]])
                Rb = _rot(-ph)
                icr_b = Rb @ (icr_c - (com0 + u)) + com0
                icr_x[kk], icr_y[kk] = icr_b
        else:
            sliding[kk] = True  # slip ended within this interval
            is_sliding = False
            block = max(50, solver.block // 10)
            k = kk + 1
            # the next sample starts a new stick scan; keep its pose consistent
            continue
        k = kk

    # corners
    corners_b = cluster.corners()  # rel. geometric centre
    cphi, sphi = np.cos(phi), np.sin(phi)
    rel = corners_b - com0  # (4,2)
    cx = (com0[0] + ux)[:, None] + cphi[:, None] * rel[None, :, 0] - sphi[:, None] * rel[None, :, 1]
    cy = (com0[1] + uy)[:, None] + sphi[:, None] * rel[None, :, 0] + cphi[:, None] * rel[None, :, 1]
    corner_xy = np.stack([cx, cy], axis=-1)
    corner_disp = np.hypot(cx - corners_b[None, :, 0], cy - corners_b[None, :, 1])

    res = SimulationResult(
        t=t,
        clock=kin.clock,
        ux=ux,
        uy=uy,
        phi=phi,
        vx=vx,
        vy=vy,
        omega_rel=om,
        sliding=sliding,
        mu_req=mu_req,
        mu_trans=mu_tr,
        mu_rot=mu_rt,
        fh_com=np.hypot(fhx, fhy),
        fhx_com=fhx,
        fhy_com=fhy,
        normal=normal,
        icr_x=icr_x,
        icr_y=icr_y,
        corner_disp=corner_disp,
        corner_xy=corner_xy,
        mu_req_initial=mu_req_initial,
        patch=patch,
        cluster=cluster,
        contact=contact,
        sensor=sensor,
        events=pd.DataFrame(),
        truncated=truncated,
    )
    res.events = extract_events(res, kin)
    res.runtime_s = time.perf_counter() - t_start_wall
    return res


# --------------------------------------------------------------------------- #
# Post-processing
# --------------------------------------------------------------------------- #
def _segments(mask: np.ndarray, merge_gap: int = 0) -> list[tuple[int, int]]:
    """Contiguous True segments [i0, i1] (inclusive), merging gaps <= merge_gap samples."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    segs = []
    start = prev = idx[0]
    for i in idx[1:]:
        if i - prev > merge_gap + 1:
            segs.append((start, prev))
            start = i
        prev = i
    segs.append((start, prev))
    return segs


def is_turning(kin: Kinematics, i0: int, i1: int, pad_s: float = 0.5, threshold_deg_s: float = 2.0) -> bool:
    """True when the carrier is rotating steadily (turntable), not just jolted by an impact."""
    n = int(round(pad_s * kin.fs))
    a, b = max(0, i0 - n), min(len(kin.wz), i1 + n + 1)
    return bool(np.median(np.abs(kin.wz[a:b])) > np.deg2rad(threshold_deg_s))


def classify_pivot(cluster: ClusterParams, x: float, y: float) -> str:
    if not (np.isfinite(x) and np.isfinite(y)):
        return "-"
    corners = cluster.corners()
    d = np.hypot(corners[:, 0] - x, corners[:, 1] - y)
    i = int(np.argmin(d))
    tol = 0.25 * min(cluster.length, cluster.width)
    if d[i] <= tol:
        return f"Near {CORNER_NAMES[i]} corner"
    if abs(x) <= cluster.length / 2 and abs(y) <= cluster.width / 2:
        if np.hypot(x, y) <= tol:
            return "Near centre"
        return "Inside footprint"
    return "Outside footprint"


def extract_events(res: SimulationResult, kin: Kinematics) -> pd.DataFrame:
    """Build the slip-event table."""
    merge = max(1, int(round(0.05 * kin.fs)))
    segs = _segments(res.sliding, merge_gap=merge)
    rows = []
    cl = res.cluster
    for n, (i0, i1) in enumerate(segs, start=1):
        a = max(i0 - 1, 0)
        b = i1
        pk = a + int(np.nanargmax(np.where(np.isfinite(res.mu_req[a : b + 1]), res.mu_req[a : b + 1], -1)))
        dux = (res.ux[b] - res.ux[a]) * 1e3
        duy = (res.uy[b] - res.uy[a]) * 1e3
        dphi = np.rad2deg(res.phi[b] - res.phi[a])
        corner_shift = (
            np.max(
                np.hypot(
                    res.corner_xy[b, :, 0] - res.corner_xy[a, :, 0], res.corner_xy[b, :, 1] - res.corner_xy[a, :, 1]
                )
            )
            * 1e3
        )
        mt, mr = res.mu_trans[pk], res.mu_rot[pk]
        if mr > 1.5 * mt:
            driver = "Rotation (yaw acceleration)"
        elif mt > 1.5 * mr:
            driver = "Translation (linear acceleration)"
        else:
            driver = "Combined"
        turning = is_turning(kin, a, b)
        # pivot: rotation-weighted median of the ICR while the cluster is rotating
        rot_share = abs(np.deg2rad(dphi)) * res.patch.ell * 1e3
        lin = np.hypot(dux, duy)
        ix = res.icr_x[a : b + 1]
        iy = res.icr_y[a : b + 1]
        wgt = np.abs(res.omega_rel[a : b + 1])
        ok = np.isfinite(ix) & np.isfinite(iy) & (wgt > 0)
        if ok.sum() > 0 and rot_share > 0.3 * max(lin, 1e-9) and rot_share > 0.05:
            order = np.argsort(ix[ok])
            cw = np.cumsum(wgt[ok][order])
            px = ix[ok][order][np.searchsorted(cw, 0.5 * cw[-1])]
            order = np.argsort(iy[ok])
            cw = np.cumsum(wgt[ok][order])
            py = iy[ok][order][np.searchsorted(cw, 0.5 * cw[-1])]
            pivot = classify_pivot(cl, px, py)
        else:
            px = py = np.nan
            pivot = "None (sliding)"
        rows.append(
            {
                "Event": n,
                "Start clock": pd.Timestamp(res.clock[a]).strftime("%H:%M:%S.%f")[:-3],
                "Start [s]": round(float(res.t[a]), 2),
                "End [s]": round(float(res.t[b]), 2),
                "Duration [s]": round(float(res.t[b] - res.t[a]), 3),
                "Peak mu required": round(float(res.mu_req[pk]), 3) if np.isfinite(res.mu_req[pk]) else np.inf,
                "Driver": driver,
                "Carrier motion": "Turning" if turning else "Straight",
                "Peak horiz. accel. at CoM [g]": round(float(np.max(res.fh_com[a : b + 1])), 3),
                "Peak yaw accel. [deg/s2]": round(float(np.rad2deg(np.max(np.abs(kin.alpha[a : b + 1])))), 1),
                "Min. vertical [g]": round(float(np.min(kin.fz[a : b + 1])), 3),
                "dX [mm]": round(dux, 2),
                "dY [mm]": round(duy, 2),
                "dRotation [deg]": round(dphi, 3),
                "Max corner shift [mm]": round(corner_shift, 2),
                "Pivot": pivot,
                "Pivot X [m]": round(float(px), 3) if np.isfinite(px) else np.nan,
                "Pivot Y [m]": round(float(py), 3) if np.isfinite(py) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summary(res: SimulationResult, kin: Kinematics) -> dict:
    T = len(res.t)
    slip_time = float(np.sum(res.sliding)) * kin.dt
    mu_noslip = float(np.nanmax(np.where(np.isfinite(res.mu_req_initial), res.mu_req_initial, np.nan))) if T else np.nan
    return {
        "duration_s": kin.duration,
        "n_events": len(res.events),
        "slip_time_s": slip_time,
        "mu_noslip": mu_noslip,
        "margin": res.contact.mu_s / mu_noslip if mu_noslip > 0 else np.inf,
        "final_dx_mm": float(res.ux[-1] * 1e3),
        "final_dy_mm": float(res.uy[-1] * 1e3),
        "final_rot_deg": float(np.rad2deg(res.phi[-1])),
        "final_corner_mm": float(res.max_corner_disp[-1] * 1e3),
        "peak_corner_mm": float(res.max_corner_disp.max() * 1e3),
        "peak_fh_g": float(res.fh_com.max()),
        "min_fz_g": float(kin.fz.min()),
        "peak_yaw_rate_deg_s": float(np.rad2deg(np.max(np.abs(kin.wz)))),
        "peak_yaw_acc_deg_s2": float(np.rad2deg(np.max(np.abs(kin.alpha)))),
    }


def results_table(res: SimulationResult) -> pd.DataFrame:
    cols = {
        "time_s": res.t,
        "clock": res.clock,
        "sliding": res.sliding.astype(int),
        "mu_required": res.mu_req,
        "mu_translational": res.mu_trans,
        "mu_rotational": res.mu_rot,
        "fh_com_g": res.fh_com,
        "fx_com_g": res.fhx_com,
        "fy_com_g": res.fhy_com,
        "normal_N": res.normal,
        "com_dx_mm": res.ux * 1e3,
        "com_dy_mm": res.uy * 1e3,
        "rotation_deg": np.rad2deg(res.phi),
        "max_corner_disp_mm": res.max_corner_disp * 1e3,
    }
    for i, name in enumerate(CORNER_NAMES):
        cols[f"{name.lower()}_disp_mm"] = res.corner_disp[:, i] * 1e3
    return pd.DataFrame(cols)


def friction_sweep(
    kin: Kinematics,
    cluster: ClusterParams,
    contact: ContactParams,
    sensor: SensorGeometry,
    mus: list[float],
    solver: SolverParams = SolverParams(),
    keep_ratio: bool = True,
    max_slip_seconds: float = 120.0,
) -> pd.DataFrame:
    """Repeat the simulation for several static friction coefficients."""
    ratio = contact.mu_k / contact.mu_s if contact.mu_s > 0 else 1.0
    rows = []
    for mu in mus:
        c = ContactParams(
            mu_s=mu,
            mu_k=mu * ratio if keep_ratio else min(contact.mu_k, mu),
            model=contact.model,
            com_offset=contact.com_offset,
            corner_shares=contact.corner_shares,
            corner_inset=contact.corner_inset,
            pad_size=contact.pad_size,
            nx=contact.nx,
            ny=contact.ny,
            corner_mu_factors=contact.corner_mu_factors,
        )
        r = simulate(kin, cluster, c, sensor, solver, max_slip_seconds=max_slip_seconds)
        sm = summary(r, kin)
        rows.append(
            {
                "mu_s": mu,
                "mu_k": c.mu_k,
                "Slip events": sm["n_events"],
                "Slip time [s]": sm["slip_time_s"],
                "Final max corner shift [mm]": sm["final_corner_mm"],
                "Peak corner shift [mm]": sm["peak_corner_mm"],
                "Final rotation [deg]": sm["final_rot_deg"],
                "Final CoM shift [mm]": float(np.hypot(sm["final_dx_mm"], sm["final_dy_mm"])),
                "Truncated": r.truncated,
            }
        )
    return pd.DataFrame(rows)


def critical_moments(res: SimulationResult, kin: Kinematics, n: int = 10, min_sep_s: float = 2.0) -> pd.DataFrame:
    """Moments with the highest friction demand (cluster in nominal position), slip or not."""
    from scipy.signal import find_peaks

    mu = np.where(np.isfinite(res.mu_req_initial), res.mu_req_initial, 10.0)
    dist = max(1, int(round(min_sep_s * kin.fs)))
    peaks, _ = find_peaks(mu, distance=dist)
    if len(peaks) == 0:
        peaks = np.array([int(np.argmax(mu))])
    peaks = peaks[np.argsort(-mu[peaks])][:n]
    rows = []
    for i in peaks:
        mt, mr = res.mu_trans[i], res.mu_rot[i]
        if mr > 1.5 * mt:
            driver = "Rotation (yaw acceleration)"
        elif mt > 1.5 * mr:
            driver = "Translation (linear acceleration)"
        else:
            driver = "Combined"
        rows.append(
            {
                "Clock": pd.Timestamp(res.clock[i]).strftime("%H:%M:%S.%f")[:-3],
                "Time [s]": round(float(res.t[i]), 2),
                "Required mu": round(float(mu[i]), 3),
                "Margin vs mu_s": f"{(res.contact.mu_s / mu[i] - 1) * 100:+.0f}%" if mu[i] > 0 else "-",
                "Driver": driver,
                "Carrier motion": "Turning" if is_turning(kin, i, i) else "Straight",
                "Horiz. accel. at CoM [g]": round(float(res.fh_com[i]), 3),
                "Vertical [g]": round(float(kin.fz[i]), 3),
                "Yaw accel. [deg/s2]": round(float(np.rad2deg(kin.alpha[i])), 1),
                "Predicted slip": "Yes" if res.sliding[max(i - 1, 0) : i + 2].any() else "No",
            }
        )
    return pd.DataFrame(rows)
