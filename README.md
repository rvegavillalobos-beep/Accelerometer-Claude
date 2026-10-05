# GOT Cluster Slip Simulator

Streamlit application that simulates the movement of a battery cell cluster resting on the GOT (lower housing)
— held in place by friction only — from conveyor IMU recordings exported by the WitMotion software.

It predicts **when** the cluster slips, **how far** it moves, **how much it rotates** and **where it pivots**.

## Features

- Reads WitMotion tab-separated exports (Bluetooth timestamp jitter, duplicate packets and sensor offset handled).
- Transfers the carrier motion to the cluster centre, including turntable rotations (centripetal and tangential terms).
- Planar stick-slip model with a distributed Coulomb friction contact (full-area or four corner supports,
  optional non-uniform friction per corner region).
- Turntable detection from the gyroscope: angle, fast / slow phases, spin-up and braking, friction demand at
  entry, rotation and exit, and an estimate of the turntable axis position relative to the sensor.
- Animated top view (carrier or plant frame), friction demand vs. capacity, slip-event table with pivot location.
- Most critical moments, video synchronisation, friction sensitivity analysis and comparison of several recordings.
- CSV / JSON export of all results.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

Push the repository to GitHub, create a new app and select `app.py` as the entry point.
`requirements.txt` is installed automatically.

## Files

| File | Content |
|---|---|
| `app.py` | Streamlit user interface |
| `data_io.py` | WitMotion parser and signal pre-processing |
| `slip_model.py` | Friction contact model, stick-slip simulation, event extraction |
| `turns.py` | Turntable / rotation detection and analysis |
| `visuals.py` | Plotly figures and the animated top view |

## Coordinate convention

Carrier frame: **X** along the cluster length (Front = +X), **Y** along the width (Left = +Y), **Z** up.
The sensor orientation and position on the workpiece carrier are set in the sidebar.

## Notes

- Measurement files are not part of this repository. Do not commit plant data to a public repository.
- At a sample rate of about 10 Hz, short impacts (10–50 ms) are under-sampled and slip is likely underestimated.
  Record at 100 Hz or more when possible.
- Results are model estimates and should be validated against video and physical measurements.
