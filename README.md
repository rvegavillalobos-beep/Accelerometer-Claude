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
  entry, rotation and exit, and an estimate of the turntable axis position relative to the sensor. Friction demand
  (required μ × vertical acceleration, including the rotational load) is shown right below the yaw rate against
  the dynamic friction limit μs·f_z.
- Turntable comparison: pick a turn in one recording and a turn in another (or the same) recording and overlay
  yaw rate, friction demand vs. dynamic limit and yaw acceleration, aligned at the turn start or end.
- Dynamic friction limit μs·f_z: the friction capacity drops whenever the carrier bumps (vertical acceleration < 1 g).
- Speed profile of the whole route (travel speed, start / stop ramps, friction demand vs. dynamic friction limit).
- Travel direction of the carrier per motion segment (auto-detected, manually editable) and a motion phase for
  every critical moment (accelerating, braking / stop, lateral load, stopped, rotating on turntable).
- Animated top view (carrier or plant frame) with travel-direction arrow, friction demand vs. capacity,
  slip-event table with pivot location.
- Measure tab: pick a zoom window on the whole-record strip, then measure an interval A–B at full resolution
  (recorded samples shown, A and B adjustable to the millisecond) with Δt, clock times, distance, mean speed,
  carrier rotation, maximum friction demand and slip.
- Most critical moments, video synchronisation, friction sensitivity analysis and comparison of several recordings
  (highest required μ per recording, plus every distinct exceedance of μs marked at its peak and counted).
- Line layout map (Compare tab): upload the plant layout once; station labels are read with OCR, missing ones are
  placed with one click, and every recording is drawn as the section in its file name (`St430-440`, `St989`,
  `St981-Buffer`), red where μs is exceeded with the number of exceedances. The layout map is saved as one
  `.json` file for the next session.
- CSV / JSON export of all results.
- Uploaded files survive a page reload: they are kept in server memory (max. 12 h) under a random id in the page
  URL (`?ws=...`), so reloading after a lost connection restores them.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

Push the repository to GitHub, create a new app and select `app.py` as the entry point.
`requirements.txt` and `packages.txt` (Tesseract OCR for the line layout map) are installed automatically.
Locally, install Tesseract with your package manager (e.g. `apt install tesseract-ocr`); without it the layout map
still works, stations are then placed by hand.

## Files

| File | Content |
|---|---|
| `app.py` | Streamlit user interface |
| `data_io.py` | WitMotion parser and signal pre-processing |
| `slip_model.py` | Friction contact model, stick-slip simulation, event extraction |
| `turns.py` | Turntable / rotation detection and analysis |
| `motion.py` | Travel direction, motion phase and speed profile |
| `line_map.py` | Layout OCR, station positions and line sections from file names |
| `visuals.py` | Plotly figures and the animated top view |

## Coordinate convention

Carrier frame: **X** along the cluster length (Front = +X), **Y** along the width (Left = +Y), **Z** up.
The sensor orientation and position on the workpiece carrier are set in the sidebar.

## Notes

- Measurement files and plant layouts are not part of this repository. Do not commit plant data to a public repository.
- At a sample rate of about 10 Hz, short impacts (10–50 ms) are under-sampled and slip is likely underestimated.
  Record at 100 Hz or more when possible.
- Results are model estimates and should be validated against video and physical measurements.
