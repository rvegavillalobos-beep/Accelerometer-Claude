# GOT Cluster Slip Simulator

Streamlit application that simulates the movement of a battery cell cluster resting on the GOT (lower housing)
— held in place by friction only — from conveyor IMU recordings exported by the WitMotion software.

It predicts **when** the cluster slips, **how far** it moves, **how much it rotates** and **where it pivots**.

## Features

- Reads WitMotion tab-separated exports (Bluetooth timestamp jitter, duplicate packets and sensor offset handled).
- Spike filter: isolated transmission glitches (a single sample of several g, typical at high sample rates) are
  detected against the median of the neighbouring samples and replaced by interpolation; real impacts, which last
  several samples, are kept. Optional hard limit in g. The removed samples are listed in the Signals tab.
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
- Whole route (Slip events and Friction sensitivity tabs): all uploaded recordings put in line order (ST 130 →
  buffer) and added up, as if one part had travelled all the measured segments: slip events of every recording,
  shift accumulated along the route, and the friction sweep for the whole route with the effective μ that explains
  the shift measured on the part at the end of the line.
- Line map tab: the plant layout, the conveyor path and the station positions are embedded in the app. Every
  uploaded recording (one or more) is drawn where its file name says it was measured: `St430-440` as the transport
  along the conveyors in the flow direction (ST 130 → buffer), `St981-Buffer` to the buffer, and a name with one
  station only (`St989`, `St540 deindex`, `St540Loop540`) as a ring at the station, because it is a process inside
  the station (indexing, deindexing, ...). Red where μs is exceeded, with the number of exceedances.
- Uploaded files survive a page reload: they are kept in server memory (max. 12 h) under a random id in the page
  URL (`?ws=...`), so reloading after a lost connection restores them.

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
| `motion.py` | Travel direction and motion phase of the carrier |
| `visuals.py` | Plotly figures and the animated top view |
| `whole_route.py` | Shifts of all recordings added up along the line, effective μ from a measured shift |
| `line_layout.py` | File names → line sections, shortest way along the conveyors, loops |
| `line_layout_data.py` | Embedded conveyor graph and station table (generated) |
| `assets/line_layout.png` | Layout background of the line map (generated) |
| `tools/build_line_layout.py` | Builds the two generated files from `assets/source/line_layout_paths.png` |

## Coordinate convention

Carrier frame: **X** along the cluster length (Front = +X), **Y** along the width (Left = +Y), **Z** up.
The sensor orientation and position on the workpiece carrier are set in the sidebar.

## Notes

- Measurement files are not part of this repository. Do not commit plant data to a public repository.
- At a sample rate of about 10 Hz, short impacts (10–50 ms) are under-sampled and slip is likely underestimated.
  Record at 100 Hz or more when possible.
- Results are model estimates and should be validated against video and physical measurements.

## Updating the line layout

The map is generated from `assets/source/line_layout_paths.png`: the layout with the conveyor path drawn in red,
white boxes with a red frame for station names that are hidden on the drawing (e.g. `ST395`), and the light
grey-blue box with the word *buffer*. Station label positions are listed in `STATION_LABELS` in
`tools/build_line_layout.py`, the flow direction in `FLOW_START`, `FLOW_END` and `RECIRCULATION` (loops that run
back upstream, such as the loop through ST 540). After changing either, run (Tesseract is needed to read the boxes):

```bash
pip install opencv-python-headless scikit-image scipy numpy pillow
python tools/build_line_layout.py
```

and commit `line_layout_data.py` and `assets/line_layout.png`.

