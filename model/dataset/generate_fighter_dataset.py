"""
Synthetic Fighter Sortie Dataset Generator (default 10 Hz)
----------------------------------------------------------
Outputs time-series CSVs with coarse regime labels (no geometry).

Columns:
  time_s, altitude_ft, vertical_speed_fpm, ias_kt, pitch_deg, roll_deg, heading_deg,
  throttle_pct, afterburner_on, gear_down, flaps_deg, speedbrake_pct, weight_on_wheels, label

Regimes (COARSE):
  Taxi, Takeoff, Climb, Level, Cruise, Turn Left, Turn Right, Descent, Landing

Files & Splits:
  dataset/fighter_regimes_synth/train/sortie_001.csv ...
  dataset/fighter_regimes_synth/test/sortie_001.csv  ...

Usage:
  python generate_fighter_dataset.py --out_root ./dataset/fighter_regimes_synth \
    --train 20 --test 10 --hz 10 --seed-train 42 --seed-test 777
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd

# --------------------- Config & helpers ---------------------
@dataclass
class Ranges:
    taxi_speed: tuple = (6, 24)                # kt
    takeoff_vr: tuple = (150, 180)             # kt
    approach_speed: tuple = (150, 190)         # kt
    descent_vs_fpm: tuple = (-5000, -2000)     # fpm
    cruise_speed: tuple = (280, 360)           # kt
    pitch_rot_dps: tuple = (6, 12)             # deg/s
    bank_turn_deg: tuple = (15, 45)            # deg
    hdg_rate_turn_dps: tuple = (2.0, 5.0)      # deg/s

LABELS = [
    "Taxi","Takeoff","Climb","Level","Cruise","Turn Left","Turn Right","Descent","Landing"
]
R = Ranges()

# globals set by CLI
HZ = 10.0
DT = 1.0 / HZ

def set_rate(hz: float):
    global HZ, DT
    HZ = float(hz); DT = 1.0 / HZ

def _r(rng, low, high): return rng.uniform(low, high)
def _noise(rng, n, sd): return rng.normal(0.0, sd, n)
def _hold(n, val): return np.full(n, float(val))
def _ramp(n, a, b): return np.linspace(float(a), float(b), int(n))
def _npoints(seconds: float): return max(1, int(round(seconds * HZ)))

def _append(buf: Dict[str, List], seg: Dict[str, np.ndarray], t0: float, alt0: float, hdg0: float):
    """Append one segment, offsetting time/altitude/heading to ensure continuity."""
    n = len(seg["ias_kt"])
    time = np.arange(n) * DT + t0
    buf["time_s"].extend(time.tolist())

    alt = seg["altitude_rel_ft"] + alt0
    buf["altitude_ft"].extend(alt.tolist())
    vs = np.r_[0, np.diff(alt)] * HZ * 60.0
    buf["vertical_speed_fpm"].extend(vs.tolist())

    hdg = (seg["heading_rel_deg"] + hdg0 + 360.0) % 360.0
    buf["heading_deg"].extend(hdg.tolist())

    for k in ["ias_kt","pitch_deg","roll_deg","throttle_pct","afterburner_on",
              "gear_down","flaps_deg","speedbrake_pct","weight_on_wheels","label"]:
        buf[k].extend(seg[k].tolist())

    return time[-1] + DT, float(alt[-1]), float(hdg[-1])

def _new_buf():
    return {k: [] for k in ["time_s","altitude_ft","vertical_speed_fpm","ias_kt","pitch_deg","roll_deg",
                            "heading_deg","throttle_pct","afterburner_on","gear_down",
                            "flaps_deg","speedbrake_pct","weight_on_wheels","label"]}

# --------------------- Segment generators (relative) ---------------------
def seg_taxi(rng, length_s=60):
    n = _npoints(length_s)
    ias = _hold(n, _r(rng, *R.taxi_speed)) + _noise(rng, n, 0.6)
    thr = _hold(n, _r(rng, 10, 25)) + _noise(rng, n, 0.5)
    pitch = _hold(n, _r(rng, -1.0, 2.0)) + _noise(rng, n, 0.2)
    roll  = _hold(n, 0.0) + _noise(rng, n, 0.5)
    flaps = _hold(n, 0.0); spbrk = _hold(n, 0.0)
    gear  = _hold(n, 1.0); wow = _hold(n, 1.0); ab = _hold(n, 0.0)
    alt_rel = _hold(n, 0.0) + _noise(rng, n, 0.1)
    hdg_rel = _hold(n, 0.0) + _noise(rng, n, 0.1)
    return dict(altitude_rel_ft=alt_rel, heading_rel_deg=hdg_rel, ias_kt=ias,
                pitch_deg=pitch, roll_deg=roll, throttle_pct=thr, afterburner_on=ab,
                gear_down=gear, flaps_deg=flaps, speedbrake_pct=spbrk, weight_on_wheels=wow,
                label=np.array(["Taxi"]*n, dtype=object))

def seg_takeoff_roll(rng, length_s=25, start_speed=None, use_ab=None):
    n = _npoints(length_s)
    v0 = start_speed if start_speed is not None else _r(rng, 10, 25)
    vr = _r(rng, *R.takeoff_vr)
    ias = _ramp(n, v0, vr) + _noise(rng, n, 0.8)
    thr = _hold(n, _r(rng, 88, 100)) + _noise(rng, n, 0.4)
    ab  = _hold(n, 1.0 if (use_ab if use_ab is not None else rng.random() < 0.5) else 0.0)
    pitch = _hold(n, _r(rng, 0.0, 2.0))
    roll  = _hold(n, _r(rng, -1.5, 1.5))
    flaps = _hold(n, _r(rng, 8, 15))
    spbrk = _hold(n, 0.0); gear  = _hold(n, 1.0); wow = _hold(n, 1.0)
    alt_rel = _hold(n, 0.0); hdg_rel = _hold(n, 0.0)
    return dict(altitude_rel_ft=alt_rel, heading_rel_deg=hdg_rel, ias_kt=ias,
                pitch_deg=pitch, roll_deg=roll, throttle_pct=thr, afterburner_on=ab,
                gear_down=gear, flaps_deg=flaps, speedbrake_pct=spbrk, weight_on_wheels=wow,
                label=np.array(["Takeoff"]*n, dtype=object))

def seg_rotation(rng, length_s=3):
    n = _npoints(length_s)
    pitch_rate = _hold(n, _r(rng, *R.pitch_rot_dps))
    pitch = np.cumsum(pitch_rate)*DT + _r(rng, 2, 5)
    pitch = np.clip(pitch, -5, 18)
    alt_rel = _ramp(n, 0.0, _r(rng, 80, 180)) + _noise(rng, n, 1.0)
    ias = _hold(n, _r(rng, *R.takeoff_vr)) + _noise(rng, n, 1.0)
    thr = _hold(n, _r(rng, 90, 100)); roll = _hold(n, _r(rng, -2, 2))
    flaps = _hold(n, _r(rng, 8, 15)); spbrk = _hold(n, 0.0)
    gear  = _ramp(n, 1.0, 0.0); wow = _hold(n, 0.0); ab = _hold(n, 0.0)
    hdg_rel = _hold(n, 0.0)
    return dict(altitude_rel_ft=alt_rel, heading_rel_deg=hdg_rel, ias_kt=ias,
                pitch_deg=pitch, roll_deg=roll, throttle_pct=thr, afterburner_on=ab,
                gear_down=gear, flaps_deg=flaps, speedbrake_pct=spbrk, weight_on_wheels=wow,
                label=np.array(["Takeoff"]*n, dtype=object))

def seg_initial_climb(rng, length_s=25):
    n = _npoints(length_s)
    target = _r(rng, 1200, 2000)
    alt_rel = _ramp(n, _r(rng, 150, 250), target) + _noise(rng, n, 2.0)
    ias = _ramp(n, _r(rng, 170, 230), _r(rng, 240, 300)) + _noise(rng, n, 1.5)
    thr = _hold(n, _r(rng, 85, 95)) + _noise(rng, n, 0.5)
    pitch = _hold(n, _r(rng, 8, 15)) + _noise(rng, n, 0.5)
    roll  = _hold(n, _r(rng, -3, 3))
    flaps = _ramp(n, _r(rng, 10, 15), 0.0)
    spbrk = _hold(n, 0.0); gear=_hold(n,0.0); wow=_hold(n,0.0); ab=_hold(n,0.0)
    hdg_rel = _hold(n, 0.0)
    return dict(altitude_rel_ft=alt_rel, heading_rel_deg=hdg_rel, ias_kt=ias,
                pitch_deg=pitch, roll_deg=roll, throttle_pct=thr, afterburner_on=ab,
                gear_down=gear, flaps_deg=flaps, speedbrake_pct=spbrk, weight_on_wheels=wow,
                label=np.array(["Takeoff"]*n, dtype=object))

def seg_climb(rng, length_s=120):
    n = _npoints(length_s)
    gain = _r(rng, 15000, 26000)
    alt_rel = _ramp(n, 0.0, gain) + _noise(rng, n, 3.0)
    ias = _hold(n, _r(rng, 260, 340)) + _noise(rng, n, 1.2)
    thr = _hold(n, _r(rng, 75, 90)) + _noise(rng, n, 0.4)
    pitch = _hold(n, _r(rng, 4, 10)) + _noise(rng, n, 0.3)
    roll  = _hold(n, 0.0)
    flaps = _hold(n, 0.0); spbrk=_hold(n,0.0); gear=_hold(n,0.0); wow=_hold(n,0.0); ab=_hold(n,0.0)
    hdg_rel = _hold(n, 0.0)
    return dict(altitude_rel_ft=alt_rel, heading_rel_deg=hdg_rel, ias_kt=ias,
                pitch_deg=pitch, roll_deg=roll, throttle_pct=thr, afterburner_on=ab,
                gear_down=gear, flaps_deg=flaps, speedbrake_pct=spbrk, weight_on_wheels=wow,
                label=np.array(["Climb"]*n, dtype=object))

def seg_level_hold(rng, length_s=40):
    n = _npoints(length_s)
    alt_rel = _hold(n, 0.0) + _noise(rng, n, 2.0)
    ias = _hold(n, _r(rng, 230, 320)) + _noise(rng, n, 1.2)
    thr = _hold(n, _r(rng, 50, 75)) + _noise(rng, n, 0.5)
    pitch = _hold(n, _r(rng, -1.5, 2.0)) + _noise(rng, n, 0.2)
    roll  = _hold(n, _r(rng, -3, 3)) + _noise(rng, n, 0.3)
    flaps=_hold(n,0.0); spbrk=_hold(n,0.0); gear=_hold(n,0.0); wow=_hold(n,0.0); ab=_hold(n,0.0)
    hdg_rel = _noise(rng, n, 0.15)
    return dict(altitude_rel_ft=alt_rel, heading_rel_deg=hdg_rel, ias_kt=ias,
                pitch_deg=pitch, roll_deg=roll, throttle_pct=thr, afterburner_on=ab,
                gear_down=gear, flaps_deg=flaps, speedbrake_pct=spbrk, weight_on_wheels=wow,
                label=np.array(["Level"]*n, dtype=object))

def seg_cruise(rng, length_s=180):
    n = _npoints(length_s)
    alt_rel = _hold(n, 0.0) + _noise(rng, n, 1.0)
    ias = _hold(n, _r(rng, *R.cruise_speed)) + _noise(rng, n, 0.8)
    thr = _hold(n, _r(rng, 60, 80)) + _noise(rng, n, 0.4)
    pitch = _hold(n, _r(rng, -1.0, 1.0)) + _noise(rng, n, 0.15)
    roll  = _hold(n, _r(rng, -2, 2)) + _noise(rng, n, 0.15)
    flaps=_hold(n,0.0); spbrk=_hold(n,0.0); gear=_hold(n,0.0); wow=_hold(n,0.0); ab=_hold(n,0.0)
    hdg_rel = _noise(rng, n, 0.05)
    return dict(altitude_rel_ft=alt_rel, heading_rel_deg=hdg_rel, ias_kt=ias,
                pitch_deg=pitch, roll_deg=roll, throttle_pct=thr, afterburner_on=ab,
                gear_down=gear, flaps_deg=flaps, speedbrake_pct=spbrk, weight_on_wheels=wow,
                label=np.array(["Cruise"]*n, dtype=object))

def seg_turn(rng, length_s=40, direction="L", context="level"):
    n = _npoints(length_s)
    bank = _r(rng, *R.bank_turn_deg) * (-1 if direction=="L" else 1)
    hdg_rate = _r(rng, *R.hdg_rate_turn_dps) * (1 if direction=="R" else -1)
    hdg_rel = np.cumsum(np.full(n, hdg_rate) * DT)
    roll  = _hold(n, bank) + _noise(rng, n, 0.8)
    pitch = _hold(n, _r(rng, -1, 3)) + _noise(rng, n, 0.3)
    thr   = _hold(n, _r(rng, 60, 85)) + _noise(rng, n, 0.4)
    ias   = _hold(n, _r(rng, 260, 340)) + _noise(rng, n, 1.2)
    if context == "climb":
        vs = _hold(n, _r(rng, 3000, 7000)) + _noise(rng, n, 40.0); pitch += 2; thr += 5
    elif context == "descent":
        vs = _hold(n, _r(rng, -4000, -2000)) + _noise(rng, n, 40.0); pitch -= 2; thr -= 5
    else:
        vs = _hold(n, 0.0) + _noise(rng, n, 40.0)
    alt_rel = np.cumsum(vs / (HZ * 60.0))
    flaps=_hold(n,0.0); spbrk=_hold(n,0.0); gear=_hold(n,0.0); wow=_hold(n,0.0); ab=_hold(n,0.0)
    label = "Turn Left" if direction=="L" else "Turn Right"
    return dict(altitude_rel_ft=alt_rel, heading_rel_deg=hdg_rel, ias_kt=ias,
                pitch_deg=pitch, roll_deg=roll, throttle_pct=thr, afterburner_on=ab,
                gear_down=gear, flaps_deg=flaps, speedbrake_pct=spbrk, weight_on_wheels=wow,
                label=np.array([label]*n, dtype=object))

def seg_descent(rng, length_s=120):
    n = _npoints(length_s)
    vs = _hold(n, _r(rng, *R.descent_vs_fpm)) + _noise(rng, n, 40.0)
    alt_rel = np.cumsum(vs / (HZ * 60.0))
    ias = _hold(n, _r(rng, 260, 320)) + _noise(rng, n, 1.0)
    thr = _hold(n, _r(rng, 35, 55)) + _noise(rng, n, 0.4)
    pitch = _hold(n, _r(rng, -3, 1)) + _noise(rng, n, 0.2)
    roll  = _hold(n, _r(rng, -3, 3)) + _noise(rng, n, 0.2)
    flaps=_hold(n,0.0); spbrk=_hold(n,_r(rng,0,40)); gear=_hold(n,0.0); wow=_hold(n,0.0); ab=_hold(n,0.0)
    hdg_rel = _noise(rng, n, 0.1)
    return dict(altitude_rel_ft=alt_rel, heading_rel_deg=hdg_rel, ias_kt=ias,
                pitch_deg=pitch, roll_deg=roll, throttle_pct=thr, afterburner_on=ab,
                gear_down=gear, flaps_deg=flaps, speedbrake_pct=spbrk, weight_on_wheels=wow,
                label=np.array(["Descent"]*n, dtype=object))

def seg_approach(rng, length_s=90):
    n = _npoints(length_s)
    vs = _ramp(n, _r(rng, -1400, -900), _r(rng, -900, -600)) + _noise(rng, n, 30.0)
    alt_rel = np.cumsum(vs / (HZ * 60.0))
    ias = _hold(n, _r(rng, *R.approach_speed)) + _noise(rng, n, 1.0)
    thr = _hold(n, _r(rng, 30, 45)) + _noise(rng, n, 0.3)
    pitch = _hold(n, _r(rng, -1, 4)) + _noise(rng, n, 0.2)
    roll  = _hold(n, _r(rng, -5, 5)) + _noise(rng, n, 0.2)
    flaps=_hold(n,_r(rng,10,20)); spbrk=_hold(n,_r(rng,0,20)); gear=_hold(n,1.0); wow=_hold(n,0.0); ab=_hold(n,0.0)
    hdg_rel = _noise(rng, n, 0.15)
    # Labeled as Landing in the coarse scheme
    return dict(altitude_rel_ft=alt_rel, heading_rel_deg=hdg_rel, ias_kt=ias,
                pitch_deg=pitch, roll_deg=roll, throttle_pct=thr, afterburner_on=ab,
                gear_down=gear, flaps_deg=flaps, speedbrake_pct=spbrk, weight_on_wheels=wow,
                label=np.array(["Landing"]*n, dtype=object))

def seg_landing_roll(rng, length_s=45):
    n = _npoints(length_s)
    ias = _ramp(n, _r(rng, 140, 170), _r(rng, 12, 25)) + _noise(rng, n, 0.8)
    thr = _hold(n, _r(rng, 5, 15)) + _noise(rng, n, 0.3)
    pitch = _hold(n, _r(rng, -2, 5)) + _noise(rng, n, 0.2)
    roll  = _hold(n, _r(rng, -3, 3)) + _noise(rng, n, 0.2)
    flaps = _hold(n, _r(rng, 10, 20))
    spbrk = _hold(n, _r(rng, 30, 80))
    gear  = _hold(n, 1.0); wow   = _hold(n, 1.0)
    alt_rel = _hold(n, 0.0) + _noise(rng, n, 0.1)
    hdg_rel = _noise(rng, n, 0.1)
    return dict(altitude_rel_ft=alt_rel, heading_rel_deg=hdg_rel, ias_kt=ias,
                pitch_deg=pitch, roll_deg=roll, throttle_pct=thr, afterburner_on=_hold(n,0.0),
                gear_down=gear, flaps_deg=flaps, speedbrake_pct=spbrk, weight_on_wheels=wow,
                label=np.array(["Landing"]*n, dtype=object))

# --------------------- Sortie composers (varied starts) ---------------------
def compose_takeoff_then_land(rng):
    segs = []
    segs += [("Taxi", seg_taxi(rng, _r(rng, 40, 90))),
             ("TO",   seg_takeoff_roll(rng, _r(rng, 18, 28))),
             ("ROT",  seg_rotation(rng, _r(rng, 2, 4))),
             ("IC",   seg_initial_climb(rng, _r(rng, 18, 35))),
             ("CLB",  seg_climb(rng, _r(rng, 60, 120))),
             ("CRZ",  seg_cruise(rng, _r(rng, 60, 160)))]
    if rng.random() < 0.7: segs.append(("TNL", seg_turn(rng, _r(rng, 25, 60), direction="L", context="level")))
    if rng.random() < 0.7: segs.append(("TNR", seg_turn(rng, _r(rng, 25, 60), direction="R", context="level")))
    segs += [("DSC",  seg_descent(rng, _r(rng, 60, 120))),
             ("LVL",  seg_level_hold(rng, _r(rng, 20, 40))),
             ("APR",  seg_approach(rng, _r(rng, 60, 100))),
             ("LR",   seg_landing_roll(rng, _r(rng, 25, 45)))]
    return segs

def compose_landing_then_takeoff(rng):
    segs = []
    segs += [("APR", seg_approach(rng, _r(rng, 50, 90))),
             ("LR",  seg_landing_roll(rng, _r(rng, 20, 35))),
             ("TAX", seg_taxi(rng, _r(rng, 30, 60))),
             ("TO",  seg_takeoff_roll(rng, _r(rng, 15, 25))),
             ("ROT", seg_rotation(rng, _r(rng, 2, 4))),
             ("IC",  seg_initial_climb(rng, _r(rng, 15, 30)))]
    if rng.random() < 0.6: segs.append(("CLB", seg_climb(rng, _r(rng, 40, 90))))
    if rng.random() < 0.5: segs.append(("TNL", seg_turn(rng, _r(rng, 20, 50), direction="L", context="level")))
    if rng.random() < 0.5: segs.append(("TNR", seg_turn(rng, _r(rng, 20, 50), direction="R", context="level")))
    segs += [("DSC", seg_descent(rng, _r(rng, 40, 90))),
             ("APR", seg_approach(rng, _r(rng, 50, 90))),
             ("LR",  seg_landing_roll(rng, _r(rng, 25, 45)))]
    return segs

def compose_pattern(rng):
    segs = []
    segs += [("TAX", seg_taxi(rng, _r(rng, 30, 60))),
             ("TO",  seg_takeoff_roll(rng, _r(rng, 15, 25))),
             ("ROT", seg_rotation(rng, _r(rng, 2, 3))),
             ("IC",  seg_initial_climb(rng, _r(rng, 12, 20)))]
    for _ in range(np.random.randint(1, 3)):
        segs += [("LVL", seg_level_hold(rng, _r(rng, 20, 40))),
                 ("TNL", seg_turn(rng, _r(rng, 25, 45), direction="L", context="level")),
                 ("TNR", seg_turn(rng, _r(rng, 25, 45), direction="R", context="level"))]
    segs += [("DSC", seg_descent(rng, _r(rng, 40, 80))),
             ("APR", seg_approach(rng, _r(rng, 50, 80))),
             ("LR",  seg_landing_roll(rng, _r(rng, 25, 45)))]
    return segs

def compose_goaround(rng):
    segs = compose_takeoff_then_land(rng)[:-1]  # up to approach
    segs += [("CLB", seg_climb(rng, _r(rng, 20, 60))),
             ("TNR", seg_turn(rng, _r(rng, 20, 40), direction="R", context="climb")),
             ("CRZ", seg_cruise(rng, _r(rng, 20, 60))),
             ("APR", seg_approach(rng, _r(rng, 50, 90))),
             ("LR",  seg_landing_roll(rng, _r(rng, 25, 45)))]
    return segs

def compose_rto(rng):
    return [("TAX", seg_taxi(rng, _r(rng, 40, 70))),
            ("TO",  seg_takeoff_roll(rng, _r(rng, 15, 25))),
            ("LR",  seg_landing_roll(rng, _r(rng, 25, 45)))]

def compose_touchngo(rng):
    return [("APR", seg_approach(rng, _r(rng, 50, 90))),
            ("LR",  seg_landing_roll(rng, _r(rng, 15, 25))),
            ("TO",  seg_takeoff_roll(rng, _r(rng, 12, 20))),
            ("ROT", seg_rotation(rng, _r(rng, 2, 3))),
            ("IC",  seg_initial_climb(rng, _r(rng, 15, 25))),
            ("CRZ", seg_cruise(rng, _r(rng, 20, 40))),
            ("DSC", seg_descent(rng, _r(rng, 40, 80))),
            ("LVL", seg_level_hold(rng, _r(rng, 15, 30))),
            ("APR", seg_approach(rng, _r(rng, 50, 90))),
            ("LR",  seg_landing_roll(rng, _r(rng, 25, 45)))]

# --------------------- Core generators ---------------------
def generate_sortie(rng=None, template: str | None = None) -> pd.DataFrame:
    rng = rng or np.random.default_rng()
    if template is None:
        templates = [
            ("takeoff_then_land", 0.35),
            ("landing_then_takeoff", 0.20),
            ("pattern", 0.15),
            ("goaround", 0.15),
            ("rto", 0.07),
            ("touchngo", 0.08),
        ]
        probs = np.array([w for _, w in templates]); probs /= probs.sum()
        template = rng.choice([k for k, _ in templates], p=probs)

    if template == "takeoff_then_land": segs = compose_takeoff_then_land(rng)
    elif template == "landing_then_takeoff": segs = compose_landing_then_takeoff(rng)
    elif template == "pattern": segs = compose_pattern(rng)
    elif template == "goaround": segs = compose_goaround(rng)
    elif template == "rto": segs = compose_rto(rng)
    elif template == "touchngo": segs = compose_touchngo(rng)
    else: segs = compose_takeoff_then_land(rng)

    buf = _new_buf()
    t0 = 0.0; alt0 = 0.0; hdg0 = np.random.uniform(0, 360.0)
    for _, s in segs:
        t0, alt0, hdg0 = _append(buf, s, t0, alt0, hdg0)

    df = pd.DataFrame(buf)
    df["vertical_speed_fpm"] = pd.Series(df["vertical_speed_fpm"]).rolling(3, center=True, min_periods=1).mean()
    df["label"] = df["label"].astype(object)
    return df

# --------------------- Split writers ---------------------
def write_split(out_dir: Path, n_sorties: int, seed: int, hz: float):
    set_rate(hz)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(1, n_sorties + 1):
        df = generate_sortie(rng)
        df.to_csv(out_dir / f"sortie_{i:03d}.csv", index=False)

    (out_dir / "README.txt").write_text(
        "Synthetic fighter regime dataset (time-series, no geometry)\n"
        f"Sampling rate: {HZ} Hz (dt={DT:.3f} s)\n"
        "Columns: time_s, altitude_ft, vertical_speed_fpm, ias_kt, pitch_deg, roll_deg, heading_deg, "
        "throttle_pct, afterburner_on, gear_down, flaps_deg, speedbrake_pct, weight_on_wheels, label\n"
        "Regimes: " + ", ".join(LABELS) + "\n"
    )
    meta = {"hz": HZ, "labels": LABELS, "n_sorties": n_sorties, "seed": seed}
    pd.Series(meta).to_json(out_dir / "meta.json", indent=2)

# --------------------- CLI ---------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate synthetic fighter sorties (coarse regime labels).")
    ap.add_argument("--out_root", type=str, default="dataset/fighter_regimes_synth", help="Root output folder")
    ap.add_argument("--train", type=int, default=20, help="Number of TRAIN sorties")
    ap.add_argument("--test", type=int, default=10, help="Number of TEST sorties")
    ap.add_argument("--hz", type=float, default=10.0, help="Sampling rate (Hz). 10 Hz recommended.")
    ap.add_argument("--seed-train", type=int, default=42, dest="seed_train", help="Random seed for train split")
    ap.add_argument("--seed-test", type=int, default=777, dest="seed_test", help="Random seed for test split")
    args = ap.parse_args()

    root = Path(args.out_root)
    train_dir = root / "train"
    test_dir  = root / "test"

    write_split(train_dir, n_sorties=args.train, seed=args.seed_train, hz=args.hz)
    write_split(test_dir,  n_sorties=args.test,  seed=args.seed_test,  hz=args.hz)

    print(f"Done.\n Train: {train_dir.resolve()}\n Test : {test_dir.resolve()}")
