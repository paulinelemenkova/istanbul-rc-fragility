#!/usr/bin/env python3
"""
esm_io.py - reader for ESM (Engineering Strong-Motion database) ASCII records.

ESM distributes processed acceleration as ASCII files with a header of ~55
"KEY: value" lines followed by one acceleration sample per line. The header
carries everything needed, so no manual pre-processing is required:

    EVENT_NAME, EVENT_DATE_YYYYMMDD, EVENT_ID, MAGNITUDE_W,
    STATION_CODE, STATION_LATITUDE_DEGREE, STATION_LONGITUDE_DEGREE,
    VS30_M/S, EPICENTRAL_DISTANCE_KM, JB_DISTANCE_KM,
    STREAM (e.g. HGE / HGN), UNITS (cm/s^2), SAMPLING_INTERVAL_S, NDATA,
    LOW_CUT_FREQUENCY_HZ, HIGH_CUT_FREQUENCY_HZ, PGA_CM/S^2

Usage
-----
    recs = load_folder("records/")          # list of Record
    recs = select_horizontal_pairs(recs)    # keep the stronger horizontal
"""
import glob
import os
import re

import numpy as np

G_CM = 980.665            # cm/s^2 per g


class Record:
    __slots__ = ("acc", "dt", "meta", "path")

    def __init__(self, acc, dt, meta, path):
        self.acc = acc            # acceleration in g
        self.dt = dt              # s
        self.meta = meta
        self.path = path

    # -- convenience accessors ------------------------------------------
    @property
    def rid(self):
        return self.meta.get("id", os.path.basename(self.path))

    @property
    def station(self):
        return self.meta.get("station_code", "")

    @property
    def event(self):
        return self.meta.get("event_name", self.meta.get("event_id", ""))

    @property
    def mw(self):
        return _f(self.meta.get("magnitude_w"))

    @property
    def rjb(self):
        return _f(self.meta.get("jb_distance_km"), _f(self.meta.get("epicentral_distance_km")))

    @property
    def vs30(self):
        return _f(self.meta.get("vs30_m/s"))

    @property
    def pga(self):
        return float(np.max(np.abs(self.acc)))

    @property
    def channel(self):
        return self.meta.get("stream", "")

    def __repr__(self):
        return (f"Record({self.rid}, {self.event}, Mw {self.mw}, "
                f"Rjb {self.rjb} km, {len(self.acc)} pts @ {self.dt} s, "
                f"PGA {self.pga:.3f} g)")


def _f(v, default=np.nan):
    try:
        return float(str(v).split()[0])
    except (TypeError, ValueError, IndexError):
        return default


def read_esm_ascii(path):
    """Parse one ESM ASCII file into a Record (acceleration in g)."""
    meta, data = {}, []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            m = re.match(r"^([A-Z0-9_\./\^\- ]+):\s*(.*)$", s)
            if m and not _is_number(s.split(":")[0]):
                meta[m.group(1).strip().lower()] = m.group(2).strip()
            else:
                try:
                    data.append(float(s.split()[0]))
                except (ValueError, IndexError):
                    pass
    if not data:
        raise ValueError(f"no samples parsed from {path}")
    dt = _f(meta.get("sampling_interval_s"))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"missing SAMPLING_INTERVAL_S in {path}")
    acc = np.asarray(data, float)
    n = meta.get("ndata")
    if n and str(n).isdigit() and int(n) <= acc.size:
        acc = acc[:int(n)]
    units = meta.get("units", "cm/s^2").lower()
    if "cm" in units:
        acc = acc / G_CM
    elif "m/s" in units and "cm" not in units:
        acc = acc / 9.80665
    acc = acc - acc.mean()                    # remove any residual offset
    meta["id"] = f"{meta.get('event_id','EV')}_{meta.get('station_code','ST')}_{meta.get('stream','')}"
    return Record(acc, dt, meta, path)


def _is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def load_folder(folder, pattern="*"):
    """Read every ESM ASCII file in a folder (recursively)."""
    out = []
    for p in sorted(glob.glob(os.path.join(folder, "**", pattern), recursive=True)):
        if os.path.isdir(p) or p.lower().endswith((".zip", ".pdf", ".png", ".csv")):
            continue
        try:
            out.append(read_esm_ascii(p))
        except Exception as e:                                    # noqa: BLE001
            print(f"  skipped {os.path.basename(p)}: {e}")
    return out


def select_horizontal_pairs(recs):
    """Keep, per event+station, the stronger of the two horizontal components.

    Vertical channels (stream ending in Z, or containing 'HGZ'/'HNZ') are
    dropped: the frame model is two-dimensional.
    """
    hor = [r for r in recs if not r.channel.upper().endswith("Z")]
    best = {}
    for r in hor:
        key = (r.meta.get("event_id", ""), r.station)
        if key not in best or r.pga > best[key].pga:
            best[key] = r
    return sorted(best.values(), key=lambda r: -r.mw if np.isfinite(r.mw) else 0)


def trim_significant_duration(rec, lo=0.05, hi=0.95, pad_s=5.0):
    """Trim a record to its Arias-intensity significant duration, with a fixed
    pre/post pad. This removes the long low-amplitude coda some ESM records
    carry (occasionally 300+ s) without discarding any of the shaking that
    matters for NLTHA; it does not change amplitude, only length.

    lo/hi default to the standard 5-95% D_{5-95} definition (Trifunac & Brady,
    1975); pad_s is extra time kept before/after that window so the ramp-up
    and decay are not clipped.
    """
    a = rec.acc
    ia = np.cumsum(a ** 2)
    ia = ia / ia[-1]
    i0 = int(np.searchsorted(ia, lo))
    i1 = int(np.searchsorted(ia, hi))
    pad = int(round(pad_s / rec.dt))
    i0 = max(0, i0 - pad)
    i1 = min(len(a), i1 + pad)
    trimmed = a[i0:i1] - a[i0:i1].mean()
    return Record(trimmed, rec.dt, rec.meta, rec.path)


def summary(recs):
    print(f"{len(recs)} records")
    for r in recs:
        print(f"  {r.rid:38s} Mw {r.mw:4.1f}  Rjb {r.rjb:5.1f} km  "
              f"Vs30 {r.vs30:6.0f}  PGA {r.pga:5.3f} g  "
              f"{len(r.acc):6d} pts @ {r.dt:.4f} s")
