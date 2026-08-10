#!/usr/bin/env python3
"""
Extract the unique SMD-TR stations inside the Figure 2 map window and write the
lon,lat,vs30 CSV that make_fig02.py expects.

Input : Metadata.csv from NHERI DesignSafe PRJ-3950 v3 (DOI 10.17603/ds2-f21x-s189)
Output: AFAD_vs30.csv  ->  code,lon,lat,vs30,vs30_source

The flatfile is record-wise (one row per waveform), so the same station appears
many times; this collapses it to one row per station. Column names differ a
little between SMD-TR versions, so they are matched case-insensitively against a
list of candidates and the actual names found are printed for checking.
"""
import csv, sys, collections
from pathlib import Path

# This file lives in <repo>/scripts/; paths resolve from __file__ so the script
# runs from any working directory. Metadata.csv is third-party and is NOT
# redistributed here: download it from DOI 10.17603/ds2-f21x-s189 (NHERI
# DesignSafe PRJ-3950, v3) into <repo>/data/external/.
ROOT = Path(__file__).resolve().parent.parent
SRC = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "external" / "Metadata.csv")
OUT = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "data" / "AFAD_vs30.csv")
if not Path(SRC).exists():
    raise SystemExit(
        f"SMD-TR station table not found: {SRC}\n"
        "Download Metadata.csv from https://doi.org/10.17603/ds2-f21x-s189 "
        "(NHERI DesignSafe PRJ-3950, v3) into data/external/.")
W, E, S, N = 28.0, 30.0, 40.5, 41.5          # Figure 2 window

CAND = {
    "code": ["station_code", "stationcode", "sta_code", "station_id", "stationid", "station"],
    "lat":  ["stat_latitude", "station_latitude", "sta_lat", "station_lat",
             "st_latitude", "latitude_station"],
    "lon":  ["stat_longitude", "station_longitude", "sta_lon", "station_lon",
             "st_longitude", "longitude_station"],
    "vs30": ["vs30_(m/s)", "vs30", "vs30_m_s", "vs30(m/s)", "vs_30", "vs30_mps"],
    "flag": ["vs30_flag", "vs30_type", "site_class_flag", "vs30_measured", "vs30_source"],
}

# Fuzzy fallback, applied only when no exact name matches. The flatfile carries
# BOTH source and site coordinates (EQ_Lat/EQ_Lon alongside Stat_Latitude/
# Stat_Longitude), so a loose substring match on "lat" would silently plot
# hypocentres instead of recording sites. A candidate must therefore contain a
# station marker and must not contain an event/epicentre/hypocentre marker.
FUZZY = {
    "lat":  (("lat",),  ("stat", "sta", "st")),
    "lon":  (("lon",),  ("stat", "sta", "st")),
    "vs30": (("vs30",), ()),
}
REJECT = ("ev", "eq", "epi", "hyp")


def _norm(h):
    return h.strip().lower().replace(" ", "_")


def pick(header, names, key=None):
    low = {_norm(h): h for h in header}
    for n in names:
        if n in low:
            return low[n]
    if key in FUZZY:
        want, station = FUZZY[key]
        for n, orig in low.items():
            if not all(w in n for w in want):
                continue
            if any(r in n for r in REJECT):
                continue                      # EQ_Lat, Epi_*, Hyp_* -> not a site
            if station and not any(sm in n for sm in station):
                continue
            return orig
    return None


with open(SRC, newline="", encoding="utf-8-sig", errors="replace") as f:
    rdr = csv.DictReader(f)
    hdr = rdr.fieldnames
    cols = {k: pick(hdr, v, k) for k, v in CAND.items()}
    print("matched columns:", cols)
    missing = [k for k in ("lat", "lon", "vs30") if cols[k] is None]
    if missing:
        sys.exit(f"could not find {missing}. Header is:\n  " + "\n  ".join(hdr))

    seen = {}
    for row in rdr:
        try:
            lo, la, vs = (float(row[cols["lon"]]), float(row[cols["lat"]]),
                          float(row[cols["vs30"]]))
        except (TypeError, ValueError):
            continue
        if not (W <= lo <= E and S <= la <= N) or vs <= 0:
            continue
        key = row[cols["code"]] if cols["code"] else (round(lo, 5), round(la, 5))
        seen.setdefault(key, (lo, la, vs,
                              row.get(cols["flag"], "") if cols["flag"] else ""))

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["code", "lon", "lat", "vs30", "vs30_source"])
    for k, (lo, la, vs, fl) in sorted(seen.items(), key=lambda kv: str(kv[0])):
        w.writerow([k, f"{lo:.5f}", f"{la:.5f}", f"{vs:.1f}", fl])

print(f"{len(seen)} unique stations inside {W}-{E}E / {S}-{N}N -> {OUT}")
if seen:
    v = sorted(x[2] for x in seen.values())
    print(f"Vs30 range {v[0]:.0f}-{v[-1]:.0f} m/s, median {v[len(v) // 2]:.0f}")
    print("flag values:", dict(collections.Counter(x[3] for x in seen.values())))
