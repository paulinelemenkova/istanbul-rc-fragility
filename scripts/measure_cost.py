#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
measure_cost.py -- fills the surrogate rows of Table "Computational cost".

Run this on the SAME machine that produced simulation_database.csv, so that
the NLTHA timings (read from the runtime_s column) and the surrogate timings
below refer to identical hardware, as the table caption requires:

    python measure_cost.py

Prints every quantity the table asks for: NLTHA per analysis and in total,
surrogate training time, single-prediction latency, batch-evaluation time,
the per-evaluation speed-up and the break-even number of predictions.
"""
import time

import numpy as np
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier

from oof_one import load, FEATURES, SEED

REPS = 200          # repetitions for the single-prediction latency

d, X = load()
y_reg = np.log10(d["midr"].to_numpy(float))
y_cls = d["ds_index"].to_numpy(int)
cnt = np.bincount(y_cls, minlength=5).astype(float)
w = (len(y_cls) / (5 * cnt))[y_cls]

# ---- simulator cost, read from the database itself -------------------------
rt = d["runtime_s"]
print(f"NLTHA, single analysis      : {rt.mean():.2f} s (mean), "
      f"{rt.median():.2f} s (median), {rt.max():.1f} s (max)")
print(f"NLTHA, full database        : {rt.sum()/3600:.2f} core-hours "
      f"({len(d)} analyses)")

# ---- surrogate training ----------------------------------------------------
kw = dict(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8,
          colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, n_jobs=-1)
t0 = time.perf_counter()
reg = XGBRegressor(objective="reg:squarederror", **kw).fit(X, y_reg)
clf = XGBClassifier(objective="multi:softprob", num_class=5, **kw).fit(
    X, y_cls, sample_weight=w)
t_train = time.perf_counter() - t0
print(f"Surrogate training          : {t_train:.1f} s "
      f"(regressor + classifier, final configuration, no tuning)")

# ---- batch evaluation ------------------------------------------------------
t0 = time.perf_counter()
reg.predict(X); clf.predict(X)
t_batch = time.perf_counter() - t0
print(f"Surrogate, full database    : {t_batch:.2f} s for {len(d)} cases")

# ---- single-prediction latency --------------------------------------------
x1 = X[:1]
reg.predict(x1)                                    # warm-up
t0 = time.perf_counter()
for _ in range(REPS):
    reg.predict(x1); clf.predict(x1)
t_one = (time.perf_counter() - t0) / REPS
print(f"Surrogate, single prediction: {t_one*1e3:.2f} ms")

# ---- derived quantities ----------------------------------------------------
speedup = rt.mean() / t_one
breakeven = (t_train + rt.sum()) / max(rt.mean() - t_one, 1e-12)
print(f"Speed-up per evaluation     : ~{speedup:.3g}x "
      f"(10^{np.log10(speedup):.1f})")
print(f"Break-even                  : {breakeven:,.0f} predictions "
      f"(recovers database generation + training)")
print(f"Break-even, training only   : "
      f"{t_train/max(rt.mean()-t_one,1e-12):,.0f} predictions")
