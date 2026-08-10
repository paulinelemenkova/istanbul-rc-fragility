#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Out-of-fold predictions for ONE candidate model under ONE grouping.

    python oof_one.py <model> <partition>
        model     = linear | rf | xgb | mlp
        partition = record | frame

Writes oof_<model>_<partition>.npz with the pooled held-out regression and
classification predictions (GroupKFold, 5 folds).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from xgboost import XGBRegressor, XGBClassifier

# ---------------------------------------------------------------- paths ----
# This file lives in <repo>/scripts/. Every path below is resolved from
# __file__, so the script runs from any working directory and on any machine.
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
EXTERNAL = DATA / "external"
RESULTS = ROOT / "results"
DERIVED = RESULTS / "derived"
DERIVED.mkdir(parents=True, exist_ok=True)

CSV = str(RESULTS / "simulation_database.csv")
SEED, N_FOLDS = 7, 5

FEATURES = ["sa_t1_g", "pga_scaled_g", "sa_ratio", "mw", "rjb_km", "vs30",
            "n_storey", "T1_s", "fc_MPa", "rho_long", "r_k",
            "soft_i", "deficient_i"]


def load():
    d = pd.read_csv(CSV, keep_default_na=False, na_values=[""])
    d["sa_t1_g"] = d["sa_target_g"]                 # scaled Sa(T1) applied
    d["pga_scaled_g"] = d["pga_g"] * d["scale"]     # scaled PGA
    d["sa_ratio"] = d["sa_record_g"] / d["pga_g"]   # spectral shape Sa(T1)/PGA
    d["soft_i"] = d["soft"].astype(int)
    d["deficient_i"] = d["deficient"].astype(int)
    X = d[FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    nan = np.isnan(X)                               # one record lacks Vs30
    if nan.any():
        X[nan] = np.take(np.nanmedian(X, axis=0), np.where(nan)[1])
    return d, X


def make(kind):
    if kind == "linear":
        return (make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))),
                make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)))
    if kind == "rf":
        return (RandomForestRegressor(n_estimators=300, min_samples_leaf=4,
                                      random_state=SEED),
                RandomForestClassifier(n_estimators=300, min_samples_leaf=4,
                                       random_state=SEED))
    if kind == "xgb":
        kw = dict(n_estimators=500, max_depth=5, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                  random_state=SEED, n_jobs=1)
        return (XGBRegressor(objective="reg:squarederror", **kw),
                XGBClassifier(objective="multi:softprob", num_class=5, **kw))
    if kind == "mlp":
        return (make_pipeline(StandardScaler(),
                              MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=400,
                                           early_stopping=True, random_state=SEED)),
                make_pipeline(StandardScaler(),
                              MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400,
                                            early_stopping=True, random_state=SEED)))
    raise SystemExit(f"unknown model {kind}")


def main(kind, part):
    d, X = load()
    y_reg = np.log10(d["midr"].to_numpy(float))
    y_cls = d["ds_index"].to_numpy(int)
    groups = d["rid"].to_numpy() if part == "record" else d["fid"].to_numpy()
    cnt = np.bincount(y_cls, minlength=5).astype(float)
    w = (len(y_cls) / (5 * cnt))[y_cls]             # inverse-frequency weights

    pr = np.empty(len(d)); pc = np.empty(len(d), int); pp = np.empty((len(d), 5))
    for f, (tr, te) in enumerate(GroupKFold(n_splits=N_FOLDS).split(X, y_reg, groups), 1):
        reg, clf = make(kind)
        reg.fit(X[tr], y_reg[tr])
        pr[te] = reg.predict(X[te])
        try:
            clf.fit(X[tr], y_cls[tr], sample_weight=w[tr])
        except (TypeError, ValueError):
            step = clf.steps[-1][0]
            if step.startswith("logistic"):
                clf.fit(X[tr], y_cls[tr], **{f"{step}__sample_weight": w[tr]})
            else:
                clf.fit(X[tr], y_cls[tr])           # MLP has no sample_weight
        pc[te] = clf.predict(X[te])
        pp[te] = clf.predict_proba(X[te])
        print(f"  fold {f}/{N_FOLDS} done", flush=True)

    out = DERIVED / f"oof_{kind}_{part}.npz"
    np.savez(out, pr=pr, pc=pc, pp=pp)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
