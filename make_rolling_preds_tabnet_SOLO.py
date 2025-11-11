#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_rolling_preds_tabnet_SOLO.py (FINAL)
----------------------------------------
- Pipeline original reducido a **solo TabNet**, alineado a los otros 3.
- Dos horizontes fijos: corto y mediano (se corren ambos en orden).
- Sin fugas: sin bfill, rollings con shift(1), recorte por lags.
- Escalado por ventana en X e Y (inversión al predecir y en holdout).
- Logs claros: tqdm por ventana, R2 y tiempo por ventana; prints de secciones.
- Reanudación por progress_TabNet.txt; guardados por horizonte/estación.

Ejemplo:
  python make_rolling_preds_tabnet_SOLO.py \
    --data_dir data/con_lags \
    --stations "S-E EL CALVARIO_con_lags" \
    --min_train 1500 --max_windows 200 \
    --tabnet_epochs 120 --tabnet_patience 15 --tabnet_batch 512
"""

import time
import argparse
import traceback
from pathlib import Path
import sys
import math
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, r2_score

try:
    import torch
    from pytorch_tabnet.tab_model import TabNetRegressor
    TABNET_AVAILABLE = True
except Exception as e:
    TABNET_AVAILABLE = False
    _TABNET_ERR = e

# ------------------------------
# Config
# ------------------------------
RESULTS_DIR = Path("resultados_tabnet_SOLO")
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
SEED = 42
np.random.seed(SEED)

DEFAULTS = {
    "MIN_TRAIN": 1500,
    "TABNET_MAX_EPOCHS": 120,
    "TABNET_PATIENCE": 15,
    "TABNET_BATCH": 512,
}

HORIZONS = {
    "corto":   {"WINDOW": 168, "STEP": 24, "PURGE": 24},
    "mediano": {"WINDOW": 720, "STEP": 72, "PURGE": 72},
}

# ------------------------------
# Utils
# ------------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def log_write(handle, msg, also_console=False):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    line = f"[{ts}] {msg}\n"
    try:
        handle.write(line); handle.flush()
    except Exception:
        pass
    if also_console:
        print(line, end="")

def metrics_dict(y_true, y_pred, tol=5.0):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(np.mean((y_true - y_pred) ** 2))
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")
    within_tol = float(np.mean(np.abs(y_true - y_pred) <= tol) * 100)
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "%Tol": within_tol}

def append_metrics_csv(path: Path, row: dict):
    df = pd.DataFrame([row])
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)

# ------------------------------
# Feature builder (sin fugas)
# ------------------------------
def safe_read_and_build(path_csv: Path, log_handle=None):
    try:
        df = pd.read_csv(path_csv, parse_dates=["datetime"])  # requiere 'datetime'
    except Exception as e:
        if log_handle:
            log_write(log_handle, f"ERROR reading {path_csv}: {e}")
        raise

    if "potencia" not in df.columns:
        if log_handle:
            log_write(log_handle, f"ERROR: columna 'potencia' no encontrada en {path_csv}. Columnas: {list(df.columns)}")
        return None

    df = df.sort_values("datetime").reset_index(drop=True)
    # Limpieza sin bfill
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill()

    # calendario
    df["hour"] = df["datetime"].dt.hour
    df["dow"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month

    # lags 4..24
    for l in range(4, 25):
        df[f"potencia_lag_{l}h"] = df["potencia"].shift(l)

    # rollings solo pasado
    df["pot_roll3"]  = df["potencia"].shift(1).rolling(3,  min_periods=1).mean()
    df["pot_roll6"]  = df["potencia"].shift(1).rolling(6,  min_periods=1).mean()
    df["pot_roll24"] = df["potencia"].shift(1).rolling(24, min_periods=1).mean()

    # Recorte por lags
    df = df.iloc[24:].reset_index(drop=True)
    df = df.ffill()
    return df

# ------------------------------
# TabNet
# ------------------------------
def fit_tabnet(X_train, y_train, max_epochs, patience, batch_size, use_gpu=True):
    if not TABNET_AVAILABLE:
        raise RuntimeError(f"pytorch_tabnet no disponible: {_TABNET_ERR}")

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    inner_split = int(0.85 * len(X_train))
    inner_split = min(max(inner_split, 1), len(X_train) - 1)
    X_tr, X_val = X_train[:inner_split], X_train[inner_split:]
    y_tr, y_val = y_train[:inner_split], y_train[inner_split:]

    device_name = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"

    model = TabNetRegressor(
        seed=SEED,
        device_name=device_name,
        n_d=32, n_a=32, n_steps=4,
        gamma=1.3, lambda_sparse=1e-6,
        optimizer_fn=torch.optim.AdamW,
        optimizer_params=dict(lr=2e-3, weight_decay=1e-5),
        mask_type="entmax",
        verbose=1,
    )

    t0 = time.time()
    model.fit(
        X_tr, y_tr.reshape(-1, 1),
        eval_set=[(X_val, y_val.reshape(-1, 1))],
        eval_metric=["rmse"],
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        virtual_batch_size=max(64, batch_size // 4),
        num_workers=0,
        drop_last=False,
    )
    t1 = time.time()
    print(f"[TabNet] fit en {t1 - t0:.1f}s | device={device_name} | batch={batch_size}")
    return model

# ------------------------------
# Rolling SOLO TabNet
# ------------------------------
def rolling_backtest_tabnet(X, y, station, horizon_label,
                            min_train, window, purge, step,
                            max_windows, scaler_ctor, log_handle, fit_kwargs):
    n = len(X)
    preds = np.full(n, np.nan)
    starts = list(range(min_train, n - window + 1, step))
    if max_windows is not None:
        starts = starts[:max_windows]
    total_windows = len(starts)

    if log_handle:
        log_write(log_handle, f"Rolling {horizon_label}: {total_windows} ventanas (train={min_train}, W={window}, STEP={step}, PURGE={purge})")

    station_dir = RESULTS_DIR / horizon_label / station
    ensure_dir(station_dir)

    progress_path = station_dir / "progress_TabNet.txt"
    resume_from = 0
    if progress_path.exists():
        try:
            resume_from = int(progress_path.read_text().strip())
            print(f"[Reanudar] saltando hasta ventana {resume_from}")
        except Exception:
            resume_from = 0

    pbar = tqdm(enumerate(starts, 1), total=total_windows, desc=f"{station}-TabNet-{horizon_label}", unit="win", file=sys.stdout)
    for w_idx, start in pbar:
        if w_idx <= resume_from:
            continue
        t0w = time.time()
        try:
            train_idx = np.arange(0, start - purge, dtype=int)
            test_idx = np.arange(start, start + window, dtype=int)
            if len(train_idx) < min_train:
                if log_handle: log_write(log_handle, f"Skip win {w_idx}: train {len(train_idx)} < {min_train}")
                continue

            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te = X[test_idx]

            scaler_x = scaler_ctor().fit(X_tr)
            X_tr_s = scaler_x.transform(X_tr).astype(np.float32)
            X_te_s = scaler_x.transform(X_te).astype(np.float32)

            from sklearn.preprocessing import RobustScaler as _RS
            scaler_y = _RS().fit(y_tr.reshape(-1, 1))
            y_tr_s = scaler_y.transform(y_tr.reshape(-1, 1)).astype(np.float32).ravel()

            m = fit_tabnet(X_tr_s, y_tr_s, **fit_kwargs)

            preds_s = m.predict(np.nan_to_num(X_te_s, 0.0, 0.0, 0.0)).reshape(-1, 1)
            preds_vals = scaler_y.inverse_transform(preds_s).ravel()
            preds[test_idx] = preds_vals

            with open(progress_path, "w", encoding="utf-8") as f:
                f.write(str(w_idx))

            t1w = time.time()
            y_true_win = y[test_idx]
            mask = np.isfinite(preds_vals) & np.isfinite(y_true_win)
            if mask.any():
                md = metrics_dict(y_true_win[mask], preds_vals[mask])
                pbar.set_postfix({"R2": f"{md['R2']:.3f}", "fit_s": f"{(t1w - t0w):.1f}"})
                if log_handle:
                    log_write(log_handle, f"win {w_idx} OK | fit+pred {(t1w - t0w):.1f}s | R2={md['R2']:.4f} | {100*w_idx/total_windows:.1f}%")
        except Exception as e:
            if log_handle:
                log_write(log_handle, f"ERROR win {w_idx}: {e}")
                log_write(log_handle, traceback.format_exc())
            continue

    pred_dir = station_dir / "predictions"; ensure_dir(pred_dir)
    out_path = pred_dir / "preds_TabNet.csv"
    pd.DataFrame({"y_true": y, "y_pred": preds}).to_csv(out_path, index=False)
    if log_handle: log_write(log_handle, f"Saved predictions -> {out_path}")

    mask_all = np.isfinite(preds)
    metrics = {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan"), "%Tol": float("nan")}
    if mask_all.sum() > 0:
        metrics = metrics_dict(y[mask_all], preds[mask_all])
    return metrics

# ------------------------------
# Main
# ------------------------------
def main(args):
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir {data_dir} no existe.")

    files = sorted([f.stem for f in data_dir.glob("*.csv")])
    if args.stations:
        requested = [s.strip() for s in args.stations.split(",") if s.strip()]
        stations = [s for s in requested if s in files]
        missing = [s for s in requested if s not in files]
        if missing:
            print(f"Advertencia: estaciones no encontradas: {missing}")
    else:
        stations = files

    for h in HORIZONS.keys():
        ensure_dir(RESULTS_DIR / h)
    rolling_csv_template = RESULTS_DIR / "{h}" / "metrics_rolling.csv"
    holdout_csv_template = RESULTS_DIR / "{h}" / "metrics_holdout.csv"

    gpu_avail = torch.cuda.is_available() if TABNET_AVAILABLE else False
    print(f"GPU disponible para TabNet: {gpu_avail}")
    if not TABNET_AVAILABLE:
        print(f"ERROR: TabNet no disponible: {_TABNET_ERR}")
        return

    total = len(stations)
    overall_t0 = time.time()

    for idx, station in enumerate(stations, 1):
        start_sta = time.time()
        print(f"\n>>> [{idx}/{total}] Estación: {station}")
        log_dir = RESULTS_DIR / "logs"; ensure_dir(log_dir)
        log_file = log_dir / f"log_{station}.txt"
        lh = open(log_file, "a", buffering=1, encoding="utf-8")
        log_write(lh, f"=== START {time.asctime()} ===")

        csv_path = data_dir / f"{station}.csv"
        try:
            df = safe_read_and_build(csv_path, log_handle=lh)
        except Exception as e:
            log_write(lh, f"Fallo reading/building features para {station}: {e}")
            lh.close(); continue
        if df is None:
            log_write(lh, "Skipping station: falta 'potencia'"); lh.close(); continue

        feature_cols = [c for c in df.columns if c.startswith("potencia_lag_")] + [
            "pot_roll3", "pot_roll6", "pot_roll24", "hour", "dow", "month"
        ]

        split_idx = int(0.9 * len(df))
        df_train = df.iloc[:split_idx].reset_index(drop=True)
        df_hold = df.iloc[split_idx:].reset_index(drop=True)
        X_train_all, y_train_all = df_train[feature_cols].values, df_train["potencia"].values
        X_hold, y_hold = df_hold[feature_cols].values, df_hold["potencia"].values

        for horizon_label, cfg in HORIZONS.items():
            print(f"\n>>>   --- Horizonte: {horizon_label} (W={cfg['WINDOW']}, STEP={cfg['STEP']}, PURGE={cfg['PURGE']})")
            ensure_dir(RESULTS_DIR / horizon_label / station)
            done_flag = RESULTS_DIR / horizon_label / station / f"complete.{horizon_label}.flag"
            if done_flag.exists():
                print(f"⏭️  {station}-{horizon_label} ya completado, saltando...")
                continue

            rolling_csv = Path(str(rolling_csv_template).format(h=horizon_label))
            holdout_csv = Path(str(holdout_csv_template).format(h=horizon_label))

            fit_kwargs = {
                "max_epochs": args.tabnet_epochs if args.tabnet_epochs is not None else DEFAULTS["TABNET_MAX_EPOCHS"],
                "patience": args.tabnet_patience if args.tabnet_patience is not None else DEFAULTS["TABNET_PATIENCE"],
                "batch_size": args.tabnet_batch if args.tabnet_batch is not None else DEFAULTS["TABNET_BATCH"],
                "use_gpu": gpu_avail,
            }

            metrics_roll = rolling_backtest_tabnet(
                X_train_all, y_train_all,
                station=station, horizon_label=horizon_label,
                min_train=args.min_train,
                window=cfg["WINDOW"], purge=cfg["PURGE"], step=cfg["STEP"],
                max_windows=args.max_windows, scaler_ctor=RobustScaler,
                log_handle=lh, fit_kwargs=fit_kwargs,
            )

            metrics_roll.update({"Station": station, "Model": "C_TabNet", "Horizon": horizon_label})
            append_metrics_csv(rolling_csv, metrics_roll)
            log_write(lh, f"Rolling metrics saved: {metrics_roll}", also_console=True)
            print(f"    -> Rolling metrics: {metrics_roll}")

            try:
                log_write(lh, "Retraining final TabNet on 90% for holdout")
                scaler_final_x = RobustScaler().fit(X_train_all)
                X_train_all_s = scaler_final_x.transform(X_train_all).astype(np.float32)
                X_hold_s = scaler_final_x.transform(X_hold).astype(np.float32)

                from sklearn.preprocessing import RobustScaler as _RS
                scaler_final_y = _RS().fit(y_train_all.reshape(-1, 1))
                y_train_all_s = scaler_final_y.transform(y_train_all.reshape(-1, 1)).astype(np.float32).ravel()

                final_model = fit_tabnet(X_train_all_s, y_train_all_s, **fit_kwargs)

                preds_hold_s = final_model.predict(np.nan_to_num(X_hold_s, 0.0, 0.0, 0.0)).reshape(-1, 1)
                preds_hold = scaler_final_y.inverse_transform(preds_hold_s).ravel()

                metrics_hold = metrics_dict(y_hold, preds_hold)
                metrics_hold.update({"Station": station, "Model": "C_TabNet", "Horizon": horizon_label})
                append_metrics_csv(holdout_csv, metrics_hold)
                log_write(lh, f"Holdout metrics saved: {metrics_hold}", also_console=True)

                model_dir = RESULTS_DIR / horizon_label / station / "models"; ensure_dir(model_dir)
                try:
                    final_model.save_model(str(model_dir / "C_TabNet.zip"))
                except Exception as e:
                    log_write(lh, f"WARNING: no se pudo guardar modelo TabNet: {e}")
                pd.DataFrame({"y_true": y_hold, "y_pred": preds_hold}).to_csv(model_dir / "holdout_preds_C_TabNet.csv", index=False)

            except Exception as e:
                log_write(lh, f"ERROR en holdout: {e}")
                log_write(lh, traceback.format_exc())
                print("  ERROR en holdout (ver log)")

            try:
                with open(done_flag, "w", encoding="utf-8") as f:
                    f.write(f"Completed at {time.asctime()}\n")
            except Exception:
                pass

        elapsed = time.time() - start_sta
        log_write(lh, f"=== END station {station} ({elapsed/60:.1f} min) ===")
        lh.close()
        print(f">>> ✅ Estación {station} completada ({elapsed/60:.1f} min)")

    total_elapsed = time.time() - overall_t0
    print(f"\nAll stations & horizons finished in {total_elapsed/60:.1f} minutes. Results in {RESULTS_DIR}")

# ------------------------------
# CLI
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rolling backtest SOLO TabNet, alineado 1:1 con el pipeline original.")
    parser.add_argument("--data_dir", type=str, default="data/con_lags")
    parser.add_argument("--stations", type=str, default=None)
    parser.add_argument("--min_train", type=int, default=DEFAULTS["MIN_TRAIN"])
    parser.add_argument("--max_windows", type=int, default=None)
    parser.add_argument("--tabnet_epochs", type=int, default=DEFAULTS["TABNET_MAX_EPOCHS"]) 
    parser.add_argument("--tabnet_patience", type=int, default=DEFAULTS["TABNET_PATIENCE"]) 
    parser.add_argument("--tabnet_batch", type=int, default=DEFAULTS["TABNET_BATCH"]) 
    args = parser.parse_args()

    main(args)
