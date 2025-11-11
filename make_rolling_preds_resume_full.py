#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_rolling_preds_resume_full.py
Versión final y definitiva:
- Ejecuta 3 horizontes: corto / mediano / largo (configurables más abajo)
- 4 modelos: A_EvoXGB, B_XGB, C_TabNet, D_FTT
- GPU para XGBoost (device='cuda'), TabNet y FTTransformer si está disponible
- Reanudación automática por estación + horizonte (flags)
- Barra tqdm activa por ventanas
- Safe NaN handling, logs y guardado de resultados por horizonte
"""
import os
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

# xgboost (asume versión con soporte cuda, p.ej. 3.0.4)
from xgboost import XGBRegressor
try:
    from pytorch_tabnet.tab_model import TabNetRegressor
    TABNET_AVAILABLE = True
except Exception:
    TABNET_AVAILABLE = False

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ------------------------------
# Defaults / configuración global
# ------------------------------
RESULTS_DIR = Path("resultados_final_v2")
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Default "global" values (optimizados para reducir tiempo sin perder calidad)
DEFAULTS = {
    "MIN_TRAIN": 1500,      # mínimo para empezar (unas 3 semanas de datos)
    "WINDOW": 168,          # tamaño de ventana: 168 horas = 1 semana
    "PURGE": 24,            # purga de 24 horas para evitar leakage
    "STEP": 24,             # avanzar de 1 día en 1 día
    "MAX_WINDOWS": None,    # sin límite, recorre todo el dataset
    "XGB_N_EST": 200,
    "TABNET_MAX_EPOCHS": 100,
    "TABNET_PATIENCE": 10,
    "TABNET_BATCH": 256,
    "FTT_MAX_EPOCHS": 100,
    "FTT_PATIENCE": 10,
    "FTT_BATCH": 256,
    "FTT_TOKEN_DIM": 192,
    "FTT_NUM_LAYERS": 3,
    "FTT_NUM_HEADS": 4,
    "FTT_DROPOUT": 0.2,
    "EVOXGB_ENSEMBLE_SIZE": 4,
}


# Horizon configurations (optimizadas para reducir tiempo sin perder calidad)
HORIZONS = {
    "corto":   {"WINDOW": 168, "STEP": 24, "PURGE": 24},
    "mediano": {"WINDOW": 720, "STEP": 72, "PURGE": 72}
}


# ------------------------------
# Utilidades
# ------------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def log_write(handle, msg, also_console=False):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    line = f"[{ts}] {msg}\n"
    try:
        handle.write(line)
        handle.flush()
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
# Feature builder & data checks
# ------------------------------
def safe_read_and_build(path_csv: Path, log_handle=None):
    try:
        df = pd.read_csv(path_csv, parse_dates=["datetime"])
    except Exception as e:
        if log_handle:
            log_write(log_handle, f"ERROR reading {path_csv}: {e}")
        raise

    if "potencia" not in df.columns:
        if log_handle:
            log_write(log_handle, f"ERROR: columna 'potencia' no encontrada en {path_csv}. Columnas disponibles: {list(df.columns)}")
        return None

    df = df.sort_values("datetime").reset_index(drop=True)
    # Sin FutureWarning: usar ffill/bfill
    df = df.ffill().bfill()
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()

    # features
    df["hour"] = df["datetime"].dt.hour
    df["dow"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month

    for l in range(4, 25):
        df[f"potencia_lag_{l}h"] = df["potencia"].shift(l)

    df["pot_roll3"] = df["potencia"].rolling(3, min_periods=1).mean()
    df["pot_roll6"] = df["potencia"].rolling(6, min_periods=1).mean()
    df["pot_roll24"] = df["potencia"].rolling(24, min_periods=1).mean()

    df = df.ffill().bfill()
    return df

# ------------------------------
# Model wrappers (GPU-ready)
# ------------------------------
def fit_evo_xgb(X_train, y_train, n_steps=DEFAULTS["EVOXGB_ENSEMBLE_SIZE"], base_params=None, use_gpu=False):
    """
    Evo ensemble sobre residuales usando XGBoost como motor.
    use_gpu -> activa device='cuda' para XGBoost (si disponible)
    """
    if base_params is None:
        base_params = {
            "n_estimators": int(DEFAULTS["XGB_N_EST"] / max(1, n_steps)),
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "random_state": SEED,
            "tree_method": "hist",
        }

    if use_gpu:
        # API moderna XGBoost (>=2.0): tree_method hist + device cuda
        base_params.update({"tree_method": "hist", "device": "cuda"})

    models = []
    preds_train = np.zeros(len(y_train))
    residual = y_train - preds_train

    for i in range(n_steps):
        m = XGBRegressor(**base_params)
        m.fit(X_train, residual)
        pred_i = m.predict(X_train)
        preds_train += pred_i
        residual = y_train - preds_train
        models.append(m)

    class EvoXGBWrapper:
        def __init__(self, models):
            self.models = models
        def predict(self, X):
            p = np.zeros(X.shape[0])
            for m in self.models:
                p += m.predict(X)
            return p
        def save(self, path_prefix: Path):
            for idx, m in enumerate(self.models):
                try:
                    m.save_model(str(path_prefix / f"evoxgb_part{idx+1}.json"))
                except Exception:
                    import joblib
                    joblib.dump(m, path_prefix / f"evoxgb_part{idx+1}.pkl")
    return EvoXGBWrapper(models)

def fit_xgb(X_train, y_train, n_estimators=None, use_gpu=False):
    if n_estimators is None:
        n_estimators = DEFAULTS["XGB_N_EST"]
    params = {
        "n_estimators": n_estimators,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "random_state": SEED,
        "tree_method": "hist",
    }
    if use_gpu:
        params.update({"tree_method": "hist", "device": "cuda"})
    model = XGBRegressor(**params)
    model.fit(X_train, y_train, verbose=False)
    return model

def fit_tabnet(X_train, y_train, max_epochs=None, patience=None, batch_size=None, use_gpu=True):
    if not TABNET_AVAILABLE:
        raise RuntimeError("pytorch_tabnet no disponible en el entorno.")

    if max_epochs is None:
        max_epochs = DEFAULTS["TABNET_MAX_EPOCHS"]
    if patience is None:
        patience = DEFAULTS["TABNET_PATIENCE"]
    if batch_size is None:
        batch_size = DEFAULTS["TABNET_BATCH"]

    # 💥 Limpieza más agresiva de NaNs e infinitos
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)

    # 🔁 Normalización robusta dentro del modelo para estabilizar
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0) + 1e-8
    X_train = (X_train - X_mean) / X_std

    device_name = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"

    # ⚙️ Config más robusta
    model = TabNetRegressor(
        seed=SEED,
        device_name=device_name,
        optimizer_fn=torch.optim.AdamW,
        optimizer_params=dict(lr=2e-3, weight_decay=1e-4),
        scheduler_params={"step_size": 15, "gamma": 0.85},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        n_d=32, n_a=32,
        n_steps=4,
        gamma=1.5,
        lambda_sparse=1e-4,
        clip_value=1.0,
        verbose=0
    )

    # División 85/15
    inner_split = int(0.85 * len(X_train))
    X_tr, X_val = X_train[:inner_split], X_train[inner_split:]
    y_tr, y_val = y_train[:inner_split], y_train[inner_split:]

    # Seguridad adicional por NaNs
    X_tr = np.nan_to_num(X_tr)
    X_val = np.nan_to_num(X_val)
    y_tr = np.nan_to_num(y_tr)
    y_val = np.nan_to_num(y_val)

    try:
        model.fit(
            X_tr, y_tr.reshape(-1, 1),
            eval_set=[(X_val, y_val.reshape(-1, 1))],
            patience=patience,
            max_epochs=max_epochs,
            batch_size=batch_size,
            virtual_batch_size=batch_size // 4,
            num_workers=0,
            drop_last=False,
            verbose=0
        )
    except Exception:
        with open(RESULTS_DIR / "tabnet_error_log.txt", "a", encoding="utf-8") as f:
            f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] TabNet Error: {traceback.format_exc()}\n")
        return None

    return model


# ------------------------------
# FTTransformer (PyTorch)
# ------------------------------
class FTTransformerImproved(nn.Module):
    def __init__(self, num_features, num_targets=1, token_dim=DEFAULTS["FTT_TOKEN_DIM"],
                 num_layers=DEFAULTS["FTT_NUM_LAYERS"], num_heads=DEFAULTS["FTT_NUM_HEADS"],
                 dropout=DEFAULTS["FTT_DROPOUT"], embed_dropout=0.1):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(num_features, token_dim),
            nn.GELU(),
            nn.Dropout(embed_dropout),
            nn.LayerNorm(token_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(token_dim)
        self.head = nn.Sequential(
            nn.Linear(token_dim, token_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim // 2, num_targets)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        x = self.norm(x)
        return self.head(x)

def fit_fttransformer(X_train, y_train, 
                      max_epochs=None, patience=None, batch_size=None, 
                      token_dim=None, num_layers=None, num_heads=None, dropout=None, 
                      use_gpu=True):

    if max_epochs is None:
        max_epochs = DEFAULTS["FTT_MAX_EPOCHS"]
    if patience is None:
        patience = DEFAULTS["FTT_PATIENCE"]
    if batch_size is None:
        batch_size = DEFAULTS["FTT_BATCH"]
    if token_dim is None:
        token_dim = DEFAULTS["FTT_TOKEN_DIM"]
    if num_layers is None:
        num_layers = DEFAULTS["FTT_NUM_LAYERS"]
    if num_heads is None:
        num_heads = DEFAULTS["FTT_NUM_HEADS"]
    if dropout is None:
        dropout = DEFAULTS["FTT_DROPOUT"]

    # 💪 Fuerza uso de GPU si está disponible
    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    print(f"[FTTransformer] Usando dispositivo: {device}")

    model = FTTransformerImproved(
        num_features=X_train.shape[1],
        token_dim=token_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    # Split 85/15 para validación
    inner_split = int(0.85 * len(X_train))
    X_tr, y_tr = X_train[:inner_split], y_train[:inner_split]
    X_val, y_val = X_train[inner_split:], y_train[inner_split:]

    # 🔥 Mueve todo a GPU ANTES del entrenamiento
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

    train_ds = TensorDataset(X_tr_t, y_tr_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    best_loss = float("inf")
    wait = 0
    best_state = None

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            preds = model(xb)
            loss = loss_fn(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validación
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = loss_fn(val_preds, y_val_t).item()

        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

        if epoch % 10 == 0 or epoch == max_epochs - 1:
            print(f"  [Epoch {epoch+1}/{max_epochs}] train_loss={total_loss/len(train_loader):.4f}, val_loss={val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # 🧠 Devuelve el modelo en GPU listo para predecir
    model.to(device)
    return model


# ------------------------------
# Rolling backtest (sin cambios funcionales principales)
# ------------------------------
def rolling_backtest(X, y, fit_model_fn, station, model_name, horizon_label,
                     is_torch=False, is_tabnet=False,
                     min_train=2000, window=24, purge=1, step=1,
                     max_windows=None, scaler_ctor=RobustScaler, log_handle=None,
                     fit_kwargs=None):
    """
    Ejecuta rolling backtest para un solo horizonte (configurado por window/step/purge).
    horizon_label -> 'corto'|'mediano'|'largo' (usar para logs/rutas)
    """
    n = len(X)
    preds = np.full(n, np.nan)
    starts = list(range(min_train, n - window + 1, step))
    total_windows = len(starts)
    if max_windows is not None:
        starts = starts[:max_windows]
        total_windows = len(starts)

    if log_handle:
        log_write(log_handle, f"    Rolling ({horizon_label}): {total_windows} windows (MIN_TRAIN={min_train}, WINDOW={window}, STEP={step}, PURGE={purge})")

    last_model = None
    last_scaler = None

    pbar = tqdm(enumerate(starts, 1), total=len(starts),
                desc=f"{station}-{model_name}-{horizon_label}",
                unit="win", file=sys.stdout)
    for w_idx, start in pbar:
        
        # 🔄 Si hay un punto de reanudación, saltar ventanas previas
        if fit_kwargs.get("resume_from", 0) > 0 and w_idx <= fit_kwargs["resume_from"]:
            continue

        # --- 🔄 Reanudar si ya existen predicciones parciales guardadas ---
        station_dir = RESULTS_DIR / horizon_label / station / "predictions"
        ensure_dir(station_dir)
        partial_path = station_dir / f"preds_{model_name}.csv"

        if partial_path.exists():
            try:
                done_rows = sum(1 for _ in open(partial_path)) - 1  # líneas ya guardadas
                if w_idx <= done_rows:
                    if log_handle:
                        log_write(log_handle, f"      ⏭️ Reanudando: ventana {w_idx} ya procesada previamente, se omite.")
                    continue
            except Exception as e:
                if log_handle:
                    log_write(log_handle, f"      ⚠️ Error verificando progreso previo: {e}")
        # ----------------------------------------------------------
                # 🔄 Reanudar: si hay progreso previo guardado, saltar ventanas ya completadas
        progress_path = RESULTS_DIR / horizon_label / station / f"progress_{model_name}.txt"
        if progress_path.exists():
            try:
                last_done = int(open(progress_path).read().strip())
                if w_idx <= last_done:
                    continue
            except:
                pass
        t_window_start = time.time()
        try:
            train_idx = np.arange(0, start - purge, dtype=int)
            test_idx = np.arange(start, start + window, dtype=int)
            if len(train_idx) < min_train:
                if log_handle:
                    log_write(log_handle, f"      Skipping window {w_idx}: train too pequeño ({len(train_idx)} < {min_train})")
                continue

            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te = X[test_idx]

            scaler = scaler_ctor().fit(X_tr)
            X_tr_s = scaler.transform(X_tr)
            X_te_s = scaler.transform(X_te)

            if log_handle:
                log_write(log_handle, f"      Window {w_idx}/{total_windows}: train {len(X_tr)} -> test {len(X_te)}")

            t_fit0 = time.time()
            try:
                if fit_kwargs:
                    model = fit_model_fn(X_tr_s, y_tr, **fit_kwargs)
                else:
                    model = fit_model_fn(X_tr_s, y_tr)
            except Exception as e:
                if log_handle:
                    log_write(log_handle, f"      ERROR training window {w_idx}: {e}")
                    log_write(log_handle, traceback.format_exc())
                continue
            t_fit1 = time.time()

            if model is None:
                if log_handle:
                    log_write(log_handle, f"      WARNING: model returned None for window {w_idx}. Skipping predictions for this window.")
                continue

            try:
                if is_torch:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model_device = next(model.parameters()).device if any(p.requires_grad for p in model.parameters()) else device
                    model.eval()
                    with torch.no_grad():
                        X_te_tensor = torch.tensor(X_te_s, dtype=torch.float32).to(model_device)
                        preds_vals = model(X_te_tensor).cpu().numpy().flatten()
                        preds[test_idx] = preds_vals
                elif is_tabnet:
                    preds_vals = model.predict(X_te_s).flatten()
                    preds[test_idx] = preds_vals
                else:
                    # intentar usar DMatrix en cuda para evitar warnings de device mismatch
                    try:
                        import xgboost as xgb
                        dtest = xgb.DMatrix(X_te_s, device='cuda')
                        preds_vals = model.predict(dtest)
                    except Exception:
                        preds_vals = model.predict(X_te_s)
                    preds[test_idx] = preds_vals
                     # 🔄 Guardar progreso actual
                progress_path = RESULTS_DIR / horizon_label / station / f"progress_{model_name}.txt"
                try:
                    with open(progress_path, "w") as f:
                        f.write(str(w_idx))
                    if log_handle:
                        log_write(log_handle, f"      💾 Progreso guardado (ventana {w_idx})")
                except Exception as e:
                    if log_handle:
                        log_write(log_handle, f"      ⚠️ No se pudo guardar progreso: {e}")
            except Exception as e:
                if log_handle:
                    log_write(log_handle, f"      ERROR predicting window {w_idx}: {e}")
                    log_write(log_handle, traceback.format_exc())
                continue

            t_window_end = time.time()
            last_model = model
            last_scaler = scaler

            elapsed_fit = t_fit1 - t_fit0
            elapsed_window = t_window_end - t_window_start
            if log_handle:
                log_write(log_handle, f"      Window {w_idx} done (fit {elapsed_fit:.1f}s, total {elapsed_window:.1f}s)")

        except Exception as e:
            if log_handle:
                log_write(log_handle, f"      UNEXPECTED ERROR in window {w_idx}: {e}")
                log_write(log_handle, traceback.format_exc())
            continue


    # guardar preds parciales por horizonte
    station_dir = RESULTS_DIR / horizon_label / station / "predictions"
    ensure_dir(station_dir)
    out_path = station_dir / f"preds_{model_name}.csv"
    pd.DataFrame({"y_true": y, "y_pred": preds}).to_csv(out_path, index=False)
    if log_handle:
        log_write(log_handle, f"    Saved predictions partial to {out_path}")

    mask = ~np.isnan(preds)
    if mask.sum() == 0:
        metrics = {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan"), "%Tol": float("nan")}
    else:
        metrics = metrics_dict(y[mask], preds[mask])

    return metrics, last_model, last_scaler

# ------------------------------
# MAIN pipeline
# ------------------------------
def main(args):
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir {data_dir} no existe.")

    # archivos disponibles (stem)
    files = sorted([f.stem for f in data_dir.glob("*.csv")])
    if args.stations:
        requested = [s.strip() for s in args.stations.split(",") if s.strip()]
        stations = [s for s in requested if s in files]
        missing = [s for s in requested if s not in files]
        if missing:
            print(f"Advertencia: estaciones solicitadas no encontradas: {missing}")
    else:
        stations = files

    # CSVs globales (guardados por horizonte)
    for h in HORIZONS.keys():
        ensure_dir(RESULTS_DIR / h)
    rolling_csv_template = RESULTS_DIR / "{h}" / "metrics_rolling.csv"
    holdout_csv_template = RESULTS_DIR / "{h}" / "metrics_holdout.csv"

    total = len(stations)
    overall_t0 = time.time()

    gpu_avail = torch.cuda.is_available()
    print(f"GPU disponible: {gpu_avail}")
    if not gpu_avail:
        print("ADVERTENCIA: GPU no detectada. Algunos modelos entrenarán en CPU y será más lento.")

    # recorrer estaciones
    for idx, station in enumerate(stations, 1):
        start_sta = time.time()
        print(f"\n>>> [{idx}/{total}] Procesando estación: {station}")

        # cargar y construir features
        csv_path = data_dir / f"{station}.csv"
        log_dir = RESULTS_DIR / "logs"
        ensure_dir(log_dir)
        log_file = log_dir / f"log_{station}.txt"
        lh = open(log_file, "a", buffering=1, encoding="utf-8")
        log_write(lh, f"=== START {time.asctime()} ===")
        log_write(lh, f"Station {station} ({idx}/{total})", also_console=True)

        try:
            df = safe_read_and_build(csv_path, log_handle=lh)
        except Exception as e:
            log_write(lh, f"Fallo reading/building features para {station}: {e}")
            lh.close()
            continue

        if df is None:
            log_write(lh, "Skipping station due to missing 'potencia' or other critical error.")
            lh.close()
            continue

        feature_cols = [c for c in df.columns if c.startswith("potencia_lag_")] + \
                       ["pot_roll3", "pot_roll6", "pot_roll24", "hour", "dow", "month"]

        # split 90/10 como antes (holdout final)
        split_idx = int(0.9 * len(df))
        df_train = df.iloc[:split_idx].reset_index(drop=True)
        df_hold = df.iloc[split_idx:].reset_index(drop=True)

        X_train_all_full, y_train_all_full = df_train[feature_cols].values, df_train["potencia"].values
        X_hold, y_hold = df_hold[feature_cols].values, df_hold["potencia"].values

        # modelos definidos
        models = [
            ("A_EvoXGB", fit_evo_xgb, False, False),
            ("B_XGB", fit_xgb, False, False),
            ("C_TabNet", fit_tabnet, False, True),
            ("D_FTT", fit_fttransformer, True, False),
        ]

        # Recorremos horizontes (corto, mediano, largo)
        for horizon_label, cfg in HORIZONS.items():
            print(f"\n>>>   --- Horizonte: {horizon_label} (WINDOW={cfg['WINDOW']}, STEP={cfg['STEP']}, PURGE={cfg['PURGE']})")
            # crear directorios por horizonte
            ensure_dir(RESULTS_DIR / horizon_label / station)
            done_flag = RESULTS_DIR / horizon_label / station / f"complete.{horizon_label}.flag"
            # reanudación por horizonte
            if done_flag.exists():
                print(f"⏭️ Saltando {station} - {horizon_label} (ya completado)")
                continue

            # precomputar X,y para el rolling de este horizonte (usamos df_train completo)
            X_train_all = X_train_all_full.copy()
            y_train_all = y_train_all_full.copy()

            # archivos de métricas para este horizonte
            rolling_csv = Path(str(rolling_csv_template).format(h=horizon_label))
            holdout_csv = Path(str(holdout_csv_template).format(h=horizon_label))

            # Ejecutar cada modelo
            for model_name, fit_fn, is_torch, is_tabnet in models:
                model_t0 = time.time()
                log_write(lh, f"-> Starting model {model_name} (horizon={horizon_label})", also_console=True)
                print(f"  -> Entrenando {model_name} (rolling)")
            # 🔄 Skip si ya existe archivo de predicciones para este modelo y horizonte
                preds_path = RESULTS_DIR / horizon_label / station / "predictions" / f"preds_{model_name}.csv"
                if preds_path.exists():
                   print(f"⏭️  {station} - {model_name} ({horizon_label}) ya tiene predicciones guardadas, se omite reentrenamiento.")
                   log_write(lh, f"⏭️  {station} - {model_name} ({horizon_label}) ya tiene predicciones guardadas, se omite reentrenamiento.", also_console=True)
                   continue
                fit_kwargs = {}
                if model_name == "A_EvoXGB":
                    fit_kwargs = {"n_steps": args.evoxgb_steps} if getattr(args, "evoxgb_steps", None) else {"n_steps": DEFAULTS["EVOXGB_ENSEMBLE_SIZE"]}
                    fit_kwargs["use_gpu"] = gpu_avail
                if model_name == "B_XGB":
                    fit_kwargs = {"n_estimators": args.xgb_n_estim} if getattr(args, "xgb_n_estim", None) else {"n_estimators": DEFAULTS["XGB_N_EST"]}
                    fit_kwargs["use_gpu"] = gpu_avail
                if model_name == "C_TabNet":
                    fit_kwargs = {
                        "max_epochs": args.tabnet_epochs if args.tabnet_epochs is not None else DEFAULTS["TABNET_MAX_EPOCHS"],
                        "patience": args.tabnet_patience if args.tabnet_patience is not None else DEFAULTS["TABNET_PATIENCE"],
                        "batch_size": args.tabnet_batch if args.tabnet_batch is not None else DEFAULTS["TABNET_BATCH"],
                        "use_gpu": gpu_avail
                    }
                if model_name == "D_FTT":
                    fit_kwargs = {
                        "max_epochs": args.ftt_epochs if args.ftt_epochs is not None else DEFAULTS["FTT_MAX_EPOCHS"],
                        "patience": args.ftt_patience if args.ftt_patience is not None else DEFAULTS["FTT_PATIENCE"],
                        "batch_size": args.ftt_batch if args.ftt_batch is not None else DEFAULTS["FTT_BATCH"],
                        "token_dim": args.ftt_token_dim if args.ftt_token_dim is not None else DEFAULTS["FTT_TOKEN_DIM"],
                        "num_layers": args.ftt_num_layers if args.ftt_num_layers is not None else DEFAULTS["FTT_NUM_LAYERS"],
                        "num_heads": args.ftt_num_heads if args.ftt_num_heads is not None else DEFAULTS["FTT_NUM_HEADS"],
                        "dropout": args.ftt_dropout if args.ftt_dropout is not None else DEFAULTS["FTT_DROPOUT"],
                        "use_gpu": gpu_avail
                    }

                # Rolling para este horizonte y modelo
                try:
                    metrics_roll, trained_model, trained_scaler = rolling_backtest(
                        X_train_all, y_train_all,
                        fit_model_fn=fit_fn,
                        station=station,
                        model_name=model_name,
                        horizon_label=horizon_label,
                        is_torch=is_torch,
                        is_tabnet=is_tabnet,
                        min_train=args.min_train,
                        window=cfg["WINDOW"],
                        purge=cfg["PURGE"],
                        step=cfg["STEP"],
                        max_windows=args.max_windows,
                        scaler_ctor=RobustScaler,
                        log_handle=lh,
                        fit_kwargs=fit_kwargs
                    )
                except Exception as e:
                    log_write(lh, f"ERROR during rolling for {model_name} horizon {horizon_label}: {e}")
                    log_write(lh, traceback.format_exc())
                    print(f"  ERROR during rolling for {model_name}. See log.")
                    continue

                metrics_roll["Station"] = station
                metrics_roll["Model"] = model_name
                metrics_roll["Horizon"] = horizon_label
                append_metrics_csv(rolling_csv, metrics_roll)
                log_write(lh, f"  Rolling metrics saved: {metrics_roll}", also_console=True)
                print(f"    -> Rolling metrics: {metrics_roll}")

                # Retrain final model on full 90% and evaluate holdout (misma lógica)
                try:
                    log_write(lh, f"  Retraining final model on full 90% for holdout evaluation (horizon={horizon_label})")
                    final_scaler = RobustScaler().fit(X_train_all)
                    X_train_all_s = final_scaler.transform(X_train_all)
                    X_hold_s = final_scaler.transform(X_hold)

                    if fit_kwargs:
                        final_model = fit_fn(X_train_all_s, y_train_all, **fit_kwargs)
                    else:
                        final_model = fit_fn(X_train_all_s, y_train_all)

                    if final_model is None:
                        log_write(lh, f"  Final model for {model_name} is None (skipping holdout evaluation)")
                        continue

                    if is_torch:
                        device = torch.device("cuda" if gpu_avail else "cpu")
                        final_model.eval()
                        with torch.no_grad():
                            preds_hold = final_model(torch.tensor(X_hold_s, dtype=torch.float32).to(device)).cpu().numpy().flatten()
                    elif is_tabnet:
                        preds_hold = final_model.predict(X_hold_s).flatten()
                    else:
                        try:
                            import xgboost as xgb
                            dtest = xgb.DMatrix(X_hold_s, device='cuda')
                            preds_hold = final_model.predict(dtest)
                        except Exception:
                            preds_hold = final_model.predict(X_hold_s)

                    metrics_hold = metrics_dict(y_hold, preds_hold)
                    metrics_hold["Station"] = station
                    metrics_hold["Model"] = model_name
                    metrics_hold["Horizon"] = horizon_label
                    append_metrics_csv(holdout_csv, metrics_hold)
                    log_write(lh, f"  Holdout metrics saved: {metrics_hold}", also_console=True)
                    print(f"    -> Holdout metrics: {metrics_hold}")

                    # save final model & scaler per horizon
                    model_dir = RESULTS_DIR / horizon_label / station / "models"
                    ensure_dir(model_dir)
                    try:
                        if model_name == "A_EvoXGB":
                            save_prefix = model_dir / f"{model_name}"
                            ensure_dir(save_prefix)
                            try:
                                final_model.save(save_prefix)
                            except Exception:
                                pass
                        elif model_name == "C_TabNet" and TABNET_AVAILABLE:
                            final_model.save_model(str(model_dir / f"{model_name}.zip"))
                        elif is_torch:
                            torch.save(final_model.state_dict(), model_dir / f"{model_name}.pth")
                        else:
                            try:
                                final_model.save_model(str(model_dir / f"{model_name}.json"))
                            except Exception:
                                import joblib
                                joblib.dump(final_model, model_dir / f"{model_name}.pkl")
                        try:
                            import joblib
                            joblib.dump(final_scaler, model_dir / f"{model_name}_scaler.pkl")
                        except Exception:
                            pd.DataFrame({"info": [f"Could not save scaler with joblib for {model_name}"]}).to_csv(model_dir / f"{model_name}_scaler_info.txt")
                    except Exception as e:
                        log_write(lh, f"  Warning: could not save final model {model_name}: {e}")

                    preds_hold_df = pd.DataFrame({"y_true": y_hold, "y_pred": preds_hold})
                    preds_hold_df.to_csv(model_dir / f"holdout_preds_{model_name}.csv", index=False)

                except Exception as e:
                    log_write(lh, f"  ERROR in holdout evaluation for {model_name}: {e}")
                    log_write(lh, traceback.format_exc())
                    print(f"  ERROR in holdout evaluation for {model_name}. See log.")

                model_t1 = time.time()
                log_write(lh, f"-> Model {model_name} (horizon={horizon_label}) finished in {(model_t1 - model_t0):.1f}s")
                print(f"  -> {model_name} done in {(model_t1 - model_t0):.1f}s")

            # marcar horizonte de la estación como completado
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
    parser = argparse.ArgumentParser(description="Rolling backtest final con 4 modelos, 3 horizontes y holdout. Script robusto y listo para producción.")
    parser.add_argument("--data_dir", type=str, default="data/con_lags", help="Directorio con CSVs por estación (unidad temporal debe ser por hora).")
    parser.add_argument("--stations", type=str, default=None, help="Lista separada por comas de estaciones a procesar (default: todas)")
    parser.add_argument("--min_train", type=int, default=DEFAULTS["MIN_TRAIN"])
    parser.add_argument("--window", type=int, default=DEFAULTS["WINDOW"])
    parser.add_argument("--purge", type=int, default=DEFAULTS["PURGE"])
    parser.add_argument("--step", type=int, default=DEFAULTS["STEP"], help="Paso entre inicios de ventanas (aumentar acelera)")
    parser.add_argument("--max_windows", type=int, default=DEFAULTS["MAX_WINDOWS"], help="Limita número de ventanas por modelo (None=infinito)")
    parser.add_argument("--xgb_n_estim", type=int, default=DEFAULTS["XGB_N_EST"])
    parser.add_argument("--evoxgb_steps", type=int, default=DEFAULTS["EVOXGB_ENSEMBLE_SIZE"])
    parser.add_argument("--tabnet_epochs", type=int, default=DEFAULTS["TABNET_MAX_EPOCHS"])
    parser.add_argument("--tabnet_patience", type=int, default=DEFAULTS["TABNET_PATIENCE"])
    parser.add_argument("--tabnet_batch", type=int, default=DEFAULTS["TABNET_BATCH"])
    parser.add_argument("--ftt_epochs", type=int, default=DEFAULTS["FTT_MAX_EPOCHS"])
    parser.add_argument("--ftt_patience", type=int, default=DEFAULTS["FTT_PATIENCE"])
    parser.add_argument("--ftt_batch", type=int, default=DEFAULTS["FTT_BATCH"])
    parser.add_argument("--ftt_token_dim", type=int, default=DEFAULTS["FTT_TOKEN_DIM"])
    parser.add_argument("--ftt_num_layers", type=int, default=DEFAULTS["FTT_NUM_LAYERS"])
    parser.add_argument("--ftt_num_heads", type=int, default=DEFAULTS["FTT_NUM_HEADS"])
    parser.add_argument("--ftt_dropout", type=float, default=DEFAULTS["FTT_DROPOUT"])

    args = parser.parse_args()

    # Actualizamos defaults con posibles flags (se mantienen pero usamos HORIZONS fijas)
    DEFAULTS["MIN_TRAIN"] = args.min_train
    DEFAULTS["WINDOW"] = args.window
    DEFAULTS["PURGE"] = args.purge
    DEFAULTS["STEP"] = args.step
    DEFAULTS["MAX_WINDOWS"] = args.max_windows

    if args.max_windows is None:
        args.max_windows = None

    main(args)
