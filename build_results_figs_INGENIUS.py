#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_results_figs_INGENIUS.py (robusto y editorial)
---------------------------------------------------
Genera tablas y figuras para el artículo comparando 4 modelos (EvoXGB, XGB, TabNet, FTT)
en una estación representativa y dos horizontes (p.ej. corto, mediano).

Salidas (en `--out_dir`, por defecto figuras_INGENIUS/):
- Tabla 1: tab1_rolling_metrics.csv  (rolling por modelo×horizonte, +%Tol@δ)
- Tabla 2: tab2_holdout_metrics.csv  (si existe)
- Fig 1a/b: Barras R² y RMSE (anotadas)
- Fig 2a: Serie real vs predicción (rolling, tramo final)
- Fig 2b: Serie real vs predicción (rolling, zoom 168 puntos)
- Fig 3 : Scatter combinado (ambos horizontes superpuestos por modelo)
- Fig 4 : Sensibilidad de %Tol (δ={2,5,10,15}%)

Uso típico:
python build_results_figs_INGENIUS.py ^
  --models "A_EvoXGB=.\resultados_final_v2;B_XGB=.\resultados_final_v2;C_TabNet=.\resultados_tabnet_SOLO;D_FTT=.\resultados_final_v2" ^
  --station "S-E EL CALVARIO_con_lags" ^
  --horizons "corto,mediano" ^
  --delta_tol 0.10 ^
  --last_points 672 ^
  --out_dir figuras_INGENIUS
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 10})


# -------------------------------
# Utilidades
# -------------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def savefig_dual(fig, path_png: Path):
    """Guarda PNG y PDF (mismo nombre)."""
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    try:
        fig.savefig(path_png.with_suffix(".pdf"), bbox_inches="tight")
    except Exception:
        pass

def norm(s: str) -> str:
    return s.lower().replace("-", "_").replace(" ", "_")

def infer_expected_token(model_label: str) -> str:
    ml = norm(model_label)
    if "evoxgb" in ml or ("evo" in ml and "xgb" in ml): return "evoxgb"
    if "tabnet" in ml: return "tabnet"
    if "ftt" in ml or "transformer" in ml: return "ftt"
    if "xgb" in ml: return "xgb"
    return ml

ALL_TOKENS = ["evoxgb", "xgb", "tabnet", "ftt"]

def choose_preds_file(preds_dir: Path, expected_token: str):
    """Elige el archivo preds correcto incluso si hay varios."""
    if not preds_dir.exists(): return None
    files = sorted(preds_dir.glob("preds_*.csv"))
    if not files: return None
    exp = norm(expected_token)

    exact = [f for f in files if f.name.lower() == f"preds_{exp}.csv"]
    if exact: return exact[0]

    partial = [f for f in files if exp in f.name.lower()]
    if partial: return partial[0]

    others = [t for t in ALL_TOKENS if t != exp]
    for f in files:
        name = f.name.lower()
        if any(t in name for t in others):
            continue
        return f
    return files[0]

def read_preds_for_model(preds_dir: Path, expected_token: str):
    f = choose_preds_file(preds_dir, expected_token)
    if f is None: return None
    try:
        df = pd.read_csv(f)
        if {'y_true','y_pred'}.issubset(df.columns):
            out = df[['y_true','y_pred']].copy()
        else:
            out = df.iloc[:, :2].copy(); out.columns = ['y_true','y_pred']
        out = out.replace([np.inf, -np.inf], np.nan)
        return out
    except Exception:
        return None

def mae_rmse_r2(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    m = np.isfinite(y) & np.isfinite(yhat)
    if m.sum() == 0: return np.nan, np.nan, np.nan
    mae = np.mean(np.abs(y[m]-yhat[m]))
    rmse = np.sqrt(np.mean((y[m]-yhat[m])**2))
    ybar = np.mean(y[m])
    ss_res = np.sum((y[m]-yhat[m])**2)
    ss_tot = np.sum((y[m]-ybar)**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    return mae, rmse, r2

def pct_tol_rel(y, yhat, delta=0.10):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    m = np.isfinite(y) & np.isfinite(yhat)
    if m.sum() == 0: return np.nan
    thr = delta * np.abs(y[m])
    ok = np.abs(yhat[m]-y[m]) <= thr
    return float(np.mean(ok)*100)


# -------------------------------
# Carga rolling & holdout
# -------------------------------
def load_model_horizon(results_dir: Path, station: str, horizon: str, expected_token: str):
    preds_dir = results_dir / horizon / station / "predictions"
    preds = read_preds_for_model(preds_dir, expected_token)

    hold_csv = results_dir / horizon / "metrics_holdout.csv"
    hold = None
    if hold_csv.exists():
        try:
            d = pd.read_csv(hold_csv)
            if "Station" in d.columns:
                d = d[d["Station"] == station]
            hold = d
        except Exception:
            pass
    return preds, hold


# -------------------------------
# Figuras
# -------------------------------
def _annotate_bars(ax):
    for p in ax.patches:
        h = p.get_height()
        if np.isfinite(h):
            ax.text(p.get_x() + p.get_width()/2, h,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8)

def fig_barras_r2_rmse(df_summary, out_dir: Path):
    ensure_dir(out_dir)
    models   = sorted(df_summary['Model'].unique())
    horizons = sorted(df_summary['Horizon'].unique())
    x = np.arange(len(models))
    width = 0.18

    # ----- R2 -----
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for i, h in enumerate(horizons):
        vals = []
        for m in models:
            mask = (df_summary['Model']==m) & (df_summary['Horizon']==h)
            vals.append(float(df_summary.loc[mask, 'R2'].iloc[0]) if mask.any() else np.nan)
        ax.bar(x+(i-(len(horizons)-1)/2)*width, vals, width=width, label=h,
               edgecolor="black", linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("R²"); ax.set_title("Desempeño (R²) por modelo y horizonte (rolling)")
    ax.set_ylim(0, 1.0); ax.legend(title="Horizonte")
    _annotate_bars(ax)
    plt.tight_layout(); savefig_dual(fig, out_dir/"fig1a_barras_R2.png"); plt.close(fig)

    # ----- RMSE -----
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    max_rmse = float(np.nanmax(df_summary['RMSE'].values)) if not df_summary.empty else None
    for i, h in enumerate(horizons):
        vals = []
        for m in models:
            mask = (df_summary['Model']==m) & (df_summary['Horizon']==h)
            vals.append(float(df_summary.loc[mask, 'RMSE'].iloc[0]) if mask.any() else np.nan)
        ax.bar(x+(i-(len(horizons)-1)/2)*width, vals, width=width, label=h,
               edgecolor="black", linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("RMSE"); ax.set_title("Desempeño (RMSE) por modelo y horizonte (rolling)")
    if max_rmse and np.isfinite(max_rmse):
        ax.set_ylim(0, max_rmse*1.1)
    ax.legend(title="Horizonte")
    _annotate_bars(ax)
    plt.tight_layout(); savefig_dual(fig, out_dir/"fig1b_barras_RMSE.png"); plt.close(fig)


def fig_timeseries_all_models(preds_dict, horizons, out_dir: Path, last_points=672):
    ensure_dir(out_dir)
    models = sorted({k[0] for k in preds_dict.keys()})
    for h in horizons:
        ref = None
        for m in models:
            dfm = preds_dict.get((m, h))
            if dfm is not None and not dfm.empty:
                ref = dfm.copy(); break
        if ref is None: continue
        y_true = ref.loc[np.isfinite(ref['y_true']), 'y_true'].tail(last_points)

        fig, ax = plt.subplots(figsize=(12, 3.6))
        ax.plot(y_true.values, label="Real", linewidth=1.4)
        for m in models:
            dfm = preds_dict.get((m, h))
            if dfm is None or dfm.empty: continue
            v = dfm.loc[np.isfinite(dfm['y_true']) & np.isfinite(dfm['y_pred']), 'y_pred'].tail(len(y_true))
            ax.plot(v.values, label=m, linewidth=1.0, alpha=0.95)
        ax.set_title(f"Estación representativa – Serie real vs predicción (rolling, {h})")
        ax.set_xlabel("Índice temporal"); ax.set_ylabel("Potencia")
        ax.legend(ncol=min(5, len(models)+1), fontsize="small")
        plt.tight_layout(); savefig_dual(fig, out_dir/f"fig2_timeseries_{h}.png"); plt.close(fig)


def fig_timeseries_zoom(preds_dict, horizons, out_dir: Path, zoom_points=168):
    ensure_dir(out_dir)
    models = sorted({k[0] for k in preds_dict.keys()})
    for h in horizons:
        ref = None
        for m in models:
            d = preds_dict.get((m, h))
            if d is not None and not d.empty:
                ref = d.copy(); break
        if ref is None: continue
        y_true = ref.loc[np.isfinite(ref['y_true']), 'y_true'].tail(zoom_points)

        fig, ax = plt.subplots(figsize=(12, 3.6))
        ax.plot(y_true.values, label="Real", linewidth=1.4)
        for m in models:
            d = preds_dict.get((m, h))
            if d is None or d.empty: continue
            v = d.loc[np.isfinite(d['y_true']) & np.isfinite(d['y_pred']), 'y_pred'].tail(len(y_true))
            ax.plot(v.values, label=m, linewidth=1.0, alpha=0.95)
        ax.set_title(f"Estación representativa – Serie (zoom, {h}, últimos {zoom_points})")
        ax.set_xlabel("Índice temporal"); ax.set_ylabel("Potencia")
        ax.legend(ncol=min(5, len(models)+1), fontsize="small")
        plt.tight_layout(); savefig_dual(fig, out_dir/f"fig2b_timeseries_zoom_{h}.png"); plt.close(fig)


def fig_scatter_combined(preds_dict, horizons, out_dir: Path):
    """Un solo lienzo: 1 subplot por modelo con los dos horizontes superpuestos."""
    ensure_dir(out_dir)
    models = sorted({k[0] for k in preds_dict.keys()})
    if len(horizons) < 2: return

    # límites globales
    lo, hi = np.inf, -np.inf
    for (m, h), d in preds_dict.items():
        if d is None or d.empty: continue
        v = d.loc[np.isfinite(d['y_true']) & np.isfinite(d['y_pred'])]
        if v.empty: continue
        lo = min(lo, v['y_true'].min(), v['y_pred'].min())
        hi = max(hi, v['y_true'].max(), v['y_pred'].max())
    if not np.isfinite(lo) or not np.isfinite(hi): return

    cols = 2; rows = int(np.ceil(len(models)/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows), squeeze=False)
    # colores por horizonte (matplotlib defaults)
    colors = {horizons[0]: None, horizons[1]: None}

    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r][c]
            if idx >= len(models):
                ax.axis("off"); continue
            m = models[idx]; idx += 1
            legends = []
            for h in horizons:
                d = preds_dict.get((m, h))
                if d is None or d.empty: continue
                v = d.loc[np.isfinite(d['y_true']) & np.isfinite(d['y_pred'])]
                sc = ax.scatter(v['y_true'], v['y_pred'], s=8, alpha=0.28, label=h)
                if colors[h] is None: colors[h] = sc.get_facecolors()[0]
                mae, rmse, r2 = mae_rmse_r2(v['y_true'], v['y_pred'])
                legends.append(f"{h}: R²={r2:.3f}, MAE={mae:.1f}, RMSE={rmse:.1f}")
            ax.plot([lo, hi], [lo, hi], "--", linewidth=1, color="gray")
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_xlabel("Real"); ax.set_ylabel("Predicción")
            ax.set_title(f"{m}\n" + " | ".join(legends))
    fig.suptitle("Real vs Predicción (rolling) — ambos horizontes por modelo", y=0.995, fontsize=12)
    plt.tight_layout(rect=[0,0,1,0.97]); savefig_dual(fig, out_dir/"fig3_scatter_combined.png"); plt.close(fig)


def fig_sens_tol_all(preds_dict, horizons, out_dir: Path):
    ensure_dir(out_dir)
    deltas = [0.02, 0.05, 0.10, 0.15]
    models = sorted({k[0] for k in preds_dict.keys()})
    fig, axes = plt.subplots(1, len(horizons), figsize=(7.2*len(horizons), 4.0), squeeze=False)
    for j, h in enumerate(sorted(horizons)):
        ax = axes[0][j]
        x = np.arange(len(models)); width = 0.18
        for i, dlt in enumerate(deltas):
            vals = []
            for m in models:
                d = preds_dict.get((m, h))
                if d is None or d.empty:
                    vals.append(np.nan)
                else:
                    vals.append(pct_tol_rel(d['y_true'], d['y_pred'], delta=dlt))
            ax.bar(x + (i-(len(deltas)-1)/2)*width, vals, width=width,
                   label=f"δ={int(dlt*100)}%", edgecolor='black', linewidth=0.3)
        ax.set_xticks(x); ax.set_xticklabels(models)
        ax.set_ylabel('%Tol'); ax.set_title(f'Sensibilidad de %Tol (rolling, {h})')
        ax.legend(title='Umbral relativo')
    plt.tight_layout(); savefig_dual(fig, out_dir/'fig4_sens_tol.png'); plt.close(fig)


# -------------------------------
# Main
# -------------------------------
def main():
    ap = argparse.ArgumentParser(description='Figuras/tablas compactas (4 modelos, 1 estación, 2 horizontes).')
    ap.add_argument('--models', type=str, required=True,
                    help='modelo=path;modelo=path ... Ej: "A_EvoXGB=./resultados_final_v2;B_XGB=./resultados_final_v2;C_TabNet=./resultados_tabnet_SOLO;D_FTT=./resultados_final_v2"')
    ap.add_argument('--station', type=str, required=True)
    ap.add_argument('--horizons', type=str, default='corto,mediano')
    ap.add_argument('--delta_tol', type=float, default=0.10)
    ap.add_argument('--last_points', type=int, default=672)
    ap.add_argument('--out_dir', type=str, default='figuras_INGENIUS')
    args = ap.parse_args()

    out_dir = Path(args.out_dir); ensure_dir(out_dir)

    # Parse modelos
    model_map = {}
    for part in args.models.split(';'):
        part = part.strip()
        if not part or '=' not in part:
            continue
        k, v = part.split('=', 1)
        model_map[k.strip()] = Path(v.strip())

    horizons = [h.strip() for h in args.horizons.split(',') if h.strip()]

    preds_dict, holdout_rows, rolling_rows = {}, [], []

    for model_label, rdir in model_map.items():
        expected_token = infer_expected_token(model_label)
        for h in horizons:
            preds, hold = load_model_horizon(rdir, args.station, h, expected_token)
            if preds is not None:
                preds_dict[(model_label, h)] = preds
                mae, rmse, r2 = mae_rmse_r2(preds['y_true'], preds['y_pred'])
                ptol = pct_tol_rel(preds['y_true'], preds['y_pred'], delta=args.delta_tol)
                rolling_rows.append({'Model': model_label, 'Horizon': h,
                                     'MAE': mae, 'RMSE': rmse, 'R2': r2,
                                     f'%Tol@{int(args.delta_tol*100)}%': ptol})
            if hold is not None and not hold.empty:
                last = hold.tail(1)
                row = {'Model': model_label, 'Horizon': h,
                       'MAE': float(last.get('MAE', np.nan).values[0]) if 'MAE' in last else np.nan,
                       'RMSE': float(last.get('RMSE', np.nan).values[0]) if 'RMSE' in last else np.nan,
                       'R2': float(last.get('R2', np.nan).values[0]) if 'R2' in last else np.nan}
                holdout_rows.append(row)

    # Tablas
    tab1 = pd.DataFrame(rolling_rows)
    tab1_path = out_dir / 'tab1_rolling_metrics.csv'
    tab1.to_csv(tab1_path, index=False)
    print(f'✅ Tabla 1 (rolling) -> {tab1_path}')

    tab2 = pd.DataFrame(holdout_rows)
    if not tab2.empty:
        tab2_path = out_dir / 'tab2_holdout_metrics.csv'
        tab2.to_csv(tab2_path, index=False)
        print(f'✅ Tabla 2 (holdout) -> {tab2_path}')
    else:
        print('ℹ️  No se generó Tabla 2 (no se encontraron métricas de holdout).')

    # Figuras
    if not tab1.empty:
        df_sum = tab1[['Model','Horizon','R2','RMSE']].copy()
        fig_barras_r2_rmse(df_sum, out_dir)
    else:
        print('⚠️  No hay datos de rolling para Fig.1')

    fig_timeseries_all_models(preds_dict, horizons, out_dir, last_points=args.last_points)
    fig_timeseries_zoom(preds_dict, horizons, out_dir, zoom_points=168)
    fig_scatter_combined(preds_dict, horizons, out_dir)
    fig_sens_tol_all(preds_dict, horizons, out_dir)

    print(f'🎯 Listo. Salida en: {out_dir}')

if __name__ == '__main__':
    main()
