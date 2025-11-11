📌 Proyecto

Predicción rolling de potencia con ensambles y deep tabular + generación de figuras editoriales

Este repositorio contiene tres scripts principales:

make_rolling_preds_resume_full.py
Pipeline “todo en uno” para backtesting rolling con 4 modelos (A_EvoXGB, B_XGB, C_TabNet, D_FTT), con soporte GPU cuando esté disponible, reanudación por estación×horizonte y guardado de métricas y predicciones. Actualmente están definidos dos horizontes (corto, mediano).

make_rolling_preds_tabnet_SOLO.py
Variante ligerísima que corre solo TabNet (misma lógica rolling, sin fugas, escalado robusto por ventana para X e Y, reanudación, y salidas análogas). Ejecuta los horizontes corto y mediano.

build_results_figs_INGENIUS.py
Genera tablas y figuras “listas para artículo” comparando los 4 modelos en una estación representativa y dos horizontes (barras de R²/RMSE, series real vs predicción, scatter combinado, sensibilidad de %Tol).

🗂️ Estructura sugerida del repo
.
├── data/
│   └── con_lags/                  # CSVs por estación (campos requeridos: datetime, potencia)
├── resultados_final_v2/           # Salidas del pipeline full (se crean al ejecutar)
├── resultados_tabnet_SOLO/        # Salidas del pipeline TabNet SOLO (se crean al ejecutar)
├── figuras_INGENIUS/              # Tablas/figuras finales (se crean al ejecutar)
├── make_rolling_preds_resume_full.py
├── make_rolling_preds_tabnet_SOLO.py
├── build_results_figs_INGENIUS.py
├── requirements.txt
└── README.md


Formato esperado de los datos

Archivos .csv por estación dentro de data/con_lags/ con columnas: datetime (parseable a fecha-hora) y potencia. Los scripts validan estas columnas y generan features de calendario, lags y rollings antes del split 90/10 para holdout.

🧰 Requisitos

Recomendado: Python 3.10+

GPU opcional para acelerar XGBoost/TabNet/FTTransformer (CUDA si está disponible).

Instala dependencias:

pip install -r requirements.txt


Nota: torch y pytorch-tabnet son opcionales si solo usarás los modelos de árbol. Si vas a correr TabNet o FTT, asegúrate de que torch esté instalado para tu plataforma/CUDA.

🚀 Uso
1) Pipeline FULL (4 modelos)

Ejemplo mínimo:

python make_rolling_preds_resume_full.py \
  --data_dir data/con_lags \
  --stations "S-E EL CALVARIO_con_lags" \
  --min_train 1500 --max_windows 200


Modelos: A_EvoXGB, B_XGB, C_TabNet, D_FTT.

Horizontes configurados: corto (WINDOW=168, STEP=24, PURGE=24) y mediano (WINDOW=720, STEP=72, PURGE=72).

Split: 90% rolling / 10% holdout, con métricas para ambos.

Reanudación y guardado incremental de predicciones/métricas por estación y horizonte.

Salidas clave (por horizonte y estación):

resultados_final_v2/<horizonte>/<estación>/predictions/preds_<modelo>.csv

resultados_final_v2/<horizonte>/metrics_rolling.csv y metrics_holdout.csv

resultados_final_v2/<horizonte>/<estación>/models/… (modelos y scalers).

Flags útiles (selección):
--xgb_n_estim, --evoxgb_steps, --tabnet_epochs, --tabnet_patience, --tabnet_batch, --ftt_* (dimensiones, capas, heads, dropout).

2) Pipeline TabNet SOLO

Ejemplo:

python make_rolling_preds_tabnet_SOLO.py \
  --data_dir data/con_lags \
  --stations "S-E EL CALVARIO_con_lags" \
  --min_train 1500 --max_windows 200 \
  --tabnet_epochs 120 --tabnet_patience 15 --tabnet_batch 512


Sin fugas (rollings con shift(1) y recorte por lags); escalado robusto por ventana en X e Y con inversión al predecir; progreso y logs por ventana.

Resultados bajo resultados_tabnet_SOLO/<horizonte>/<estación>/… (análogo al full).

3) Generación de Tablas y Figuras (estilo editorial)

Ejemplo típico:

python build_results_figs_INGENIUS.py \
  --models "A_EvoXGB=./resultados_final_v2;B_XGB=./resultados_final_v2;C_TabNet=./resultados_tabnet_SOLO;D_FTT=./resultados_final_v2" \
  --station "S-E EL CALVARIO_con_lags" \
  --horizons "corto,mediano" \
  --delta_tol 0.10 \
  --last_points 672 \
  --out_dir figuras_INGENIUS


Salidas en figuras_INGENIUS/:

tab1_rolling_metrics.csv (rolling por modelo×horizonte; incluye %Tol@δ)

tab2_holdout_metrics.csv (si existe)

fig1a_barras_R2.png/.pdf, fig1b_barras_RMSE.png/.pdf

fig2_timeseries_<h>.png/.pdf, fig2b_timeseries_zoom_<h>.png/.pdf

fig3_scatter_combined.png/.pdf

fig4_sens_tol.png/.pdf

🧪 Consejos y solución de problemas

CUDA/GPU: si no hay GPU, los modelos se entrenan en CPU (más lento). El script lo detecta y sigue.

Datos con NaN/inf: se aplican ffill, sanitización y robust scaling; TabNet y FTT incluyen defensas adicionales. Si un modelo devuelve None, la ventana se salta y queda logueado.

Reanudar trabajos: se escriben archivos de progreso por modelo/horizonte/estación; si existen predicciones previas, el entrenamiento se omite para no recalcular.

📄 Licencia

Elige la que prefieras (MIT recomendado). Ejemplo:

MIT License — Copyright (c) 2025

✍️ Cita y normas editoriales

Las figuras/tablas generadas están pensadas para cumplir buenas prácticas editoriales (etiquetado consistente, exportación PNG/PDF). Para normas tipo IEEE/Ingenius, consulta la guía de la revista para secciones, tablas/figuras y referencias (si aplica en tu artículo).

🙌 Contribuir

Issues y PRs bienvenidos. Por favor incluye: versión de Python, SO, y log comprimido si es un bug de entrenamiento.