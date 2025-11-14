Predicción Rolling Multihorizonte de la Demanda de Energía Eléctrica

Autor: Juan Carlos Castillo Matamoros
Afiliación: Universidad Técnica de Cotopaxi
Tutores: Ing. Jessica N. Castillo · Ing. Gabriel Pezantes
Año: 2025

Resumen

Este proyecto presenta un sistema de predicción rolling multihorizonte aplicado a series temporales de demanda energética, integrando modelos basados en boosting y arquitecturas deep tabulares.
El objetivo es evaluar el desempeño comparativo de distintos enfoques de modelado —desde técnicas tradicionales hasta modelos de aprendizaje profundo— en entornos de validación rolling windows, garantizando una evaluación realista y sin fuga de información.

El código desarrollado permite automatizar la generación de predicciones, el cálculo de métricas de desempeño (R², RMSE, etc.) y la creación de figuras listas para publicación académica.

Arquitectura general del proyecto

El repositorio se compone de tres scripts principales:

make_rolling_preds_resume_full.py
Implementa el pipeline completo de predicción rolling con cuatro modelos principales:

EvoXGB (evolutivo sobre XGBoost)

XGB (XGBoost clásico)

TabNet (Deep Tabular Network)

FTTransformer (Feature Token Transformer)
Este script administra la ejecución rolling, los cortes de ventana, el manejo de memoria y la exportación de resultados para cada estación y horizonte.

make_rolling_preds_tabnet_SOLO.py
Versión especializada que ejecuta únicamente el modelo TabNet, con optimización de hiperparámetros y control de early stopping adaptado a la naturaleza de la serie.

build_results_figs_INGENIUS.py
Genera las figuras y tablas de resultados finales:

Comparativas de desempeño entre modelos y horizontes.

Gráficos de series reales vs. predichas.

Análisis de sensibilidad frente a tolerancias de error.

Tablas de resultados exportables para artículos científicos.

Estructura del repositorio
.
├── data/
│   └── con_lags/                  # Datos por estación (formato CSV)
├── resultados_final_v2/           # Salidas del pipeline completo
├── resultados_tabnet_SOLO/        # Salidas del pipeline TabNet solo
├── figuras_INGENIUS/              # Figuras y tablas finales
├── make_rolling_preds_resume_full.py
├── make_rolling_preds_tabnet_SOLO.py
├── build_results_figs_INGENIUS.py
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md

Instalación y configuración

Requisitos mínimos:

Python 3.10 o superior

GPU compatible con CUDA (opcional, para acelerar entrenamiento)

Sistemas operativos soportados: Windows 10+, Ubuntu 22.04+

Instalación de dependencias:

pip install -r requirements.txt

Ejecución básica

Pipeline completo (cuatro modelos):

python make_rolling_preds_resume_full.py \
  --data_dir data/con_lags \
  --stations "S-E EL CALVARIO_con_lags" \
  --min_train 1500 --max_windows 200


Pipeline TabNet solo:

python make_rolling_preds_tabnet_SOLO.py \
  --data_dir data/con_lags \
  --stations "S-E EL CALVARIO_con_lags" \
  --tabnet_epochs 120 --tabnet_patience 15 --tabnet_batch 512


Generación de figuras y tablas:

python build_results_figs_INGENIUS.py \
  --models "A_EvoXGB=./resultados_final_v2;B_XGB=./resultados_final_v2;C_TabNet=./resultados_tabnet_SOLO;D_FTT=./resultados_final_v2" \
  --station "S-E EL CALVARIO_con_lags" \
  --horizons "corto,mediano" \
  --delta_tol 0.10 \
  --out_dir figuras_INGENIUS

Características destacadas

Validación rolling estricta y reproducible.

Reanudación automática de experimentos interrumpidos.

Escalado robusto y tratamiento de lags dinámicos.

Exportación estructurada de resultados y figuras.

Cumplimiento de normas editoriales (Ingenius, IEEE).

Arquitectura modular adaptable a nuevos modelos.

Licencia

Este proyecto se distribuye bajo la Licencia MIT.
El código y sus resultados pueden ser reutilizados con fines académicos o de investigación, citando al autor original:

Juan Carlos Castillo Matamoros — Universidad Técnica de Cotopaxi (2025)

Referencias

Arik, S. Ö., & Pfister, T. (2021). TabNet: Attentive Interpretable Tabular Learning.

Vaswani, A. et al. (2017). Attention is All You Need.

Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.



Este proyecto simboliza la búsqueda constante de precisión y comprensión en los sistemas energéticos. Aunque es solo un fragmento dentro del vasto campo del aprendizaje automático, refleja la convicción de que cada avance, por pequeño que parezca, impulsa a la ciencia un paso más adelante.
