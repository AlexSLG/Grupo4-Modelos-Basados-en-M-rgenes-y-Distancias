# Equipo 4 · Familia 2: Modelos basados en márgenes y distancias

Actividad práctica *"Despliegue y Análisis Comparativo de Familias de Aprendizaje Supervisado"* — Inteligencia Artificial.

| # | Modelo | Implementación de referencia | Implementación desde cero |
|---|--------|------------------------------|---------------------------|
| 1 | SVM con kernel RBF / gaussiano | `sklearn.svm.SVC(kernel="rbf")` | `SVMDualDesdeCero(kernel="rbf")` |
| 2 | SVM de margen suave (holguras ξᵢ) | `sklearn.svm.SVC(kernel="linear")` | `SVMDualDesdeCero(kernel="lineal")` |
| 3 | Radius Neighbors / k-NN ponderado | `RadiusNeighborsClassifier`, `KNeighborsClassifier(weights="distance")` | `VecinosRadioDesdeCero`, `KNNPonderadoDesdeCero` |

## Estructura

```
IF/
├── GUIA.md                # Guía detallada: cada archivo y cómo funciona cada modelo
├── experimentos.py        # Ejecuta todo y genera resultados/ (tablas + figuras)
├── probar_modelos.py      # Prueba rápida de los 4 modelos con los mejores hiperparámetros
├── app_simulacion.py      # App Streamlit: simulación 2D, fundamentos, casos reales y «Probar un caso»
├── requirements.txt
├── src/
│   ├── config.py          # Semilla, rutas y paleta de colores
│   ├── datos.py           # Casos reales, conjuntos sintéticos 2D y preprocesamiento
│   ├── modelos.py         # Pipelines y espacios de búsqueda de hiperparámetros
│   ├── desde_cero.py      # SVM dual (QP con SLSQP), k-NN ponderado y Radius Neighbors en NumPy
│   ├── graficas.py        # Fronteras 2D/3D, ROC, matrices, curvas de aprendizaje, importancias
│   ├── prediccion.py      # Modelos finales de los casos reales: entrenamiento y predicción
│   └── textos.py          # Textos de ayuda de la app (controles, métricas, gráficas, variables)
└── resultados/            # Se genera al ejecutar experimentos.py
    ├── figuras/           # 26 figuras PNG para el informe
    ├── cancer/, credito/  # Métricas en prueba y resultados de la validación cruzada
    ├── verificacion_desde_cero.csv
    └── resumen.json
```

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
```

```bash
python experimentos.py
```

Tarda ~2 minutos. `--rapido` usa rejillas más pequeñas; `--sin-credito` omite el caso German Credit (se descarga de OpenML la primera vez y se guarda en `datos_cache/`).

```bash
python probar_modelos.py
```

```bash
streamlit run app_simulacion.py
```

La explicación completa de cada archivo y de cada modelo está en [GUIA.md](GUIA.md).

## Metodología

**Preprocesamiento.** `ColumnTransformer` con `StandardScaler` para variables numéricas y `OneHotEncoder` para categóricas, dentro de un `Pipeline`, para que el escalado se ajuste sólo con los pliegues de entrenamiento (sin fuga de información). Escalar es obligatorio en los tres modelos: el kernel RBF y los vecinos dependen de distancias, y el margen de la SVM depende de ‖w‖. División estratificada 75 % / 25 % (`random_state=42`).

**Búsqueda de hiperparámetros** (validación cruzada estratificada de 5 pliegues, selección por ROC-AUC):

| Modelo | Búsqueda | Espacio |
|--------|----------|---------|
| SVM RBF | `GridSearchCV` (72 combinaciones) | C ∈ 10⁻²…10³, γ ∈ 10⁻⁴…10¹, `class_weight` ∈ {None, balanced} |
| SVM Margen Suave | `GridSearchCV` (12) | C ∈ 10⁻³…10², `class_weight` ∈ {None, balanced} |
| k-NN Ponderado | `GridSearchCV` (42) | k ∈ {1, 3, …, 41}, p ∈ {1, 2} |
| Radius Neighbors | `RandomizedSearchCV` (40) | r ~ U(r₅-NN, P₉₀ de distancias), pesos ∈ {uniforme, 1/d} |

Además se ajusta el **umbral de decisión** con `TunedThresholdClassifierCV` (máximo F1 en validación cruzada, sólo con datos de entrenamiento).

**Evaluación.** Exactitud, precisión, recall, F1, ROC-AUC, matriz de confusión, curvas ROC y de aprendizaje, importancia por permutación (agnóstica al modelo) y validación cruzada repetida 5 × 5 (sólo sobre entrenamiento) para comparar con su variabilidad.

## Resultados principales

### Diagnóstico de cáncer de mama (Breast Cancer Wisconsin · 569 muestras · 30 variables · 37 % malignos)

| Modelo | Exactitud | Recall | F1 | ROC-AUC | F1 con umbral ajustado | AUC CV 5×5 |
|--------|-----------|--------|----|---------|------------------------|------------|
| SVM RBF (C=10, γ=0.01) | 0.972 | 0.925 | 0.961 | 0.997 | 0.961 | 0.995 ± 0.006 |
| SVM Margen Suave (C=0.1) | **0.979** | 0.943 | **0.971** | 0.997 | 0.971 | 0.994 ± 0.006 |
| k-NN Ponderado (k=37, p=2) | 0.958 | 0.887 | 0.940 | 0.997 | 0.962 | 0.991 ± 0.005 |
| Radius Neighbors (r=10.8) | 0.720 | 0.245 | 0.394 | 0.996 | 0.953 | 0.983 ± 0.016 |

### Riesgo crediticio (German Credit · 1000 muestras · 7 numéricas + 13 categóricas · 30 % malos pagadores)

| Modelo | Exactitud | Recall | F1 | ROC-AUC | F1 con umbral ajustado | AUC CV 5×5 |
|--------|-----------|--------|----|---------|------------------------|------------|
| SVM RBF (C=1, γ=0.01, balanced) | 0.720 | 0.827 | 0.639 | **0.820** | 0.645 | 0.778 ± 0.044 |
| SVM Margen Suave (C=0.01, balanced) | 0.708 | 0.813 | 0.626 | 0.815 | 0.635 | 0.774 ± 0.045 |
| k-NN Ponderado (k=39, p=1) | 0.736 | 0.187 | 0.298 | 0.783 | 0.606 | 0.754 ± 0.048 |
| Radius Neighbors (r=4.9) | 0.716 | 0.080 | 0.145 | 0.778 | 0.637 | 0.737 ± 0.052 |

### Hallazgos para el análisis

1. **Lineal vs. RBF.** En cáncer el problema es casi linealmente separable: la SVM de margen suave iguala a la RBF con un modelo más simple e interpretable (γ óptimo pequeño = frontera casi lineal).
2. **Umbral vs. ranking.** Los métodos de vecinos ordenan bien los casos (AUC ≈ 0.99) pero, con el umbral por defecto de 0.5, en datos desbalanceados la clase mayoritaria domina los vecindarios grandes y el recall se desploma (Radius Neighbors en cáncer: 0.25). Ajustar el umbral lo corrige (0.25 → 0.96). Las SVM tienen `class_weight`; los vecinos no.
3. **Maldición de la dimensionalidad.** En 30 dimensiones el radio óptimo (≈ 10.8) es comparable a la distancia típica entre puntos: la "bola" deja de ser local, y Radius Neighbors pierde su ventaja frente a k-NN.
4. **Variables categóricas.** Con one-hot, las distancias mezclan escalas numéricas y binarias; las SVM lo toleran mejor que los vecinos (AUC 0.82 vs. 0.78 en crédito).
5. **Coste.** La SVM RBF es la más cara de ajustar (búsqueda de 12 s en crédito, 575 vectores de soporte); los vecinos no entrenan pero cada predicción recorre todo el conjunto de entrenamiento.

### Verificación de las implementaciones desde cero

| Modelo | Comprobación | Desde cero | scikit-learn |
|--------|--------------|------------|--------------|
| SVM RBF | Vectores de soporte / sesgo b | 76 / 0.0543 | 76 / 0.0543 |
| SVM RBF | Brecha de dualidad (primal − dual) | 2.7 × 10⁻⁶ | — |
| SVM Margen Suave | Ancho del margen 2/‖w‖ · Σξᵢ | 1.1433 · 23.691 | 1.1434 · 23.692 |
| SVM Margen Suave | Brecha de dualidad | 7.0 × 10⁻¹⁰ | — |
| k-NN Ponderado / Radius Neighbors | Acuerdo de predicciones | 100 % / 100 % | — |

La brecha de dualidad ≈ 0 confirma numéricamente la dualidad fuerte del problema cuadrático convexo.
