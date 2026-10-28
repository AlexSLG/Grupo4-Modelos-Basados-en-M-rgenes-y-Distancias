# Guía del proyecto · Equipo 4

**Familia 2 — Modelos basados en márgenes y distancias**

1. SVM con kernel RBF / gaussiano
2. SVM de margen suave (con variables de holgura $\xi_i$)
3. Radius Neighbors / $k$-NN ponderado

Esta guía tiene tres partes: **(A)** qué hay en cada carpeta y archivo y cómo se conectan, **(B)** cómo ejecutar y probar todo, y **(C)** cómo funciona cada modelo, en la teoría y en nuestro código.

---

## Parte A · Carpetas y archivos

### Mapa general

```
IF/
├── README.md               ← Presentación del proyecto y resultados (lo primero que ve el profesor en GitHub)
├── GUIA.md                 ← Este archivo
├── requirements.txt        ← Librerías necesarias
├── .gitignore              ← Qué NO se sube a GitHub
├── .streamlit/
│   └── config.toml         ← Tema visual (claro) de la simulación
│
├── experimentos.py         ← PROGRAMA PRINCIPAL: entrena, evalúa y genera todas las tablas y figuras
├── probar_modelos.py       ← Prueba rápida: entrena los 4 modelos con los mejores hiperparámetros
├── app_simulacion.py       ← Simulación interactiva (Streamlit)
│
├── src/                    ← Código reutilizable (lo importan los tres programas de arriba)
│   ├── __init__.py
│   ├── config.py
│   ├── datos.py
│   ├── modelos.py
│   ├── desde_cero.py
│   ├── graficas.py
│   ├── prediccion.py
│   └── textos.py
│
├── resultados/             ← GENERADO por experimentos.py (tablas y figuras)
│   ├── figuras/            ← 26 imágenes PNG para el informe
│   ├── cancer/             ← Caso real 1
│   ├── credito/            ← Caso real 2
│   └── *.csv, resumen.json
│
└── datos_cache/            ← GENERADO: copia local del dataset German Credit (no se sube a GitHub)
```

### Cómo se conectan

```
                    src/config.py   (semilla, rutas, colores)
                          │
     src/datos.py ────────┼──────── src/modelos.py ─────── src/graficas.py
   (carga y prepara       │       (pipelines + búsqueda      (todas las
    los datos)            │        de hiperparámetros)        gráficas)
                          │
                 src/desde_cero.py  (los 3 modelos programados con NumPy)
                          │
        ┌─────────────────┼──────────────────┐
        ▼                 ▼                  ▼
  experimentos.py   probar_modelos.py   app_simulacion.py
        │                 ▲                  ▲
        └──► resultados/ ─┴──────────────────┘
             (probar_modelos y la pestaña "Casos reales" de la app leen estos resultados)
```

### Archivo por archivo

#### Programas que se ejecutan

| Archivo | Qué hace | Cuándo usarlo |
|---------|----------|---------------|
| `experimentos.py` | Ejecuta **todo** el estudio: (1) los experimentos 2D de cada modelo, (2) la verificación de nuestras implementaciones desde cero frente a scikit-learn y (3) los dos casos reales con búsqueda de hiperparámetros, métricas y gráficas. Guarda todo en `resultados/`. | Para regenerar los resultados (~2 min). |
| `probar_modelos.py` | Lee los mejores hiperparámetros de `resultados/<caso>/metricas_prueba.csv`, entrena los 4 modelos, imprime el reporte de clasificación y predice casos individuales. | Para comprobar rápido que los modelos funcionan (~10 s). |
| `app_simulacion.py` | Página web interactiva con 4 pestañas: **Simulación** (abres «Parámetros de la simulación», mueves C, γ, k o r y ves cambiar la frontera en vivo; también puedes clasificar un punto nuevo), **Fundamentos** (ecuaciones), **Casos reales** (tablas y figuras de `resultados/`) y **Probar un caso** (ingresas un paciente o un solicitante de crédito y ves qué predice cada modelo). Cada control, métrica y gráfica tiene su explicación. | Para la presentación y para entender los modelos jugando con ellos. |

#### Módulos de `src/`

| Archivo | Contenido | Funciones clave |
|---------|-----------|-----------------|
| `config.py` | Constantes globales: semilla `42`, proporción de prueba `25 %`, `5` pliegues de validación cruzada, rutas de carpetas y paleta de colores (apta para daltonismo). | — |
| `datos.py` | Carga de datos y preprocesamiento. | `cargar_cancer()`, `cargar_credito()`, `dividir()` (división estratificada 75/25), `construir_preprocesador()` (escalado + one-hot), y los conjuntos 2D `lunas()`, `circulos()`, `nubes_solapadas()`, `densidad_variable()` |
| `modelos.py` | Los 4 *pipelines* (preprocesador → clasificador) y sus espacios de búsqueda de hiperparámetros. | `construir_busquedas()`, `rango_radios()`, `puntajes()` |
| `desde_cero.py` | Los 3 modelos programados "a mano" con NumPy/SciPy a partir de sus ecuaciones. Sirven para **demostrar** que entendemos la matemática: dan los mismos resultados que scikit-learn. | `SVMDualDesdeCero`, `KNNPonderadoDesdeCero`, `VecinosRadioDesdeCero`, `kernel_rbf()` |
| `graficas.py` | Todas las funciones de dibujo, compartidas entre `experimentos.py` y la app. | `frontera_2d()`, `superficie_3d()`, `curvas_roc()`, `matrices_confusion()`, `curva_aprendizaje()`, `barras_importancia()`, `mapa_calor()`, `confianza_modelos()` |
| `prediccion.py` | Entrena los 4 modelos finales de un caso real con los mejores hiperparámetros guardados, valida datos de entrada y predice. Lo usan `probar_modelos.py` y la pestaña **Probar un caso**. | `entrenar_mejores()`, `preparar_entrada()`, `predecir()` |
| `textos.py` | Todos los textos de ayuda de la app (controles, métricas, leyendas de gráficas, columnas, variables de cada caso). | `AYUDA_CONTROLES`, `AYUDA_METRICAS`, `LEYENDAS_CASOS`, `variables_cancer()`, `VARIABLES_CREDITO` |

#### Resultados generados (`resultados/`)

| Archivo | Contenido |
|---------|-----------|
| `verificacion_desde_cero.csv` | Comparación de nuestras implementaciones con scikit-learn (vectores de soporte, sesgo b, Σξ, brecha de dualidad, acuerdo de predicciones). |
| `rbf_barrido_lunas.csv` | Exactitud de entrenamiento/prueba y vectores de soporte para 54 combinaciones de γ y C. |
| `margen_suave_barrido_C.csv` | Para 25 valores de C: ancho del margen, vectores de soporte, Σξ, puntos dentro del margen, mal clasificados y exactitud. |
| `vecinos_barrido_k.csv` / `vecinos_barrido_radio.csv` | Exactitud según k (uniforme vs. ponderado) y según r (con % de puntos sin vecinos). |
| `cancer/metricas_prueba.csv`, `credito/metricas_prueba.csv` | **Tabla principal**: exactitud, precisión, recall, F1, ROC-AUC, AUC de validación, umbral ajustado, tiempos, vectores de soporte y mejores hiperparámetros. |
| `cancer/cv_*.csv`, `credito/cv_*.csv` | Resultado completo de cada búsqueda de hiperparámetros (todas las combinaciones probadas). |
| `cancer/cv_repetida_auc.csv`, `credito/cv_repetida_auc.csv` | Los 25 valores de AUC de la validación cruzada repetida 5×5. |
| `resumen.json` | Todo lo anterior en un solo archivo. |

#### Figuras (`resultados/figuras/`)

| Figura | Qué muestra | Modelo |
|--------|-------------|--------|
| `rbf_efecto_gamma_C.png` | Rejilla 3×3: cómo cambian la frontera y los vectores de soporte con γ y C. | RBF |
| `rbf_frontera_2d_3d.png` | Frontera en 2D y la función $f(x)$ como superficie 3D. | RBF |
| `rbf_truco_kernel.png` | Círculos concéntricos: inseparables en 2D, separables por un plano tras añadir una coordenada gaussiana. | RBF |
| `margen_suave_efecto_C.png` | Margen ancho (C pequeño) vs. estrecho (C grande), con 2/‖w‖, VS y Σξ. | Margen suave |
| `margen_suave_barrido_C.png` | Ancho del margen, nº de VS y exactitud en función de C. | Margen suave |
| `margen_suave_holguras.png` | Cada $\xi_i > 0$ ordenado: dentro del margen (≤ 1) vs. mal clasificado (> 1). | Margen suave |
| `vecinos_knn_vs_radio.png` | k = 1, 15 uniforme y 15 ponderado vs. radios 0.3, 0.7 y 1.5 (rayado = sin vecinos). | Vecinos |
| `vecinos_barridos.png` | Exactitud vs. k, exactitud vs. r y % sin vecinos vs. r. | Vecinos |
| `<caso>_roc.png` | Curvas ROC de los 4 modelos. | Comparativa |
| `<caso>_matrices_confusion.png` | Matrices de confusión. | Comparativa |
| `<caso>_comparativa_metricas.png` | Barras de exactitud, precisión, recall, F1 y AUC. | Comparativa |
| `<caso>_ajuste_umbral.png` | Recall y F1 antes y después de ajustar el umbral de decisión. | Comparativa |
| `<caso>_curvas_aprendizaje.png` | AUC de entrenamiento y validación según el tamaño de la muestra. | Comparativa |
| `<caso>_cv_repetida.png` | Los 25 AUC de la validación cruzada repetida por modelo. | Comparativa |
| `<caso>_importancia_permutacion.png` | Las 10 variables más importantes por modelo. | Comparativa |
| `<caso>_sensibilidad_hiperparametros.png` | AUC de validación vs. C, k y r. | Comparativa |
| `<caso>_rbf_mapa_calor.png` | AUC de validación para cada combinación C × γ. | RBF |

---

## Parte B · Cómo ejecutar y probar

### 1. Instalar (una sola vez)

```bash
pip install -r requirements.txt
```

### 2. Generar todos los resultados

```bash
python experimentos.py
```

Mensajes esperados (≈ 2 minutos):

```
[1/3] SVM con kernel RBF
[2/3] SVM de margen suave
[3/3] Radius Neighbors / k-NN ponderado
=== Caso real: cancer (569 muestras, 30 variables, 37% clase positiva) ===
  - SVM RBF ... AUC prueba = 0.997 | F1 = 0.961 | ...
...
Listo en ~100 s.
```

Opciones: `--rapido` (rejillas pequeñas, ~1 min) y `--sin-credito` (sin internet la primera vez).

### 3. Probar los modelos

```bash
python probar_modelos.py
```

```bash
python probar_modelos.py --caso credito --casos 10
```

Muestra el reporte de clasificación de cada modelo y una tabla como esta:

```
        real          SVM RBF  SVM Margen Suave   k-NN Ponderado  Radius Neighbors
12   Maligno  Maligno (+1.16)   Maligno (+3.31)  Maligno (+0.89)   Maligno (+0.87)
321  Maligno  Maligno (+3.15)   Maligno (+3.19)  Maligno (+0.93)   Benigno (+0.42)   ← falla por el umbral 0.5
```

Entre paréntesis: el valor de $f(x)$ en las SVM (positivo = clase 1) o la probabilidad de la clase 1 en los vecinos.

### 4. Abrir la simulación interactiva

```bash
streamlit run app_simulacion.py
```

Se abre en `http://localhost:8501`. Cosas para probar en la presentación:

- **SVM RBF + Lunas:** sube γ hasta 100 y la frontera se "arruga" alrededor de cada punto (sobreajuste); bájalo a 0.01 y se vuelve casi una recta.
- **SVM Margen Suave + Círculos:** la exactitud cae a ≈ 0.5 porque ninguna recta separa círculos. Cambia a **SVM RBF** y sube a ≈ 1.0. Es el argumento a favor del kernel.
- **SVM Margen Suave + Nubes solapadas:** mueve C y observa cómo cambian el ancho del margen, Σξ y el número de vectores de soporte.
- **Interruptor "implementación desde cero":** nuestra SVM resuelve el dual con SciPy; la métrica *Brecha de dualidad* (≈ 10⁻⁶) demuestra que llegó al óptimo.
- **Radius Neighbors + Densidad variable:** con r = 0.3 aparece mucho rayado (puntos sin vecinos); con r grande desaparece.
- **Probar un punto nuevo:** escribe x₁ y x₂ debajo de las métricas; aparece una estrella en la gráfica y el modelo dice su clase (en Radius, cuántos vecinos tiene dentro del radio).
- **Pestaña Probar un caso:** elige *Diagnóstico de cáncer de mama*, el caso **#321** del conjunto de prueba y pulsa **Predecir**. Radius Neighbors dice *Benigno* con el umbral por defecto (P = 0.42 < 0.5), pero *Maligno* con el umbral ajustado (0.32). Es el ejemplo perfecto de por qué importa el umbral. También puedes llenar el formulario a mano o descargar la plantilla CSV, añadir filas y subirla.

### 5. Verificar la matemática

Abre `resultados/verificacion_desde_cero.csv`. Las columnas `propio` y `sklearn` deben coincidir y la brecha de dualidad debe ser ≈ 0.

---

## Parte C · Cómo funciona cada modelo

### Ideas comunes a la familia

Los tres modelos clasifican según la **geometría** del espacio de características:

- Las **SVM** buscan la frontera que deja la mayor distancia (*margen*) entre las clases.
- Los **vecinos** clasifican un punto según las etiquetas de los puntos de entrenamiento cercanos.

Por eso los tres **necesitan escalar los datos**. Si una variable va de 0 a 1000 y otra de 0 a 1, la primera domina todas las distancias. En el código, el `StandardScaler` está dentro del `Pipeline` (`src/datos.py → construir_preprocesador`), de modo que se ajusta solo con los datos de entrenamiento de cada pliegue y no hay fuga de información.

Notación: $\mathbf{x}_i$ es un punto de entrenamiento, $y_i \in \{-1, +1\}$ su clase y $n$ el número de puntos.

---

### Modelo 1 · SVM con kernel RBF (gaussiano)

#### Intuición
Una SVM lineal solo puede trazar fronteras rectas. El kernel RBF lleva los datos a un espacio de dimensión infinita donde sí existe un hiperplano separador. Ese hiperplano, visto en el espacio original, es una frontera curva. Lo mejor es que nunca hace falta calcular ese espacio: basta con poder medir la *similitud* entre dos puntos (el **truco del kernel**).

#### Formulación (hipótesis)

Kernel gaussiano: la similitud vale 1 si los puntos coinciden y tiende a 0 cuando se alejan:

$$K(\mathbf{x}, \mathbf{z}) = \exp\left(-\gamma \, \lVert \mathbf{x} - \mathbf{z} \rVert^2\right)$$

Función de decisión: una suma de "campanas gaussianas" centradas en los vectores de soporte:

$$f(\mathbf{x}) = \sum_{i \in SV} \alpha_i \, y_i \, K(\mathbf{x}_i, \mathbf{x}) + b, \qquad h(\mathbf{x}) = \operatorname{sign}\big(f(\mathbf{x})\big)$$

#### Criterio de optimización
Es el mismo problema dual de la SVM de margen suave (modelo 2), pero con $K$ en lugar del producto escalar:

$$\max_{\boldsymbol\alpha} \; \sum_i \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j \, y_i y_j \, K(\mathbf{x}_i, \mathbf{x}_j) \quad \text{s.a.} \quad 0 \le \alpha_i \le C, \;\; \sum_i \alpha_i y_i = 0$$

#### Algoritmo de aprendizaje
Es **programación cuadrática convexa**: tiene un único óptimo global, a diferencia de una red neuronal.
- scikit-learn usa **SMO** (*Sequential Minimal Optimization*, de libsvm): optimiza dos $\alpha$ a la vez de forma analítica.
- Nuestra versión (`SVMDualDesdeCero`) construye la matriz $Q_{ij} = y_i y_j K(\mathbf{x}_i, \mathbf{x}_j)$ y resuelve el dual con SLSQP de SciPy. Después calcula $b$ con los vectores de soporte que cumplen $0 < \alpha_i < C$ (condiciones KKT).

#### Hiperparámetros
| Hiperparámetro | Efecto | Valor alto | Valor bajo |
|----------------|--------|------------|------------|
| **γ** (gamma) | Alcance de cada vector de soporte: $1/\sqrt{\gamma}$ es aproximadamente el "radio" de su campana | Campanas estrechas → frontera muy curva → **sobreajuste** | Campanas anchas → frontera casi lineal → **subajuste** |
| **C** | Castigo a los errores de entrenamiento (ver modelo 2) | Pocos errores, frontera compleja | Margen ancho, más errores tolerados |

Ver `rbf_efecto_gamma_C.png`: con γ = 10 y C = 100 la frontera rodea puntos aislados.

#### En nuestro código
- Pipeline: `SVC(kernel="rbf")` en `src/modelos.py`, rejilla de 72 combinaciones (C, γ, `class_weight`).
- Desde cero: `SVMDualDesdeCero(kernel="rbf")` en `src/desde_cero.py`.
- Experimento 2D: `experimento_rbf()` en `experimentos.py`.

#### Resultados
- **Cáncer:** C = 10, γ = 0.01 → AUC 0.997, F1 0.961. Un γ tan pequeño indica que la frontera óptima es **casi lineal**.
- **Crédito:** C = 1, γ = 0.01, `balanced` → AUC **0.820**, el mejor de los 4 modelos.

#### Ventajas y limitaciones
- ✅ Captura fronteras no lineales sin diseñar variables a mano. Óptimo global y garantías teóricas (maximiza el margen).
- ✅ El modelo solo depende de los vectores de soporte.
- ❌ El entrenamiento cuesta entre $O(n^2)$ y $O(n^3)$: es caro con cientos de miles de datos.
- ❌ Hay que ajustar dos hiperparámetros a la vez (C y γ). No da probabilidades ni importancias de forma nativa (usamos importancia por permutación).

#### Caso de uso: diagnóstico médico (cáncer de mama)
Pocos pacientes (cientos), muchas variables numéricas continuas y la necesidad de una frontera flexible pero bien regularizada. Su sensibilidad a valores atípicos es moderada: el parámetro C limita la influencia de cada punto.

---

### Modelo 2 · SVM de margen suave (con variables de holgura $\xi_i$)

#### Intuición
La **SVM de margen duro** exige que *todos* los puntos queden fuera del margen. Si las clases se solapan (casi siempre en datos reales), ese problema no tiene solución. El **margen suave** permite que algunos puntos violen el margen, pero cobra una penalización $\xi_i$ por cada violación. El parámetro $C$ decide cuánto pesa esa penalización frente a tener un margen ancho.

#### Formulación (hipótesis)

$$f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + b, \qquad h(\mathbf{x}) = \operatorname{sign}\big(f(\mathbf{x})\big)$$

El margen es la franja entre $f(\mathbf{x}) = +1$ y $f(\mathbf{x}) = -1$, y su ancho es $\dfrac{2}{\lVert \mathbf{w} \rVert}$.

#### Criterio de optimización (problema primal)

$$\min_{\mathbf{w}, b, \boldsymbol\xi} \;\; \underbrace{\frac{1}{2}\lVert \mathbf{w} \rVert^2}_{\text{margen ancho}} + \underbrace{C \sum_{i=1}^{n} \xi_i}_{\text{castigo a violaciones}} \qquad \text{s.a.} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$

**Qué significa cada $\xi_i$:** en el óptimo, $\xi_i = \max\big(0,\; 1 - y_i f(\mathbf{x}_i)\big)$.

| Valor | Situación del punto |
|-------|---------------------|
| $\xi_i = 0$ | Bien clasificado y fuera del margen |
| $0 < \xi_i \le 1$ | Bien clasificado pero **dentro** del margen |
| $\xi_i > 1$ | **Mal clasificado** (al otro lado de la frontera) |

Por eso $\sum_i \xi_i$ es una cota superior del número de errores de entrenamiento.

**Equivalencia con la pérdida bisagra (*hinge loss*):** al sustituir $\xi_i$, el problema se convierte en un problema sin restricciones:

$$\min_{\mathbf{w}, b} \; \frac{1}{2}\lVert \mathbf{w} \rVert^2 + C \sum_i \max\big(0,\; 1 - y_i(\mathbf{w}^\top \mathbf{x}_i + b)\big)$$

Es decir: **pérdida bisagra + regularización L2**. Así se conecta la SVM con el marco general de "función de pérdida".

#### Problema dual (el que se resuelve en la práctica)
Con multiplicadores de Lagrange $\alpha_i$:

$$\max_{\boldsymbol\alpha} \; \sum_i \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j \, y_i y_j \, \mathbf{x}_i^\top \mathbf{x}_j \quad \text{s.a.} \quad 0 \le \alpha_i \le C, \;\; \sum_i \alpha_i y_i = 0$$

y entonces $\mathbf{w} = \sum_i \alpha_i y_i \mathbf{x}_i$. Las condiciones KKT clasifican cada punto:

| $\alpha_i$ | Tipo de punto |
|-----------|---------------|
| $\alpha_i = 0$ | No es vector de soporte: no influye en el modelo |
| $0 < \alpha_i < C$ | Vector de soporte **sobre** el margen ($\xi_i = 0$); con estos se calcula $b$ |
| $\alpha_i = C$ | Vector de soporte que **viola** el margen ($\xi_i > 0$) |

El límite $\alpha_i \le C$ es lo que diferencia el dual del margen suave del de margen duro, donde $\alpha_i$ no tiene tope.

**Dualidad fuerte:** como el problema es convexo, el valor óptimo del primal coincide con el del dual. Nuestro código lo comprueba: la brecha es ≈ 7 × 10⁻¹⁰.

#### Algoritmo de aprendizaje
Programación cuadrática (SMO en libsvm; SLSQP en nuestra versión). Al ser convexa, no depende de la inicialización ni tiene mínimos locales.

#### Efecto de C (ver `margen_suave_efecto_C.png`)
| C | Margen | Vectores de soporte | Σξ | Riesgo |
|---|--------|---------------------|----|--------|
| 0.01 | Ancho (2/‖w‖ = 3.21) | 82 | 43.0 | Subajuste |
| 1 | Medio (1.14) | 28 | 23.7 | Equilibrio |
| 100 | Estrecho (1.01) | 24 | 23.5 | Sobreajuste a puntos ruidosos |

$C \to \infty$ recupera la SVM de margen duro.

#### En nuestro código
- Pipeline: `SVC(kernel="linear")` en `src/modelos.py`, con C de 10⁻³ a 10² y `class_weight`.
- Desde cero: `SVMDualDesdeCero(kernel="lineal")`, con los métodos `holguras()`, `ancho_margen()`, `objetivo_primal()` y `objetivo_dual()`.
- Experimento 2D: `experimento_margen_suave()` en `experimentos.py`.

#### Resultados
- **Cáncer:** C = 0.1 → la **mejor exactitud (0.979) y el mejor F1 (0.971)**, empatada en AUC con la RBF. Un modelo lineal basta.
- **Crédito:** C = 0.01, `balanced` → AUC 0.815, muy cerca de la RBF y con un modelo más simple.

#### Ventajas y limitaciones
- ✅ **Interpretable:** $\mathbf{w}$ indica el peso de cada variable. Es robusta al ruido de etiquetas (C limita la influencia de cada punto) y converge al óptimo global.
- ✅ Funciona muy bien en alta dimensión (texto, genómica), donde los datos suelen ser casi linealmente separables.
- ❌ Solo traza fronteras lineales (en círculos: exactitud 0.5).
- ❌ Hay que escalar los datos, y no da probabilidades calibradas de forma nativa.

#### Caso de uso: riesgo crediticio / detección de fraude
En crédito, las clases se **solapan** de forma natural: hay clientes con perfiles parecidos y desenlaces distintos. El margen suave tolera ese solapamiento sin sobreajustar. Además, `class_weight="balanced"` compensa el desbalance (30 % de malos pagadores), y los pesos $\mathbf{w}$ permiten **explicar** la decisión, algo que en banca exige la regulación.

---

### Modelo 3 · Radius Neighbors / $k$-NN ponderado

#### Intuición
"Dime con quién andas y te diré quién eres". No hay entrenamiento propiamente dicho (**aprendizaje perezoso**): se guardan los datos y, al predecir, se mira a los vecinos del punto nuevo. Las dos variantes difieren en **quiénes votan** y **cuánto pesa cada voto**:

| | ¿Quiénes votan? | Tamaño del vecindario |
|-|-----------------|-----------------------|
| **$k$-NN** | Los $k$ puntos más cercanos | Fijo en **número**, variable en distancia |
| **Radius Neighbors** | Todos los puntos a distancia $\le r$ | Fijo en **distancia**, variable en número |

#### Formulación (hipótesis)

$$\hat{y}(\mathbf{x}) = \arg\max_{c} \sum_{i \in \mathcal{N}(\mathbf{x})} w_i \, \mathbb{1}[y_i = c]$$

con el vecindario

$$\mathcal{N}_k(\mathbf{x}) = \{\text{los } k \text{ más cercanos}\} \qquad \text{o} \qquad \mathcal{N}_r(\mathbf{x}) = \{\, i : d(\mathbf{x}, \mathbf{x}_i) \le r \,\}$$

y los pesos

$$w_i = 1 \;\;(\text{voto uniforme}) \qquad \text{o} \qquad w_i = \frac{1}{d(\mathbf{x}, \mathbf{x}_i)} \;\;(\text{ponderado})$$

Distancia de Minkowski: $d(\mathbf{x}, \mathbf{z}) = \left(\sum_j |x_j - z_j|^p\right)^{1/p}$, con $p = 1$ (Manhattan) o $p = 2$ (Euclídea).

La "probabilidad" de cada clase es la fracción ponderada de votos: $P(c \mid \mathbf{x}) = \dfrac{\sum_{i \in \mathcal{N}} w_i \mathbb{1}[y_i = c]}{\sum_{i \in \mathcal{N}} w_i}$.

**Casos especiales que el código maneja:**
- Si un vecino está a distancia 0, solo votan los vecinos idénticos (evita dividir entre 0).
- En Radius Neighbors, si la bola está **vacía**, el punto es atípico y recibe `outlier_label` (aquí, la clase más frecuente). Esto sirve para **detectar** puntos lejanos a todo lo conocido.

#### Criterio de optimización
**No hay función de pérdida que se minimice durante el entrenamiento.** El modelo es no paramétrico: los propios datos son el modelo. Lo que sí se optimiza son los **hiperparámetros** ($k$ o $r$, la ponderación y $p$), mediante validación cruzada maximizando el ROC-AUC. Interpretación teórica: con $k \to \infty$ y $k/n \to 0$, el error de $k$-NN tiende al error de Bayes (el mínimo posible).

#### Algoritmo
- **Entrenamiento:** guardar los datos (o construir un árbol KD/Ball para buscar más rápido).
- **Predicción:** calcular distancias, seleccionar vecinos y votar. Cuesta $O(n \cdot d)$ por cada predicción.

#### Hiperparámetros
| Hiperparámetro | Pequeño | Grande |
|----------------|---------|--------|
| **k** | k = 1: frontera irregular, memoriza el ruido (sobreajuste) | Frontera suave, pero domina la clase mayoritaria |
| **r** | Muchos puntos sin vecinos (rayado en las figuras) | Vecindarios enormes: el voto se vuelve global |
| **Ponderación 1/d** | — | Reduce el efecto de los vecinos lejanos; mejora sobre todo con k grande |

#### Maldición de la dimensionalidad (hallazgo clave)
En 2D, un radio de 0.7 basta. En el caso cáncer (30 dimensiones escaladas), el radio óptimo fue **r ≈ 10.8**, del orden de la distancia típica entre *cualquier* par de puntos. En alta dimensión todas las distancias se parecen, así que la "bola" deja de ser local. Por eso en `src/modelos.py → rango_radios()` el rango de búsqueda se calcula a partir de los datos y no se fija a mano.

#### El problema del umbral (hallazgo clave)
Con vecindarios grandes y clases desbalanceadas, la clase mayoritaria domina casi todas las bolas:
- Radius Neighbors en cáncer: AUC 0.996 (ordena casi perfecto), pero con el umbral 0.5 su **recall es 0.245**: solo detecta 1 de cada 4 tumores malignos.
- Al ajustar el umbral por validación cruzada (`TunedThresholdClassifierCV`), el recall sube a **0.962**.
- A diferencia de las SVM, los vecinos **no tienen `class_weight`**, así que ajustar el umbral es su forma de compensar el desbalance.

#### En nuestro código
- Pipelines: `KNeighborsClassifier(weights="distance")` (`GridSearchCV`, k = 1…41, p ∈ {1, 2}) y `RadiusNeighborsClassifier(outlier_label="most_frequent")` (`RandomizedSearchCV`, 40 radios).
- Desde cero: `KNNPonderadoDesdeCero`, `VecinosRadioDesdeCero` y la función `_votos_ponderados()`.
- Experimento 2D: `experimento_vecinos()` en `experimentos.py`.

#### Resultados
| Caso | k-NN ponderado | Radius Neighbors |
|------|----------------|------------------|
| Cáncer | k = 37, p = 2 → AUC 0.997, F1 0.940 (0.962 con umbral ajustado) | r = 10.8 → AUC 0.996, F1 0.394 (**0.953** con umbral ajustado) |
| Crédito | k = 39, p = 1 → AUC 0.783, F1 0.298 (0.606 ajustado) | r = 4.9 → AUC 0.778, F1 0.145 (0.637 ajustado) |

#### Ventajas y limitaciones
- ✅ Sin entrenamiento, fácil de explicar ("se parece a estos casos") y se adapta de forma natural a muchas clases.
- ✅ Radius Neighbors **detecta puntos fuera de distribución** (bola vacía) y se adapta a regiones de distinta densidad.
- ❌ Predicción lenta con muchos datos y sensible a la escala y a variables irrelevantes.
- ❌ Sufre la maldición de la dimensionalidad. Con variables categóricas en one-hot, las distancias mezclan escalas (en crédito es el peor).

#### Casos de uso
- **Sistemas de recomendación / búsqueda por similitud:** "usuarios parecidos a ti compraron…"; la ponderación 1/d da más voz a los más parecidos.
- **Geolocalización y detección de anomalías (Radius Neighbors):** "¿hay transacciones conocidas en un radio de X km?". Si no hay ninguna, el punto se marca como anómalo. El radio tiene un significado físico directo.

---

## Comparativa final

| Criterio | SVM RBF | SVM Margen Suave | k-NN / Radius |
|----------|---------|------------------|---------------|
| Frontera | No lineal | Lineal | No lineal (local) |
| Entrenamiento | Caro ($O(n^2)$–$O(n^3)$) | Moderado | Ninguno |
| Predicción | Rápida (solo VS) | Muy rápida ($\mathbf{w}^\top \mathbf{x}$) | Lenta ($O(n \cdot d)$) |
| Interpretabilidad | Baja | **Alta** (pesos $\mathbf{w}$) | Media (casos similares) |
| Alta dimensión | Buena | **Muy buena** | Mala |
| Desbalance | `class_weight` | `class_weight` | Solo ajustando el umbral |
| Atípicos | C limita su influencia | C limita su influencia | k pequeño: sensible; Radius: los detecta |
| AUC cáncer / crédito | 0.997 / **0.820** | 0.997 / 0.815 | 0.997 / 0.783 · 0.996 / 0.778 |

**Conclusión:** en datos tabulares de tamaño moderado, las SVM son las más robustas. La de margen suave gana cuando el problema es casi lineal (cáncer), por simplicidad e interpretabilidad; la RBF gana cuando hay interacciones no lineales (crédito). Los vecinos compiten en ranking (AUC), pero necesitan ajuste del umbral, escalado cuidadoso y pocas dimensiones.

---

## Preguntas que pueden hacer en la exposición

**¿Por qué escalan los datos dentro del Pipeline y no antes?**
Si se escala con todos los datos, la media y la desviación del conjunto de prueba influyen en el entrenamiento (fuga de información) y las métricas salen optimistas.

**¿Por qué eligieron los modelos con ROC-AUC y no con exactitud?**
La exactitud engaña con clases desbalanceadas (en crédito, decir siempre "buen pagador" da 70 %). Además, el AUC no depende del umbral y permite comparar la $f(\mathbf{x})$ de la SVM con las probabilidades de los vecinos.

**¿Qué es un vector de soporte?**
Un punto con $\alpha_i > 0$: está sobre el margen o lo viola. Son los únicos que definen la frontera; si se eliminan los demás puntos, el modelo no cambia.

**¿Qué pasa si C → ∞?**
Las violaciones se vuelven infinitamente caras y se recupera la SVM de margen duro. Si los datos no son separables, el problema no tiene solución.

**¿Por qué la SVM lineal empata con la RBF en cáncer?**
El γ óptimo de la RBF fue pequeño (0.01): campanas muy anchas, es decir, una frontera casi lineal. El problema es esencialmente lineal.

**¿Cómo saben que su implementación desde cero es correcta?**
Da los mismos vectores de soporte, el mismo $b$, el mismo Σξ y el 100 % de las predicciones iguales que scikit-learn. Además, la brecha primal − dual es ≈ 0, lo que confirma la dualidad fuerte.

**¿Cuándo usarían Radius Neighbors en vez de k-NN?**
Cuando la distancia tiene un significado físico (kilómetros, segundos) y cuando interesa **detectar** casos sin precedentes (bola vacía) en lugar de forzarles una etiqueta.
