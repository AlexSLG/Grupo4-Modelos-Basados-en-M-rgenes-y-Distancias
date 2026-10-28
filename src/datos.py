"""Carga de datos, conjuntos sintéticos 2D y preprocesamiento."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml, load_breast_cancer, make_blobs, make_circles, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import DIR_CACHE, PROPORCION_PRUEBA, SEMILLA


# ---------------------------------------------------------------------------
# Casos reales
# ---------------------------------------------------------------------------
def cargar_cancer():
    """Breast Cancer Wisconsin (diagnóstico médico). 569 muestras, 30 variables numéricas.

    Se recodifica la etiqueta para que la clase positiva (1) sea *maligno*.
    """
    datos = load_breast_cancer(as_frame=True)
    X = datos.data
    y = (datos.target == 0).astype(int).rename("maligno")
    return X, y, ("Benigno", "Maligno")


def cargar_credito():
    """German Credit (riesgo crediticio). 1000 muestras, 7 numéricas + 13 categóricas.

    La clase positiva (1) es *mal pagador* (30 % de los casos: datos desbalanceados).
    Se descarga de OpenML una sola vez y se guarda en ``datos_cache/``.
    """
    datos = fetch_openml("credit-g", version=1, as_frame=True, data_home=str(DIR_CACHE))
    X = datos.data.copy()
    y = (datos.target == "bad").astype(int).rename("riesgo")
    return X, y, ("Buen pagador", "Mal pagador")


def dividir(X, y):
    """División estratificada entrenamiento/prueba (75 % / 25 %)."""
    return train_test_split(X, y, test_size=PROPORCION_PRUEBA, stratify=y, random_state=SEMILLA)


def construir_preprocesador(X: pd.DataFrame):
    """Escala las variables numéricas y codifica one-hot las categóricas.

    Va dentro del ``Pipeline`` para que el escalado se ajuste sólo con los pliegues
    de entrenamiento y no haya fuga de información hacia validación o prueba.
    Escalar es obligatorio en los tres modelos: todos dependen de distancias
    (k-NN, kernel RBF) o de la norma de ``w`` (margen).
    """
    numericas = X.select_dtypes(include="number").columns.tolist()
    categoricas = [c for c in X.columns if c not in numericas]
    transformadores = [("num", StandardScaler(), numericas)]
    if categoricas:
        transformadores.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categoricas)
        )
    return ColumnTransformer(transformadores)


# ---------------------------------------------------------------------------
# Conjuntos sintéticos 2D para visualizar fronteras
# ---------------------------------------------------------------------------
def lunas(n=400, ruido=0.25, semilla=SEMILLA):
    """Dos medias lunas entrelazadas: no separables linealmente (motiva el kernel RBF)."""
    return make_moons(n_samples=n, noise=ruido, random_state=semilla)


def circulos(n=400, ruido=0.08, semilla=SEMILLA):
    """Círculos concéntricos: el ejemplo clásico del mapeo implícito del kernel."""
    return make_circles(n_samples=n, noise=ruido, factor=0.45, random_state=semilla)


def nubes_solapadas(n=200, separacion=2.2, semilla=SEMILLA):
    """Dos nubes gaussianas que se solapan: no existe hiperplano sin errores (margen suave)."""
    centros = np.array([[-separacion / 2, -separacion / 4], [separacion / 2, separacion / 4]])
    return make_blobs(n_samples=n, centers=centros, cluster_std=1.0, random_state=semilla)


def densidad_variable(semilla=SEMILLA, n_atipicos=10):
    """Regiones de densidad muy distinta más puntos atípicos lejanos.

    Útil para contrastar k-NN (vecindario de tamaño fijo en *número*) con
    Radius Neighbors (vecindario de tamaño fijo en *distancia*).
    """
    rng = np.random.default_rng(semilla)
    densa_0 = rng.normal([-1.2, 0.0], 0.35, size=(150, 2))
    dispersa_0 = rng.normal([2.6, 2.2], 0.9, size=(40, 2))
    media_1 = rng.normal([1.0, -0.6], 0.6, size=(130, 2))
    atipicos = rng.uniform([-4, -4], [5, 5], size=(n_atipicos, 2))
    X = np.vstack([densa_0, dispersa_0, media_1, atipicos])
    y = np.concatenate([
        np.zeros(len(densa_0) + len(dispersa_0), dtype=int),
        np.ones(len(media_1), dtype=int),
        rng.integers(0, 2, size=n_atipicos),
    ])
    return X, y
