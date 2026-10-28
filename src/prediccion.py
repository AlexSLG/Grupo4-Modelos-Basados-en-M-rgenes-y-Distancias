"""Modelos finales de los casos reales: entrenamiento con los mejores hiperparámetros y predicción.

Lo usan probar_modelos.py y la pestaña "Probar un caso" de la app. Los hiperparámetros y el
umbral ajustado se leen de ``resultados/<caso>/metricas_prueba.csv`` (generado por experimentos.py).
"""
import ast
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from .config import DIR_RESULTADOS, SEMILLA
from .datos import cargar_cancer, cargar_credito, construir_preprocesador, dividir
from .modelos import puntajes

CASOS = {"cancer": cargar_cancer, "credito": cargar_credito}

CLASIFICADORES = {
    "SVM RBF": lambda p: SVC(kernel="rbf", random_state=SEMILLA, **p),
    "SVM Margen Suave": lambda p: SVC(kernel="linear", random_state=SEMILLA, **p),
    "k-NN Ponderado": lambda p: KNeighborsClassifier(weights="distance", **p),
    "Radius Neighbors": lambda p: RadiusNeighborsClassifier(outlier_label="most_frequent", **p),
}


@dataclass
class CasoEntrenado:
    X: pd.DataFrame
    Xtr: pd.DataFrame
    Xte: pd.DataFrame
    ytr: pd.Series
    yte: pd.Series
    clases: tuple
    parametros: dict
    umbrales: dict
    modelos: dict


def ruta_resultados(caso):
    return DIR_RESULTADOS / caso / "metricas_prueba.csv"


def casos_disponibles():
    """Casos con resultados ya generados por experimentos.py."""
    return [c for c in CASOS if ruta_resultados(c).exists()]


def entrenar_mejores(caso):
    """Carga el caso, lo divide igual que experimentos.py y entrena los cuatro Pipelines finales."""
    tabla = pd.read_csv(ruta_resultados(caso)).set_index("modelo")
    parametros = tabla["mejores_parametros"].apply(ast.literal_eval).to_dict()
    umbrales = tabla["umbral_f1"].to_dict() if "umbral_f1" in tabla else {}

    X, y, clases = CASOS[caso]()
    Xtr, Xte, ytr, yte = dividir(X, y)
    modelos = {
        nombre: Pipeline([("prep", construir_preprocesador(X)), ("clf", crear(parametros[nombre]))]).fit(Xtr, ytr)
        for nombre, crear in CLASIFICADORES.items()
    }
    return CasoEntrenado(X, Xtr, Xte, ytr, yte, clases, parametros, umbrales, modelos)


def preparar_entrada(datos, X_referencia):
    """Valida un DataFrame de entrada y le da los mismos tipos que los datos originales.

    Devuelve (DataFrame listo para predecir, lista de errores). Si hay errores, el DataFrame es None.
    """
    columnas = list(X_referencia.columns)
    faltan = [c for c in columnas if c not in datos.columns]
    sobran = [c for c in datos.columns if c not in columnas]
    errores = []
    if faltan:
        errores.append(f"Faltan columnas: {', '.join(faltan)}")
    if sobran:
        errores.append(f"Sobran columnas: {', '.join(sobran)}")
    if errores:
        return None, errores
    if datos.empty:
        return None, ["El archivo no tiene filas."]

    datos = datos[columnas].copy()
    for c in columnas:
        if isinstance(X_referencia[c].dtype, pd.CategoricalDtype):
            categorias = X_referencia[c].cat.categories
            invalidos = sorted(set(datos[c].dropna().astype(str)) - set(categorias))
            if invalidos or datos[c].isna().any():
                errores.append(f"Valores no válidos en «{c}»: {', '.join(invalidos) or 'celdas vacías'}. "
                               f"Valores permitidos: {', '.join(categorias)}")
            else:
                datos[c] = pd.Categorical(datos[c].astype(str), categories=categorias)
        else:
            convertida = pd.to_numeric(datos[c], errors="coerce")
            if convertida.isna().any():
                errores.append(f"La columna «{c}» debe ser numérica y no puede tener celdas vacías.")
            else:
                datos[c] = convertida.astype(X_referencia[c].dtype)
    return (None, errores) if errores else (datos, [])


def predecir(entrenado, X_nuevo):
    """Una fila por (caso, modelo) con la predicción, el puntaje y la predicción con el umbral ajustado.

    El puntaje es f(x) en las SVM y P(clase positiva) en los vecinos (``puntajes``). La seguridad
    lleva ambos a [0, 1]: |f(x)| hasta 1 (el borde del margen) o |P − 0.5| · 2.
    """
    filas = []
    for nombre, modelo in entrenado.modelos.items():
        es_svm = hasattr(modelo, "decision_function")
        s = puntajes(modelo, X_nuevo)
        pred = modelo.predict(X_nuevo)
        umbral = entrenado.umbrales.get(nombre, np.nan)
        seguridad = np.minimum(np.abs(s), 1.0) if es_svm else np.abs(s - 0.5) * 2
        for fila, si, pi, gi in zip(X_nuevo.index, s, pred, seguridad):
            filas.append(dict(
                fila=fila, modelo=nombre, es_svm=es_svm, prediccion=int(pi), puntaje=float(si),
                seguridad=float(gi), umbral=float(umbral),
                prediccion_umbral=int(si >= umbral) if np.isfinite(umbral) else None,
            ))
    return pd.DataFrame(filas)
