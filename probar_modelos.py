"""Prueba rápida de los modelos con los mejores hiperparámetros encontrados por experimentos.py.

Entrena cada modelo, muestra su reporte de clasificación en el conjunto de prueba y
predice algunos casos individuales.

Uso:
    python probar_modelos.py                 # caso cáncer de mama
    python probar_modelos.py --caso credito  # caso riesgo crediticio
    python probar_modelos.py --casos 10      # predice 10 casos individuales
"""
import argparse
import ast
import sys
import warnings

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.config import DIR_RESULTADOS, SEMILLA
from src.datos import cargar_cancer, cargar_credito, construir_preprocesador, dividir
from src.modelos import puntajes

warnings.filterwarnings("ignore", category=UserWarning)

CLASIFICADORES = {
    "SVM RBF": lambda p: SVC(kernel="rbf", random_state=SEMILLA, **p),
    "SVM Margen Suave": lambda p: SVC(kernel="linear", random_state=SEMILLA, **p),
    "k-NN Ponderado": lambda p: KNeighborsClassifier(weights="distance", **p),
    "Radius Neighbors": lambda p: RadiusNeighborsClassifier(outlier_label="most_frequent", **p),
}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--caso", choices=["cancer", "credito"], default="cancer")
    parser.add_argument("--casos", type=int, default=5, help="número de casos individuales a predecir")
    args = parser.parse_args()

    ruta = DIR_RESULTADOS / args.caso / "metricas_prueba.csv"
    if not ruta.exists():
        sys.exit(f"No existe {ruta}. Ejecuta primero: python experimentos.py")
    mejores = pd.read_csv(ruta).set_index("modelo")["mejores_parametros"].apply(ast.literal_eval)

    X, y, clases = (cargar_cancer if args.caso == "cancer" else cargar_credito)()
    Xtr, Xte, ytr, yte = dividir(X, y)
    print(f"Caso: {args.caso} | entrenamiento = {len(Xtr)} | prueba = {len(Xte)}\n")

    predicciones = pd.DataFrame({"real": [clases[v] for v in yte.iloc[: args.casos]]}, index=Xte.index[: args.casos])
    for nombre, crear in CLASIFICADORES.items():
        parametros = mejores[nombre]
        modelo = Pipeline([("prep", construir_preprocesador(X)), ("clf", crear(parametros))]).fit(Xtr, ytr)
        print("=" * 70)
        print(f"{nombre}  {parametros}")
        print("=" * 70)
        print(classification_report(yte, modelo.predict(Xte), target_names=clases, digits=3, zero_division=0))
        muestra = Xte.iloc[: args.casos]
        predicciones[nombre] = [f"{clases[p]} ({s:+.2f})" for p, s in zip(modelo.predict(muestra), puntajes(modelo, muestra))]

    print("Predicciones individuales — entre paréntesis: f(x) en las SVM, P(clase 1) en los vecinos")
    print(predicciones.to_string())


if __name__ == "__main__":
    main()
