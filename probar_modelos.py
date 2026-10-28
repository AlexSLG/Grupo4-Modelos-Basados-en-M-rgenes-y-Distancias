"""Prueba rápida de los modelos con los mejores hiperparámetros encontrados por experimentos.py.

Entrena cada modelo, muestra su reporte de clasificación en el conjunto de prueba y
predice algunos casos individuales.

Uso:
    python probar_modelos.py                 # caso cáncer de mama
    python probar_modelos.py --caso credito  # caso riesgo crediticio
    python probar_modelos.py --casos 10      # predice 10 casos individuales
"""
import argparse
import sys
import warnings

import pandas as pd
from sklearn.metrics import classification_report

from src.modelos import puntajes
from src.prediccion import entrenar_mejores, ruta_resultados

warnings.filterwarnings("ignore", category=UserWarning)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--caso", choices=["cancer", "credito"], default="cancer")
    parser.add_argument("--casos", type=int, default=5, help="número de casos individuales a predecir")
    args = parser.parse_args()

    ruta = ruta_resultados(args.caso)
    if not ruta.exists():
        sys.exit(f"No existe {ruta}. Ejecuta primero: python experimentos.py")
    caso = entrenar_mejores(args.caso)
    clases, Xte, yte = caso.clases, caso.Xte, caso.yte
    print(f"Caso: {args.caso} | entrenamiento = {len(caso.Xtr)} | prueba = {len(Xte)}\n")

    predicciones = pd.DataFrame({"real": [clases[v] for v in yte.iloc[: args.casos]]}, index=Xte.index[: args.casos])
    for nombre, modelo in caso.modelos.items():
        print("=" * 70)
        print(f"{nombre}  {caso.parametros[nombre]}")
        print("=" * 70)
        print(classification_report(yte, modelo.predict(Xte), target_names=clases, digits=3, zero_division=0))
        muestra = Xte.iloc[: args.casos]
        predicciones[nombre] = [f"{clases[p]} ({s:+.2f})" for p, s in zip(modelo.predict(muestra), puntajes(modelo, muestra))]

    print("Predicciones individuales — entre paréntesis: f(x) en las SVM, P(clase 1) en los vecinos")
    print(predicciones.to_string())


if __name__ == "__main__":
    main()
