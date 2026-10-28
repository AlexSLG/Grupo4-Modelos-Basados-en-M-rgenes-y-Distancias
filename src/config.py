"""Constantes compartidas: semilla, rutas y paleta de colores."""
from pathlib import Path

SEMILLA = 42
PROPORCION_PRUEBA = 0.25
PLIEGUES_CV = 5

RAIZ = Path(__file__).resolve().parent.parent
DIR_RESULTADOS = RAIZ / "resultados"
DIR_FIGURAS = DIR_RESULTADOS / "figuras"
DIR_CACHE = RAIZ / "datos_cache"

# Superficies y tinta (paleta de referencia validada para daltonismo)
SUPERFICIE = "#fcfcfb"
TEXTO = "#0b0b0b"
TEXTO_2 = "#52514e"
REJILLA = "#e6e5e0"
EJES = "#c9c8c2"

# Polaridad de clase: clase negativa (azul) <-> clase positiva (rojo), punto medio gris
CLASE_NEG = "#2a78d6"
CLASE_POS = "#e34948"
NEUTRO = "#f0efec"

# Identidad de cada modelo, en orden fijo de la paleta categórica
COLOR_MODELO = {
    "SVM RBF": "#2a78d6",
    "SVM Margen Suave": "#eb6834",
    "k-NN Ponderado": "#1baf7a",
    "Radius Neighbors": "#eda100",
}

# Rampa secuencial azul (claro -> oscuro) para mapas de calor
RAMPA_AZUL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
