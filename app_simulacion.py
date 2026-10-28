"""Simulación interactiva del Equipo 4.

Ejecutar con:
    streamlit run app_simulacion.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors, RadiusNeighborsClassifier
from sklearn.svm import SVC

from src import graficas as g
from src.config import COLOR_MODELO, DIR_FIGURAS, DIR_RESULTADOS
from src.datos import circulos, densidad_variable, lunas, nubes_solapadas
from src.desde_cero import SVMDualDesdeCero
from src.modelos import puntajes

st.set_page_config(page_title="Equipo 4 · Márgenes y distancias", layout="wide")
g.aplicar_estilo()

VALORES_LOG = [round(float(v), 4) for v in np.logspace(-3, 3, 25)]
CONJUNTOS = {
    "Lunas (no lineal)": lambda n, ruido, s: lunas(n, ruido, s),
    "Círculos concéntricos": lambda n, ruido, s: circulos(n, ruido / 3, s),
    "Nubes solapadas (lineal con ruido)": lambda n, ruido, s: nubes_solapadas(n, 3.2 - 4 * ruido, s),
    "Densidad variable + atípicos": lambda n, ruido, s: densidad_variable(s, n_atipicos=int(40 * ruido)),
}


@st.cache_data
def generar(nombre, n, ruido, semilla):
    X, y = CONJUNTOS[nombre](n, ruido, semilla)
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=semilla)


# ---------------------------------------------------------------------------
# Controles
# ---------------------------------------------------------------------------
st.sidebar.title("Parámetros")
conjunto = st.sidebar.selectbox("Conjunto de datos", list(CONJUNTOS))
n = st.sidebar.slider("Número de puntos", 100, 600, 300, 50, disabled=conjunto.startswith("Densidad"))
ruido = st.sidebar.slider("Ruido / solapamiento", 0.05, 0.6, 0.25, 0.05)
semilla = st.sidebar.number_input("Semilla", 0, 9999, 42)

st.sidebar.divider()
modelo_nombre = st.sidebar.radio("Modelo", list(COLOR_MODELO))
Xtr, Xte, ytr, yte = generar(conjunto, n, ruido, semilla)

params = {}
if modelo_nombre.startswith("SVM"):
    params["C"] = st.sidebar.select_slider("C (penalización de ξᵢ)", VALORES_LOG, value=1.0)
    if modelo_nombre == "SVM RBF":
        params["gamma"] = st.sidebar.select_slider("γ (alcance del kernel)", VALORES_LOG, value=1.0)
    desde_cero = st.sidebar.toggle("Usar implementación desde cero (dual con SLSQP)", value=False,
                                   help="Resuelve el dual con SciPy. Lento para más de ~300 puntos.")
elif modelo_nombre == "k-NN Ponderado":
    params["k"] = st.sidebar.slider("k (vecinos)", 1, 60, 15)
    params["pesos"] = st.sidebar.radio("Ponderación", ["distance", "uniform"],
                                       format_func=lambda w: "1/d (ponderado)" if w == "distance" else "uniforme")
    params["p"] = st.sidebar.radio("Métrica de Minkowski", [2, 1],
                                   format_func=lambda p: "Euclídea (p = 2)" if p == 2 else "Manhattan (p = 1)")
else:
    params["radio"] = st.sidebar.slider("Radio r", 0.05, 3.0, 0.7, 0.05)
    params["pesos"] = st.sidebar.radio("Ponderación", ["distance", "uniform"],
                                       format_func=lambda w: "1/d (ponderado)" if w == "distance" else "uniforme")

# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
if modelo_nombre == "SVM RBF":
    modelo = (SVMDualDesdeCero(C=params["C"], kernel="rbf", gamma=params["gamma"]) if desde_cero
              else SVC(kernel="rbf", C=params["C"], gamma=params["gamma"]))
elif modelo_nombre == "SVM Margen Suave":
    modelo = (SVMDualDesdeCero(C=params["C"], kernel="lineal") if desde_cero
              else SVC(kernel="linear", C=params["C"]))
elif modelo_nombre == "k-NN Ponderado":
    modelo = KNeighborsClassifier(n_neighbors=min(params["k"], len(Xtr)), weights=params["pesos"], p=params["p"])
else:
    modelo = RadiusNeighborsClassifier(radius=params["radio"], weights=params["pesos"], outlier_label="most_frequent")
modelo.fit(Xtr, ytr)

es_svm = modelo_nombre.startswith("SVM")
pred_tr, pred_te = modelo.predict(Xtr), modelo.predict(Xte)
s_te = puntajes(modelo, Xte)
vectores = None
if es_svm:
    vectores = modelo.X_vs_ if desde_cero else modelo.support_vectors_
    f_tr = modelo.decision_function(Xtr)
    xi = np.maximum(0.0, 1.0 - np.where(ytr == 1, 1.0, -1.0) * f_tr)

zona_vacia = None
if modelo_nombre == "Radius Neighbors":
    buscador = NearestNeighbors().fit(Xtr)

    def zona_vacia(Z):
        vecinos = buscador.radius_neighbors(Z, radius=params["radio"], return_distance=False)
        return np.array([len(v) == 0 for v in vecinos])

# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------
st.title("Familia 2 · Modelos basados en márgenes y distancias")
st.caption("Equipo 4 — SVM con kernel RBF · SVM de margen suave · Radius Neighbors / k-NN ponderado")

pest_sim, pest_teoria, pest_real = st.tabs(["Simulación", "Fundamentos", "Casos reales"])

with pest_sim:
    metricas = [
        ("Exactitud entren.", f"{accuracy_score(ytr, pred_tr):.3f}", None),
        ("Exactitud prueba", f"{accuracy_score(yte, pred_te):.3f}", None),
        ("F1 prueba", f"{f1_score(yte, pred_te):.3f}", None),
        ("ROC-AUC prueba", f"{roc_auc_score(yte, s_te):.3f}", None),
    ]
    if es_svm:
        metricas += [
            ("Vectores de soporte", len(vectores), None),
            ("Σ ξᵢ", f"{xi.sum():.1f}", None),
            ("Dentro del margen", int(((xi > 0) & (xi <= 1)).sum()), "0 < ξᵢ ≤ 1: bien clasificados pero violan el margen"),
            ("Mal clasificados", int((xi > 1).sum()), "ξᵢ > 1"),
        ]
        if modelo_nombre == "SVM Margen Suave":
            ancho = modelo.ancho_margen() if desde_cero else 2 / np.linalg.norm(modelo.coef_)
            metricas.append(("Ancho del margen 2/‖w‖", f"{ancho:.3f}", None))
        if desde_cero:
            metricas.append(("Brecha de dualidad", f"{modelo.objetivo_primal() - modelo.objetivo_dual():.2e}",
                             "Primal − dual. ≈ 0 indica que se alcanzó el óptimo (dualidad fuerte)."))
    if zona_vacia is not None:
        metricas.append(("Prueba sin vecinos", f"{zona_vacia(Xte).mean():.0%}",
                         "Puntos sin vecinos dentro del radio: reciben la clase más frecuente (outlier_label)"))
    for inicio in range(0, len(metricas), 4):
        for col, (etiqueta, valor, ayuda) in zip(st.columns(4), metricas[inicio:inicio + 4]):
            col.metric(etiqueta, valor, help=ayuda)

    fig, ax = plt.subplots(figsize=(9, 5.6))
    g.frontera_2d(ax, modelo, Xtr, ytr, margenes=es_svm, vectores_soporte=vectores,
                  zona_sin_vecinos=zona_vacia, resolucion=200)
    g.leyenda_frontera(fig, margenes=es_svm, vectores=es_svm, sin_vecinos=zona_vacia is not None)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    st.pyplot(fig)
    plt.close(fig)

    if es_svm:
        st.subheader("Variables de holgura ξᵢ = max(0, 1 − yᵢ f(xᵢ))")
        xi_pos = np.sort(xi[xi > 1e-6])[::-1]
        fig, ax = plt.subplots(figsize=(10, 2.6))
        ax.bar(np.arange(len(xi_pos)), xi_pos, width=0.8,
               color=np.where(xi_pos > 1, COLOR_MODELO["SVM Margen Suave"], COLOR_MODELO["SVM RBF"]))
        ax.axhline(1, color=g.TEXTO_2, lw=1, ls="--")
        ax.set_xlabel("Puntos con ξᵢ > 0 (azul: dentro del margen · naranja: mal clasificados)")
        ax.grid(axis="x", visible=False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        if modelo_nombre == "SVM RBF" and st.checkbox("Ver la superficie 3D de f(x)"):
            fig = plt.figure(figsize=(8, 6))
            g.superficie_3d(fig.add_subplot(projection="3d"), modelo, Xtr, ytr)
            st.pyplot(fig)
            plt.close(fig)

with pest_teoria:
    st.header("1 · SVM con kernel RBF")
    st.latex(r"K(\mathbf{x},\mathbf{z}) = \exp\!\left(-\gamma\,\lVert \mathbf{x}-\mathbf{z}\rVert^2\right)")
    st.latex(r"h(\mathbf{x}) = \operatorname{sign}\Big(\sum_{i\in SV} \alpha_i y_i\, K(\mathbf{x}_i,\mathbf{x}) + b\Big)")
    st.markdown("γ grande → cada vector de soporte influye sólo en su vecindad inmediata (frontera muy curva, "
                "riesgo de sobreajuste). γ pequeño → frontera casi lineal.")
    st.header("2 · SVM de margen suave")
    st.latex(r"\min_{\mathbf{w},b,\boldsymbol\xi}\ \tfrac12\lVert\mathbf{w}\rVert^2 + C\sum_{i=1}^n \xi_i"
             r"\quad\text{s.a.}\quad y_i(\mathbf{w}^\top\mathbf{x}_i+b)\ge 1-\xi_i,\ \ \xi_i\ge 0")
    st.markdown("Equivale a minimizar la **pérdida bisagra** regularizada:")
    st.latex(r"\min_{\mathbf{w},b}\ \tfrac12\lVert\mathbf{w}\rVert^2 + C\sum_i \max\big(0,\ 1-y_i(\mathbf{w}^\top\mathbf{x}_i+b)\big)")
    st.markdown("Dual (programación cuadrática, resuelto por SMO en libsvm):")
    st.latex(r"\max_{\boldsymbol\alpha}\ \sum_i\alpha_i-\tfrac12\sum_{i,j}\alpha_i\alpha_j y_i y_j K(\mathbf{x}_i,\mathbf{x}_j)"
             r"\quad\text{s.a.}\quad 0\le\alpha_i\le C,\ \ \sum_i\alpha_i y_i=0")
    st.header("3 · Radius Neighbors / k-NN ponderado")
    st.latex(r"\hat y(\mathbf{x}) = \arg\max_{c}\sum_{i\in\mathcal N(\mathbf{x})} w_i\,\mathbb 1[y_i=c],"
             r"\qquad w_i=\frac{1}{d(\mathbf{x},\mathbf{x}_i)}")
    st.latex(r"\mathcal N_k(\mathbf{x}) = \text{los } k \text{ puntos más cercanos}\qquad"
             r"\mathcal N_r(\mathbf{x}) = \{\,i : d(\mathbf{x},\mathbf{x}_i)\le r\,\}")
    st.markdown("No hay función de pérdida que optimizar durante el entrenamiento (aprendizaje *perezoso*): "
                "k o r se eligen por validación cruzada.")

with pest_real:
    casos = [c for c in ("cancer", "credito") if (DIR_RESULTADOS / c / "metricas_prueba.csv").exists()]
    if not casos:
        st.info("Ejecuta primero `python experimentos.py` para generar los resultados de los casos reales.")
    else:
        caso = st.radio("Caso", casos, horizontal=True,
                        format_func=lambda c: {"cancer": "Diagnóstico de cáncer de mama",
                                               "credito": "Riesgo crediticio (German Credit)"}[c])
        tabla = pd.read_csv(DIR_RESULTADOS / caso / "metricas_prueba.csv")
        columnas = ["modelo", "accuracy", "precision", "recall", "f1", "roc_auc", "auc_cv",
                    "recall_umbral", "f1_umbral", "vectores_soporte", "tiempo_busqueda_s", "mejores_parametros"]
        st.dataframe(tabla[[c for c in columnas if c in tabla]].round(3), hide_index=True, width="stretch")
        for sufijo, titulo in (("roc", "Curvas ROC"), ("comparativa_metricas", "Métricas en prueba"),
                               ("ajuste_umbral", "Ajuste del umbral de decisión"),
                               ("matrices_confusion", "Matrices de confusión"),
                               ("curvas_aprendizaje", "Curvas de aprendizaje"),
                               ("sensibilidad_hiperparametros", "Sensibilidad a los hiperparámetros"),
                               ("cv_repetida", "Validación cruzada repetida"),
                               ("importancia_permutacion", "Importancia por permutación")):
            ruta = DIR_FIGURAS / f"{caso}_{sufijo}.png"
            if ruta.exists():
                st.subheader(titulo)
                st.image(str(ruta))
