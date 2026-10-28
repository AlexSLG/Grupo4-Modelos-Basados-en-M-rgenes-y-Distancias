"""Simulación interactiva del Equipo 4.

Ejecutar con:
    streamlit run app_simulacion.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.lines import Line2D
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors, RadiusNeighborsClassifier
from sklearn.svm import SVC

from src import graficas as g
from src import textos as t
from src.config import CLASE_NEG, CLASE_POS, COLOR_MODELO, DIR_FIGURAS, DIR_RESULTADOS, NEUTRO, TEXTO
from src.datos import circulos, densidad_variable, lunas, nubes_solapadas
from src.desde_cero import SVMDualDesdeCero
from src.modelos import puntajes
from src.prediccion import casos_disponibles, entrenar_mejores, predecir, preparar_entrada

st.set_page_config(page_title="Equipo 4 · Márgenes y distancias", layout="wide",
                   initial_sidebar_state="collapsed")
g.aplicar_estilo()

VALORES_LOG = [round(float(v), 4) for v in np.logspace(-3, 3, 25)]
CONJUNTOS = {
    "Lunas (no lineal)": lambda n, ruido, s: lunas(n, ruido, s),
    "Círculos concéntricos": lambda n, ruido, s: circulos(n, ruido / 3, s),
    "Nubes solapadas (lineal con ruido)": lambda n, ruido, s: nubes_solapadas(n, 3.2 - 4 * ruido, s),
    "Densidad variable + atípicos": lambda n, ruido, s: densidad_variable(s, n_atipicos=int(40 * ruido)),
}
NOMBRE_PESOS = {"distance": "1/d (ponderado)", "uniform": "uniforme"}
NOMBRE_METRICA = {2: "Euclídea (p = 2)", 1: "Manhattan (p = 1)"}


@st.cache_data
def generar(nombre, n, ruido, semilla):
    X, y = CONJUNTOS[nombre](n, ruido, semilla)
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=semilla)


@st.cache_resource(show_spinner="Entrenando los cuatro modelos con los mejores hiperparámetros…")
def modelos_caso(caso):
    return entrenar_mejores(caso)


st.title("Familia 2 · Modelos basados en márgenes y distancias")
st.caption("Equipo 4 — SVM con kernel RBF · SVM de margen suave · Radius Neighbors / k-NN ponderado")

pest_sim, pest_teoria, pest_real, pest_caso = st.tabs(["Simulación", "Fundamentos", "Casos reales", "Probar un caso"])

# ---------------------------------------------------------------------------
# Simulación 2D
# ---------------------------------------------------------------------------
with pest_sim:
    st.markdown(t.INTRO_PESTANAS["Simulación"])

    with st.expander("Parámetros de la simulación", expanded=False):
        st.markdown("##### Datos")
        c1, c2, c3, c4 = st.columns(4)
        conjunto = c1.selectbox("Conjunto de datos", list(CONJUNTOS), help=t.AYUDA_CONTROLES["conjunto"])
        n = c2.slider("Número de puntos", 100, 600, 300, 50, disabled=conjunto.startswith("Densidad"),
                      help=t.AYUDA_CONTROLES["n"])
        ruido = c3.slider("Ruido / solapamiento", 0.05, 0.6, 0.25, 0.05, help=t.AYUDA_CONTROLES["ruido"])
        semilla = c4.number_input("Semilla", 0, 9999, 42, help=t.AYUDA_CONTROLES["semilla"])

        st.markdown("##### Modelo")
        m1, m2, m3, m4 = st.columns(4)
        modelo_nombre = m1.radio("Modelo", list(COLOR_MODELO), help=t.AYUDA_CONTROLES["modelo"])
        params = {}
        desde_cero = False
        if modelo_nombre.startswith("SVM"):
            params["C"] = m2.select_slider("C (penalización de ξᵢ)", VALORES_LOG, value=1.0,
                                           help=t.AYUDA_CONTROLES["C"])
            if modelo_nombre == "SVM RBF":
                params["gamma"] = m3.select_slider("γ (alcance del kernel)", VALORES_LOG, value=1.0,
                                                   help=t.AYUDA_CONTROLES["gamma"])
            desde_cero = m4.toggle("Usar implementación desde cero (dual con SLSQP)", value=False,
                                   help=t.AYUDA_CONTROLES["desde_cero"])
        elif modelo_nombre == "k-NN Ponderado":
            params["k"] = m2.slider("k (vecinos)", 1, 60, 15, help=t.AYUDA_CONTROLES["k"])
            params["pesos"] = m3.radio("Ponderación", ["distance", "uniform"], format_func=NOMBRE_PESOS.get,
                                       help=t.AYUDA_CONTROLES["pesos"])
            params["p"] = m4.radio("Métrica de Minkowski", [2, 1], format_func=NOMBRE_METRICA.get,
                                   help=t.AYUDA_CONTROLES["p"])
        else:
            params["radio"] = m2.slider("Radio r", 0.05, 3.0, 0.7, 0.05, help=t.AYUDA_CONTROLES["radio"])
            params["pesos"] = m3.radio("Ponderación", ["distance", "uniform"], format_func=NOMBRE_PESOS.get,
                                       help=t.AYUDA_CONTROLES["pesos"])

    descripcion = {"C": "C", "gamma": "γ", "k": "k", "radio": "r", "pesos": "pesos", "p": "p"}
    valores = {"pesos": NOMBRE_PESOS, "p": NOMBRE_METRICA}
    resumen = " · ".join(f"{descripcion[k]} = {valores.get(k, {}).get(v, v)}" for k, v in params.items())
    st.caption(f"**Datos:** {conjunto} · **Modelo:** {modelo_nombre} ({resumen})"
               + (" · implementación desde cero" if desde_cero else ""))

    # Entrenamiento
    Xtr, Xte, ytr, yte = generar(conjunto, n, ruido, semilla)
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

    # Métricas
    metricas = [
        ("Exactitud entren.", f"{accuracy_score(ytr, pred_tr):.3f}"),
        ("Exactitud prueba", f"{accuracy_score(yte, pred_te):.3f}"),
        ("F1 prueba", f"{f1_score(yte, pred_te):.3f}"),
        ("ROC-AUC prueba", f"{roc_auc_score(yte, s_te):.3f}"),
    ]
    if es_svm:
        metricas += [
            ("Vectores de soporte", len(vectores)),
            ("Σ ξᵢ", f"{xi.sum():.1f}"),
            ("Dentro del margen", int(((xi > 0) & (xi <= 1)).sum())),
            ("Mal clasificados", int((xi > 1).sum())),
        ]
        if modelo_nombre == "SVM Margen Suave":
            ancho = modelo.ancho_margen() if desde_cero else 2 / np.linalg.norm(modelo.coef_)
            metricas.append(("Ancho del margen 2/‖w‖", f"{ancho:.3f}"))
        if desde_cero:
            metricas.append(("Brecha de dualidad", f"{modelo.objetivo_primal() - modelo.objetivo_dual():.2e}"))
    if zona_vacia is not None:
        metricas.append(("Prueba sin vecinos", f"{zona_vacia(Xte).mean():.0%}"))
    for inicio in range(0, len(metricas), 4):
        for col, (etiqueta, valor) in zip(st.columns(4), metricas[inicio:inicio + 4]):
            col.metric(etiqueta, valor, help=t.AYUDA_METRICAS[etiqueta])

    # Punto nuevo
    st.markdown("##### Probar un punto nuevo")
    p1, p2, p3 = st.columns([1, 1, 3])
    x1 = p1.number_input("Coordenada x₁", value=0.0, step=0.1, format="%.2f", help=t.AYUDA_CONTROLES["x1"])
    x2 = p2.number_input("Coordenada x₂", value=0.0, step=0.1, format="%.2f", help=t.AYUDA_CONTROLES["x2"])
    punto = np.array([[x1, x2]])
    clase_punto = int(modelo.predict(punto)[0])
    s_punto = float(puntajes(modelo, punto)[0])
    if es_svm:
        detalle = f"f(x) = {s_punto:+.3f}" + (" (fuera del margen)" if abs(s_punto) >= 1 else " (dentro del margen)")
    else:
        detalle = f"P(clase 1) = {s_punto:.3f}"
    mensaje = f"El modelo predice la **clase {clase_punto}** ({'roja' if clase_punto else 'azul'}) · {detalle}."
    if modelo_nombre == "Radius Neighbors":
        n_vecinos = len(buscador.radius_neighbors(punto, radius=params["radio"], return_distance=False)[0])
        mensaje += (f" Tiene **{n_vecinos} vecinos** dentro del radio r = {params['radio']}." if n_vecinos else
                    " **No tiene vecinos** dentro del radio: recibe la clase más frecuente del entrenamiento.")
    elif modelo_nombre == "k-NN Ponderado":
        mensaje += f" Votaron sus {modelo.n_neighbors} vecinos más cercanos."
    p3.info(mensaje)

    fig, ax = plt.subplots(figsize=(9, 5.6))
    g.frontera_2d(ax, modelo, Xtr, ytr, margenes=es_svm, vectores_soporte=vectores,
                  zona_sin_vecinos=zona_vacia, resolucion=200)
    (xmin, xmax), (ymin, ymax) = ax.get_xlim(), ax.get_ylim()
    fuera = not (xmin <= x1 <= xmax and ymin <= x2 <= ymax)
    ax.scatter([x1], [x2], marker="*", s=420, c=CLASE_POS if clase_punto else CLASE_NEG,
               edgecolors=TEXTO, linewidths=1.2, zorder=6, clip_on=False)
    estrella = Line2D([], [], marker="*", ls="", markerfacecolor=NEUTRO, markeredgecolor=TEXTO,
                      markersize=13, label="Punto nuevo")
    g.leyenda_frontera(fig, margenes=es_svm, vectores=es_svm, sin_vecinos=zona_vacia is not None, extra=[estrella])
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    st.pyplot(fig)
    plt.close(fig)
    st.caption(t.LEYENDA_FRONTERA + (t.LEYENDA_FRONTERA_SVM if es_svm else "")
               + (t.LEYENDA_FRONTERA_RADIO if zona_vacia is not None else "") + t.LEYENDA_ESTRELLA
               + (" Ahora está fuera del área dibujada." if fuera else ""))

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
        st.caption(t.LEYENDA_HOLGURAS)
        if modelo_nombre == "SVM RBF" and st.checkbox(
                "Ver la superficie 3D de f(x)", help="Dibuja f(x) como un relieve sobre el plano."):
            fig = plt.figure(figsize=(8, 6))
            g.superficie_3d(fig.add_subplot(projection="3d"), modelo, Xtr, ytr)
            st.pyplot(fig)
            plt.close(fig)
            st.caption(t.LEYENDA_3D)

# ---------------------------------------------------------------------------
# Fundamentos
# ---------------------------------------------------------------------------
with pest_teoria:
    st.markdown(t.INTRO_PESTANAS["Fundamentos"])
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

# ---------------------------------------------------------------------------
# Casos reales
# ---------------------------------------------------------------------------
with pest_real:
    st.markdown(t.INTRO_PESTANAS["Casos reales"])
    casos = casos_disponibles()
    if not casos:
        st.info("Ejecuta primero `python experimentos.py` para generar los resultados de los casos reales.")
    else:
        caso = st.radio("Caso", casos, horizontal=True, format_func=t.NOMBRES_CASOS.get, key="caso_resultados",
                        help="Problema real sobre el que se evaluaron los cuatro modelos.")
        tabla = pd.read_csv(DIR_RESULTADOS / caso / "metricas_prueba.csv")
        columnas = [c for c in t.COLUMNAS_TABLA if c in tabla]
        st.dataframe(tabla[columnas].round(3), hide_index=True, width="stretch",
                     column_config={c: st.column_config.Column(help=t.COLUMNAS_TABLA[c]) for c in columnas})
        st.caption("Pasa el cursor sobre el nombre de cada columna para ver qué mide. Las métricas son sobre el "
                   "conjunto de prueba; las que terminan en *_umbral* usan el umbral ajustado para maximizar F1.")
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
                st.caption(t.LEYENDAS_CASOS[sufijo])


# ---------------------------------------------------------------------------
# Probar un caso
# ---------------------------------------------------------------------------
def _paso_y_formato(serie):
    """Paso de ~1 % del rango de la variable, con los decimales necesarios para mostrarlo."""
    rango = float(serie.max() - serie.min()) or 1.0
    exponente = int(np.floor(np.log10(rango / 100)))
    return 10.0 ** exponente, f"%.{max(0, -exponente)}f"


def campo(contenedor, Xtr, columna, valor, etiqueta, ayuda, clave):
    """Un control de entrada para una variable, con límites tomados del entrenamiento."""
    serie = Xtr[columna]
    if isinstance(serie.dtype, pd.CategoricalDtype):
        opciones = list(serie.cat.categories)
        return contenedor.selectbox(etiqueta, opciones, index=opciones.index(valor), help=ayuda, key=clave,
                                    format_func=lambda v: t.nombre_categoria(columna, v))
    if pd.api.types.is_integer_dtype(serie):
        valor = int(round(valor))
        return contenedor.number_input(etiqueta, min_value=min(int(serie.min()), valor),
                                       max_value=max(int(serie.max()), valor), value=valor, step=1,
                                       help=ayuda, key=clave)
    paso, formato = _paso_y_formato(serie)
    valor = float(valor)
    return contenedor.number_input(etiqueta, min_value=min(float(serie.min()), valor),
                                   max_value=max(float(serie.max()), valor), value=valor, step=paso,
                                   format=formato, help=ayuda, key=clave)


def formulario(caso, entrenado, base, prefijo):
    """Formulario con un campo por variable, agrupado en expanders. Devuelve un DataFrame de una fila."""
    if caso == "cancer":
        st.caption(t.NOTA_CANCER)
        grupos = t.variables_cancer()
        por_fila = 5
    else:
        st.caption(t.NOTA_CREDITO)
        grupos = {grupo: [(c, *t.VARIABLES_CREDITO[c]) for c in cols] for grupo, cols in t.GRUPOS_CREDITO.items()}
        por_fila = 3
    valores = {}
    for i, (grupo, variables) in enumerate(grupos.items()):
        with st.expander(grupo, expanded=i == 0):
            columnas = st.columns(por_fila)
            for j, (col, etiqueta, ayuda) in enumerate(variables):
                valores[col] = campo(columnas[j % por_fila], entrenado.Xtr, col, base[col], etiqueta, ayuda,
                                     f"{prefijo}-{col}")
    return pd.DataFrame([valores], columns=entrenado.X.columns)


def mostrar_resultado(entrenado, X_nuevo, reales):
    """Tabla con la predicción de cada modelo y, si hay un solo caso, la gráfica de confianza."""
    negativa, positiva = entrenado.clases
    res = predecir(entrenado, X_nuevo)
    tabla = pd.DataFrame({
        "Caso": res.fila,
        "Modelo": res.modelo,
        "Predicción": [entrenado.clases[p] for p in res.prediccion],
        "Confianza": [f"f(x) = {s:+.3f}" if svm else f"P({positiva}) = {s:.3f}"
                      for s, svm in zip(res.puntaje, res.es_svm)],
        "Seguridad": 100 * res.seguridad,
        "Con umbral ajustado": [entrenado.clases[p] if p is not None else "—" for p in res.prediccion_umbral],
    })
    if reales is not None:
        tabla["Real"] = [entrenado.clases[reales[f]] for f in res.fila]
        tabla["¿Acertó?"] = np.where(res.prediccion.to_numpy() == [reales[f] for f in res.fila], "Sí", "No")
    if len(X_nuevo) == 1:
        tabla = tabla.drop(columns="Caso")
    st.dataframe(tabla, hide_index=True, width="stretch", column_config={
        "Caso": st.column_config.Column(help="Fila del archivo o índice del caso en el conjunto de datos."),
        "Predicción": st.column_config.Column(help="Clase que predice el modelo con el umbral por defecto."),
        "Confianza": st.column_config.Column(help="f(x) en las SVM; probabilidad de la clase positiva en los vecinos."),
        "Seguridad": st.column_config.ProgressColumn(
            help="Qué tan lejos está el caso del umbral por defecto, de 0 a 100 %.",
            min_value=0, max_value=100, format="%.0f %%"),
        "Con umbral ajustado": st.column_config.Column(help="Predicción con el umbral que maximiza F1."),
        "Real": st.column_config.Column(help="Etiqueta verdadera del caso original."),
    })
    if len(X_nuevo) == 1:
        fig, (ax_svm, ax_vec) = plt.subplots(1, 2, figsize=(11, 2.6))
        elementos = g.confianza_modelos(ax_svm, ax_vec, res, negativa, positiva)
        fig.legend(handles=elementos, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        st.pyplot(fig)
        plt.close(fig)
        st.caption(t.LEYENDA_GRAFICA_CONFIANZA.format(positiva=positiva, negativa=negativa))
    st.caption(t.LEYENDA_CONFIANZA.format(positiva=positiva, negativa=negativa))
    st.info(t.AVISO_UMBRAL.format(positiva=positiva))


with pest_caso:
    st.markdown(t.INTRO_PESTANAS["Probar un caso"])
    st.warning(t.AVISO_EDUCATIVO)
    casos = casos_disponibles()
    if not casos:
        st.info("Ejecuta primero `python experimentos.py` para generar los resultados de los casos reales.")
    else:
        caso = st.radio("Caso", casos, horizontal=True, format_func=t.NOMBRES_CASOS.get, key="caso_prueba",
                        help="Problema real sobre el que quieres probar los modelos.")
        entrenado = modelos_caso(caso)
        modo = st.radio("¿Cómo quieres ingresar los datos?", list(t.MODOS_ENTRADA), horizontal=True,
                        format_func=t.MODOS_ENTRADA.get, key="modo_entrada",
                        help="Partir de un caso real del conjunto de prueba, escribir los valores a mano o "
                             "subir un CSV con varios casos.")

        if modo in ("prueba", "manual"):
            if modo == "prueba":
                indice = st.selectbox(
                    "Caso del conjunto de prueba", list(entrenado.Xte.index), key=f"indice-{caso}",
                    format_func=lambda i: f"#{i} · real: {entrenado.clases[entrenado.yte.loc[i]]}",
                    help="Casos que el modelo nunca vio al entrenar. Sus valores se cargan en el formulario "
                         "y puedes modificarlos.")
                st.markdown(f"Etiqueta real del caso **#{indice}**: **{entrenado.clases[entrenado.yte.loc[indice]]}**")
                base, prefijo, reales = entrenado.Xte.loc[indice], f"{caso}-prueba-{indice}", {0: int(entrenado.yte.loc[indice])}
            else:
                base = {c: (entrenado.Xtr[c].mode()[0] if isinstance(entrenado.Xtr[c].dtype, pd.CategoricalDtype)
                            else entrenado.Xtr[c].median()) for c in entrenado.X.columns}
                prefijo, reales = f"{caso}-manual", None
                st.caption("Cada campo empieza en la mediana (o en la categoría más frecuente) de los datos de "
                           "entrenamiento. Los límites de cada campo son el mínimo y el máximo observados.")
            with st.form(f"formulario-{prefijo}"):
                fila = formulario(caso, entrenado, base, prefijo)
                enviado = st.form_submit_button("Predecir", type="primary")
            if enviado:
                X_nuevo, errores = preparar_entrada(fila, entrenado.X)
                if errores:
                    st.error("\n".join(errores))
                else:
                    st.session_state["resultado_caso"] = dict(clave=prefijo, X=X_nuevo, reales=reales)
        else:
            plantilla = entrenado.Xte.head(1).to_csv(index=False).encode("utf-8")
            st.download_button("Descargar plantilla CSV", plantilla, file_name=f"plantilla_{caso}.csv",
                               mime="text/csv", help="Archivo con las columnas correctas y un caso de ejemplo. "
                                                     "Agrega una fila por cada caso que quieras predecir.")
            archivo = st.file_uploader("Archivo CSV con los casos", type="csv", key=f"csv-{caso}",
                                       help="Debe tener exactamente las mismas columnas que la plantilla.")
            if archivo is not None:
                X_nuevo, errores = preparar_entrada(pd.read_csv(archivo), entrenado.X)
                if errores:
                    st.error("No se puede usar el archivo:\n\n" + "\n".join(f"- {e}" for e in errores))
                else:
                    st.success(f"Archivo válido: {len(X_nuevo)} caso(s).")
                    if st.button("Predecir", type="primary", key="predecir-csv"):
                        X_nuevo.index = pd.RangeIndex(1, len(X_nuevo) + 1)
                        st.session_state["resultado_caso"] = dict(clave=f"{caso}-csv-{archivo.name}",
                                                                  X=X_nuevo, reales=None)
            prefijo = f"{caso}-csv-{archivo.name}" if archivo is not None else None

        resultado = st.session_state.get("resultado_caso")
        if resultado and resultado["clave"] == prefijo:
            st.subheader("Resultado")
            mostrar_resultado(entrenado, resultado["X"], resultado["reales"])
