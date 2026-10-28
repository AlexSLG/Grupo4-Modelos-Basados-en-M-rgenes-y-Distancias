"""Experimentos del Equipo 4: SVM RBF, SVM de margen suave y Radius Neighbors / k-NN ponderado.

Genera en ``resultados/`` todas las tablas (CSV/JSON) y figuras (PNG) del informe técnico.

Uso:
    python experimentos.py            # experimento completo
    python experimentos.py --rapido   # rejillas reducidas, para una prueba rápida
"""
import argparse
import json
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV, RepeatedStratifiedKFold, TunedThresholdClassifierCV, cross_val_score, learning_curve,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors, RadiusNeighborsClassifier
from sklearn.svm import SVC

from src import graficas as g
from src.config import CLASE_NEG, CLASE_POS, COLOR_MODELO, DIR_RESULTADOS, SEMILLA, TEXTO_2
from src.datos import (
    cargar_cancer, cargar_credito, circulos, construir_preprocesador, densidad_variable, dividir,
    lunas, nubes_solapadas,
)
from src.desde_cero import KNNPonderadoDesdeCero, SVMDualDesdeCero, VecinosRadioDesdeCero
from src.modelos import construir_busquedas, puntajes

warnings.filterwarnings("ignore", category=UserWarning)


def dividir_2d(X, y):
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEMILLA)


def ancho_y_holguras(svm, X, y):
    """Para una SVC lineal: ancho del margen 2/||w|| y holguras ξ_i = max(0, 1 - y_i f(x_i))."""
    s = np.where(y == 1, 1.0, -1.0)
    xi = np.maximum(0.0, 1.0 - s * svm.decision_function(X))
    return 2.0 / np.linalg.norm(svm.coef_), xi


# ===========================================================================
# 1. SVM con kernel RBF (gaussiano)
# ===========================================================================
def experimento_rbf():
    print("[1/3] SVM con kernel RBF")
    X, y = lunas()
    Xtr, Xte, ytr, yte = dividir_2d(X, y)

    # Efecto conjunto de gamma (alcance del kernel) y C (penalización de errores)
    gammas, cs = [0.1, 1, 10], [0.1, 1, 100]
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 10.5))
    for i, gamma in enumerate(gammas):
        for j, C in enumerate(cs):
            m = SVC(kernel="rbf", C=C, gamma=gamma).fit(Xtr, ytr)
            ax = axes[i, j]
            g.frontera_2d(ax, m, Xtr, ytr, titulo=f"γ = {gamma}   ·   C = {C}",
                          margenes=True, vectores_soporte=m.support_vectors_)
            ax.set_xlabel(f"VS = {m.n_support_.sum()}   ·   exactitud prueba = {m.score(Xte, yte):.2f}",
                          fontsize=8.5)
    fig.suptitle("SVM RBF: γ controla la curvatura de la frontera y C cuánto se castigan los errores",
                 fontsize=12, fontweight="bold", y=0.995)
    g.leyenda_frontera(fig, margenes=True, vectores=True)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    g.guardar(fig, "rbf_efecto_gamma_C")

    # Barrido fino para la tabla de sobreajuste/subajuste
    filas = []
    for gamma in np.logspace(-2, 2, 9):
        for C in np.logspace(-2, 3, 6):
            m = SVC(kernel="rbf", C=C, gamma=gamma).fit(Xtr, ytr)
            filas.append(dict(gamma=gamma, C=C, vectores_soporte=int(m.n_support_.sum()),
                              exactitud_entrenamiento=m.score(Xtr, ytr), exactitud_prueba=m.score(Xte, yte)))
    pd.DataFrame(filas).to_csv(DIR_RESULTADOS / "rbf_barrido_lunas.csv", index=False)

    # Modelo ajustado por validación cruzada -> frontera 2D y superficie 3D de f(x)
    busq = GridSearchCV(SVC(kernel="rbf"), {"C": np.logspace(-1, 3, 9), "gamma": np.logspace(-2, 2, 9)},
                        cv=5, n_jobs=-1).fit(Xtr, ytr)
    mejor = busq.best_estimator_
    fig = plt.figure(figsize=(12, 5.2))
    ax1 = fig.add_subplot(1, 2, 1)
    g.frontera_2d(ax1, mejor, Xtr, ytr, margenes=True, vectores_soporte=mejor.support_vectors_,
                  titulo=f"Frontera 2D (C = {mejor.C:.3g}, γ = {mejor.gamma:.3g})")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    g.superficie_3d(ax2, mejor, Xtr, ytr, titulo="Superficie f(x) = Σ αᵢ yᵢ K(xᵢ, x) + b")
    g.leyenda_frontera(fig, margenes=True, vectores=True)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    g.guardar(fig, "rbf_frontera_2d_3d")

    # Truco del kernel: una coordenada gaussiana vuelve separables los círculos concéntricos
    Xc, yc = circulos()
    z = np.exp(-1.0 * (Xc ** 2).sum(axis=1))  # φ(x) = exp(-γ||x - 0||²), γ = 1
    fig = plt.figure(figsize=(11, 4.8))
    ax1 = fig.add_subplot(1, 2, 1)
    g.dispersion_clases(ax1, Xc, yc, tam=18)
    ax1.set_aspect("equal")
    ax1.set_title("Espacio original: no hay recta separadora")
    ax1.set_xlabel("x₁")
    ax1.set_ylabel("x₂")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    for clase, color in ((0, CLASE_NEG), (1, CLASE_POS)):
        m = yc == clase
        ax2.scatter(Xc[m, 0], Xc[m, 1], z[m], c=color, s=12, depthshade=False)
    umbral = (z[yc == 0].max() + z[yc == 1].min()) / 2
    pp, qq = np.meshgrid(np.linspace(-1.2, 1.2, 2), np.linspace(-1.2, 1.2, 2))
    ax2.plot_surface(pp, qq, np.full_like(pp, umbral), color=TEXTO_2, alpha=0.18)
    ax2.set_xlabel("x₁")
    ax2.set_ylabel("x₂")
    ax2.set_zlabel("exp(−γ‖x‖²)")
    ax2.set_title("Con una coordenada RBF: un plano las separa")
    ax2.view_init(elev=18, azim=-60)
    fig.tight_layout()
    g.guardar(fig, "rbf_truco_kernel")

    # Verificación: dual resuelto desde cero vs. libsvm
    sub = slice(0, 150)
    propio = SVMDualDesdeCero(C=mejor.C, kernel="rbf", gamma=mejor.gamma).fit(Xtr[sub], ytr[sub])
    ref = SVC(kernel="rbf", C=mejor.C, gamma=mejor.gamma).fit(Xtr[sub], ytr[sub])
    verificacion = [
        dict(modelo="SVM RBF", comparacion="Vectores de soporte", propio=len(propio.indices_vs_),
             sklearn=int(ref.n_support_.sum())),
        dict(modelo="SVM RBF", comparacion="Sesgo b", propio=propio.b_, sklearn=float(ref.intercept_[0])),
        dict(modelo="SVM RBF", comparacion="Acuerdo de predicciones en prueba",
             propio=float((propio.predict(Xte) == ref.predict(Xte)).mean()), sklearn=1.0),
        dict(modelo="SVM RBF", comparacion="Brecha de dualidad (primal − dual)",
             propio=propio.objetivo_primal() - propio.objetivo_dual(), sklearn=0.0),
    ]
    return dict(mejor_C=float(mejor.C), mejor_gamma=float(mejor.gamma),
                exactitud_prueba=float(mejor.score(Xte, yte)),
                vectores_soporte=int(mejor.n_support_.sum())), verificacion


# ===========================================================================
# 2. SVM de margen suave (variables de holgura ξ_i)
# ===========================================================================
def experimento_margen_suave():
    print("[2/3] SVM de margen suave")
    X, y = nubes_solapadas()
    Xtr, Xte, ytr, yte = dividir_2d(X, y)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    for ax, C in zip(axes, [0.01, 1, 100]):
        m = SVC(kernel="linear", C=C).fit(Xtr, ytr)
        ancho, xi = ancho_y_holguras(m, Xtr, ytr)
        g.frontera_2d(ax, m, Xtr, ytr, titulo=f"C = {C}", margenes=True, vectores_soporte=m.support_vectors_)
        ax.set_xlabel(f"2/‖w‖ = {ancho:.2f}   ·   VS = {m.n_support_.sum()}   ·   Σξ = {xi.sum():.1f}   ·   "
                      f"ξ>1: {(xi > 1).sum()}", fontsize=8.5)
    fig.suptitle("Margen suave: C pequeño = margen ancho y muchas violaciones; C grande = margen estrecho",
                 fontsize=12, fontweight="bold")
    g.leyenda_frontera(fig, margenes=True, vectores=True)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    g.guardar(fig, "margen_suave_efecto_C")

    # Barrido de C: ancho del margen, vectores de soporte y exactitud (tres escalas -> tres paneles)
    filas = []
    for C in np.logspace(-3, 3, 25):
        m = SVC(kernel="linear", C=C).fit(Xtr, ytr)
        ancho, xi = ancho_y_holguras(m, Xtr, ytr)
        filas.append(dict(C=C, ancho_margen=ancho, vectores_soporte=int(m.n_support_.sum()), suma_xi=xi.sum(),
                          en_margen=int(((xi > 0) & (xi <= 1)).sum()), mal_clasificados=int((xi > 1).sum()),
                          exactitud_entrenamiento=m.score(Xtr, ytr), exactitud_prueba=m.score(Xte, yte)))
    barrido = pd.DataFrame(filas)
    barrido.to_csv(DIR_RESULTADOS / "margen_suave_barrido_C.csv", index=False)

    color = COLOR_MODELO["SVM Margen Suave"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].plot(barrido.C, barrido.ancho_margen, color=color, marker="o", markersize=4)
    axes[0].set_title("Ancho del margen 2/‖w‖")
    axes[1].plot(barrido.C, barrido.vectores_soporte, color=color, marker="o", markersize=4)
    axes[1].set_title("Número de vectores de soporte")
    axes[2].plot(barrido.C, barrido.exactitud_entrenamiento, color=color, ls="--", label="Entrenamiento")
    axes[2].plot(barrido.C, barrido.exactitud_prueba, color=color, label="Prueba")
    axes[2].set_title("Exactitud")
    axes[2].legend()
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("C (escala log)")
    fig.tight_layout()
    g.guardar(fig, "margen_suave_barrido_C")

    # Holguras individuales para C = 1
    m = SVC(kernel="linear", C=1).fit(Xtr, ytr)
    _, xi = ancho_y_holguras(m, Xtr, ytr)
    xi_pos = np.sort(xi[xi > 1e-6])[::-1]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    colores = np.where(xi_pos > 1, COLOR_MODELO["SVM Margen Suave"], COLOR_MODELO["SVM RBF"])
    ax.bar(np.arange(len(xi_pos)), xi_pos, color=colores, width=0.8, edgecolor=g.SUPERFICIE, linewidth=1)
    ax.axhline(1, color=TEXTO_2, lw=1, ls="--")
    ax.text(len(xi_pos) - 0.5, 1.04, "ξ = 1 (sobre la frontera)", ha="right", va="bottom", color=TEXTO_2, fontsize=8.5)
    ax.set_xlabel("Puntos de entrenamiento con ξᵢ > 0, ordenados")
    ax.set_ylabel("ξᵢ = max(0, 1 − yᵢ f(xᵢ))")
    ax.set_title(f"Variables de holgura con C = 1  ({len(xi_pos)} de {len(xi)} puntos violan el margen)")
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color=COLOR_MODELO["SVM RBF"], label="0 < ξ ≤ 1: dentro del margen, bien clasificado"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_MODELO["SVM Margen Suave"], label="ξ > 1: mal clasificado"),
    ], loc="upper right")
    fig.tight_layout()
    g.guardar(fig, "margen_suave_holguras")

    # Verificación: primal = dual y mismas holguras que libsvm
    propio = SVMDualDesdeCero(C=1.0, kernel="lineal").fit(Xtr, ytr)
    verificacion = [
        dict(modelo="SVM Margen Suave", comparacion="Vectores de soporte", propio=len(propio.indices_vs_),
             sklearn=int(m.n_support_.sum())),
        dict(modelo="SVM Margen Suave", comparacion="Ancho del margen 2/‖w‖", propio=propio.ancho_margen(),
             sklearn=float(2 / np.linalg.norm(m.coef_))),
        dict(modelo="SVM Margen Suave", comparacion="Σ ξᵢ", propio=float(propio.holguras().sum()), sklearn=float(xi.sum())),
        dict(modelo="SVM Margen Suave", comparacion="Brecha de dualidad (primal − dual)",
             propio=propio.objetivo_primal() - propio.objetivo_dual(), sklearn=0.0),
        dict(modelo="SVM Margen Suave", comparacion="Acuerdo de predicciones en prueba",
             propio=float((propio.predict(Xte) == m.predict(Xte)).mean()), sklearn=1.0),
    ]
    return dict(barrido=barrido.round(4).to_dict(orient="records")), verificacion


# ===========================================================================
# 3. Radius Neighbors / k-NN ponderado
# ===========================================================================
def experimento_vecinos():
    print("[3/3] Radius Neighbors / k-NN ponderado")
    X, y = densidad_variable()
    Xtr, Xte, ytr, yte = dividir_2d(X, y)
    buscador = NearestNeighbors().fit(Xtr)

    def zona_vacia(radio):
        return lambda Z: np.array([len(v) == 0 for v in buscador.radius_neighbors(Z, radius=radio, return_distance=False)])

    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    configs_knn = [(1, "uniform", "k = 1"), (15, "uniform", "k = 15, voto uniforme"),
                   (15, "distance", "k = 15, ponderado 1/d")]
    for ax, (k, pesos, titulo) in zip(axes[0], configs_knn):
        m = KNeighborsClassifier(n_neighbors=k, weights=pesos).fit(Xtr, ytr)
        g.frontera_2d(ax, m, Xtr, ytr, titulo=titulo, resolucion=200)
        ax.set_xlabel(f"exactitud prueba = {m.score(Xte, yte):.2f}", fontsize=8.5)
    for ax, r in zip(axes[1], [0.3, 0.7, 1.5]):
        m = RadiusNeighborsClassifier(radius=r, weights="distance", outlier_label="most_frequent").fit(Xtr, ytr)
        g.frontera_2d(ax, m, Xtr, ytr, titulo=f"Radio r = {r}, ponderado 1/d", zona_sin_vecinos=zona_vacia(r),
                      resolucion=200)
        sin_vecinos = zona_vacia(r)(Xte).mean()
        ax.set_xlabel(f"exactitud prueba = {m.score(Xte, yte):.2f}   ·   prueba sin vecinos = {sin_vecinos:.0%}",
                      fontsize=8.5)
    axes[0, 0].set_ylabel("k-NN (vecindario de k puntos)", fontsize=10)
    axes[1, 0].set_ylabel("Radius Neighbors (bola de radio r)", fontsize=10)
    fig.suptitle("k-NN fija cuántos vecinos votan; Radius Neighbors fija hasta qué distancia",
                 fontsize=12, fontweight="bold")
    g.leyenda_frontera(fig, sin_vecinos=True)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    g.guardar(fig, "vecinos_knn_vs_radio")

    # Barridos de k y de r
    filas_k = []
    for k in range(1, 61):
        for pesos in ("uniform", "distance"):
            m = KNeighborsClassifier(n_neighbors=k, weights=pesos).fit(Xtr, ytr)
            filas_k.append(dict(k=k, pesos=pesos, exactitud_prueba=m.score(Xte, yte)))
    filas_r = []
    for r in np.linspace(0.1, 2.5, 25):
        m = RadiusNeighborsClassifier(radius=r, weights="distance", outlier_label="most_frequent").fit(Xtr, ytr)
        filas_r.append(dict(radio=r, exactitud_prueba=m.score(Xte, yte), pct_sin_vecinos=zona_vacia(r)(Xte).mean()))
    bk, br = pd.DataFrame(filas_k), pd.DataFrame(filas_r)
    bk.to_csv(DIR_RESULTADOS / "vecinos_barrido_k.csv", index=False)
    br.to_csv(DIR_RESULTADOS / "vecinos_barrido_radio.csv", index=False)

    color = COLOR_MODELO["k-NN Ponderado"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for pesos, estilo, etiqueta in (("uniform", "--", "Voto uniforme"), ("distance", "-", "Ponderado 1/d")):
        d = bk[bk.pesos == pesos]
        axes[0].plot(d.k, d.exactitud_prueba, color=color, ls=estilo, label=etiqueta)
    axes[0].set_title("k-NN: exactitud en prueba vs. k")
    axes[0].set_xlabel("k")
    axes[0].legend()
    color_r = COLOR_MODELO["Radius Neighbors"]
    axes[1].plot(br.radio, br.exactitud_prueba, color=color_r, marker="o", markersize=4)
    axes[1].set_title("Radius Neighbors: exactitud vs. r")
    axes[1].set_xlabel("radio r")
    axes[2].plot(br.radio, br.pct_sin_vecinos * 100, color=color_r, marker="o", markersize=4)
    axes[2].set_title("% de puntos de prueba sin vecinos")
    axes[2].set_xlabel("radio r")
    fig.tight_layout()
    g.guardar(fig, "vecinos_barridos")

    # Verificación desde cero
    knn_propio = KNNPonderadoDesdeCero(k=15).fit(Xtr, ytr).predict(Xte)
    knn_ref = KNeighborsClassifier(15, weights="distance").fit(Xtr, ytr).predict(Xte)
    rad_propio = VecinosRadioDesdeCero(radio=0.7).fit(Xtr, ytr).predict(Xte)
    rad_ref = RadiusNeighborsClassifier(radius=0.7, weights="distance", outlier_label="most_frequent").fit(Xtr, ytr).predict(Xte)
    verificacion = [
        dict(modelo="k-NN Ponderado", comparacion="Acuerdo de predicciones en prueba (k = 15)",
             propio=float((knn_propio == knn_ref).mean()), sklearn=1.0),
        dict(modelo="Radius Neighbors", comparacion="Acuerdo de predicciones en prueba (r = 0.7)",
             propio=float((rad_propio == rad_ref).mean()), sklearn=1.0),
    ]
    mejor_k = bk.loc[bk.exactitud_prueba.idxmax()]
    return dict(mejor_k=int(mejor_k.k), pesos_mejor_k=mejor_k.pesos), verificacion


# ===========================================================================
# 4. Casos reales: búsqueda de hiperparámetros, métricas y comparativa
# ===========================================================================
def _evaluar(nombre, busqueda, Xtr, ytr, Xte, yte):
    t0 = time.perf_counter()
    busqueda.fit(Xtr, ytr)
    t_busqueda = time.perf_counter() - t0
    mejor = busqueda.best_estimator_
    t0 = time.perf_counter()
    pred = mejor.predict(Xte)
    t_pred = time.perf_counter() - t0
    s = puntajes(mejor, Xte)
    clf = mejor.named_steps["clf"]
    r = dict(
        modelo=nombre,
        accuracy=accuracy_score(yte, pred),
        precision=precision_score(yte, pred, zero_division=0),
        recall=recall_score(yte, pred),
        f1=f1_score(yte, pred),
        roc_auc=roc_auc_score(yte, s),
        auc_cv=busqueda.best_score_,
        combinaciones=len(busqueda.cv_results_["params"]),
        tiempo_busqueda_s=t_busqueda,
        tiempo_ajuste_s=busqueda.refit_time_,
        tiempo_prediccion_ms=1000 * t_pred,
        vectores_soporte=int(clf.n_support_.sum()) if hasattr(clf, "n_support_") else None,
        mejores_parametros={k.replace("clf__", ""): (round(float(v), 5) if isinstance(v, float) else v)
                            for k, v in busqueda.best_params_.items()},
    )
    # El umbral por defecto (f(x) = 0 o P = 0.5) no es óptimo con clases desbalanceadas:
    # se elige por validación cruzada, sólo con datos de entrenamiento, el que maximiza F1.
    ajustado = TunedThresholdClassifierCV(mejor, scoring="f1", cv=5).fit(Xtr, ytr)
    pred_aj = ajustado.predict(Xte)
    r.update(
        umbral_f1=float(ajustado.best_threshold_),
        precision_umbral=precision_score(yte, pred_aj, zero_division=0),
        recall_umbral=recall_score(yte, pred_aj),
        f1_umbral=f1_score(yte, pred_aj),
    )
    if isinstance(clf, RadiusNeighborsClassifier):
        Z = mejor.named_steps["prep"].transform(Xte)
        r["pct_prueba_sin_vecinos"] = float(np.mean([len(v) == 0 for v in clf.radius_neighbors(Z, return_distance=False)]))
    return r, dict(puntajes=s, matriz=confusion_matrix(yte, pred), mejor=mejor, busqueda=busqueda)


def _con_class_weight(cv, valor):
    """Máscara de filas de cv_results_ con ese class_weight (pandas convierte None en NaN en param_*)."""
    return cv["params"].apply(lambda p: p["clf__class_weight"] == valor)


def _sensibilidad(caso, extras):
    """AUC de validación cruzada frente al hiperparámetro principal de cada modelo."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)

    cv = pd.DataFrame(extras["SVM RBF"]["busqueda"].cv_results_)
    mejor_cw = extras["SVM RBF"]["busqueda"].best_params_["clf__class_weight"]
    cv = cv[_con_class_weight(cv, mejor_cw)]
    piv = cv.pivot_table(index="param_clf__gamma", columns="param_clf__C", values="mean_test_roc_auc")
    mejor_gamma = extras["SVM RBF"]["busqueda"].best_params_["clf__gamma"]
    axes[0].plot(piv.columns.astype(float), piv.loc[mejor_gamma], color=COLOR_MODELO["SVM RBF"], marker="o", markersize=4)
    axes[0].set_title(f"SVM RBF (γ = {mejor_gamma:.3g})")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("C")
    axes[0].set_ylabel("ROC-AUC de validación (CV 5)")

    cv = pd.DataFrame(extras["SVM Margen Suave"]["busqueda"].cv_results_)
    for cw, estilo in ((None, "-"), ("balanced", "--")):
        d = cv[_con_class_weight(cv, cw)].sort_values("param_clf__C")
        axes[1].plot(d.param_clf__C.astype(float), d.mean_test_roc_auc, color=COLOR_MODELO["SVM Margen Suave"],
                     ls=estilo, marker="o", markersize=4, label=f"class_weight = {cw}")
    axes[1].set_title("SVM Margen Suave")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("C")
    axes[1].legend()

    cv = pd.DataFrame(extras["k-NN Ponderado"]["busqueda"].cv_results_)
    for p, estilo, nombre in ((1, "--", "Manhattan (p = 1)"), (2, "-", "Euclídea (p = 2)")):
        d = cv[cv.param_clf__p == p].sort_values("param_clf__n_neighbors")
        axes[2].plot(d.param_clf__n_neighbors, d.mean_test_roc_auc, color=COLOR_MODELO["k-NN Ponderado"],
                     ls=estilo, label=nombre)
    axes[2].set_title("k-NN Ponderado")
    axes[2].set_xlabel("k")
    axes[2].legend()

    cv = pd.DataFrame(extras["Radius Neighbors"]["busqueda"].cv_results_)
    for w, marcador, nombre in (("uniform", "s", "Voto uniforme"), ("distance", "o", "Ponderado 1/d")):
        d = cv[cv.param_clf__weights == w].sort_values("param_clf__radius")
        axes[3].plot(d.param_clf__radius.astype(float), d.mean_test_roc_auc, color=COLOR_MODELO["Radius Neighbors"],
                     marker=marcador, markersize=5, ls="-" if w == "distance" else "--", label=nombre)
    axes[3].set_title("Radius Neighbors")
    axes[3].set_xlabel("radio r (espacio escalado)")
    axes[3].legend()
    fig.tight_layout()
    g.guardar(fig, f"{caso}_sensibilidad_hiperparametros")

    # Mapa de calor C x gamma del kernel RBF
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    g.mapa_calor(ax, piv.to_numpy(), [f"{c:.2g}" for c in piv.columns], [f"{v:.2g}" for v in piv.index],
                 f"SVM RBF: ROC-AUC de validación (class_weight = {mejor_cw})", "C", "γ")
    fig.tight_layout()
    g.guardar(fig, f"{caso}_rbf_mapa_calor")


def caso_real(caso, cargar, rapido):
    X, y, nombres_clase = cargar()
    print(f"\n=== Caso real: {caso} ({X.shape[0]} muestras, {X.shape[1]} variables, "
          f"{y.mean():.0%} clase positiva) ===")
    Xtr, Xte, ytr, yte = dividir(X, y)
    prep = construir_preprocesador(X)
    busquedas = construir_busquedas(prep, Xtr, rapido)
    dir_caso = DIR_RESULTADOS / caso
    dir_caso.mkdir(parents=True, exist_ok=True)

    resultados, extras = [], {}
    for nombre, busqueda in busquedas.items():
        print(f"  - {nombre} ...", end=" ", flush=True)
        r, extra = _evaluar(nombre, busqueda, Xtr, ytr, Xte, yte)
        print(f"AUC prueba = {r['roc_auc']:.3f} | F1 = {r['f1']:.3f} | {r['mejores_parametros']}")
        pd.DataFrame(busqueda.cv_results_).to_csv(dir_caso / f"cv_{nombre.replace(' ', '_').lower()}.csv", index=False)
        resultados.append(r)
        extras[nombre] = extra
    tabla = pd.DataFrame(resultados)
    tabla.to_csv(dir_caso / "metricas_prueba.csv", index=False)

    con_puntajes = [{**r, "puntajes": extras[r["modelo"]]["puntajes"], "matriz": extras[r["modelo"]]["matriz"]}
                    for r in resultados]

    # ROC
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    g.curvas_roc(ax, con_puntajes, yte)
    ax.set_title("Curvas ROC en el conjunto de prueba")
    fig.tight_layout()
    g.guardar(fig, f"{caso}_roc")

    # Matrices de confusión
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9))
    g.matrices_confusion(axes, con_puntajes, nombres_clase)
    fig.tight_layout()
    g.guardar(fig, f"{caso}_matrices_confusion")

    # Comparativa de métricas
    fig, ax = plt.subplots(figsize=(10, 4.4))
    g.barras_metricas(ax, tabla, ["accuracy", "precision", "recall", "f1", "roc_auc"])
    ax.set_title("Métricas en el conjunto de prueba")
    fig.tight_layout()
    g.guardar(fig, f"{caso}_comparativa_metricas")

    # Efecto de ajustar el umbral de decisión (gris = umbral por defecto, color = umbral ajustado)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    for ax, (antes, despues, titulo) in zip(axes, (("recall", "recall_umbral", "Recall"), ("f1", "f1_umbral", "F1"))):
        for i, r in enumerate(resultados):
            ax.plot([r[antes], r[despues]], [i, i], color=g.EJES, lw=2, zorder=1)
            ax.scatter(r[antes], i, s=60, color=g.SUPERFICIE, edgecolors=TEXTO_2, linewidths=1.5, zorder=2)
            ax.scatter(r[despues], i, s=70, color=COLOR_MODELO[r["modelo"]], edgecolors=g.SUPERFICIE,
                       linewidths=1, zorder=3)
            ax.text(max(r[antes], r[despues]) + 0.015, i, f"{r[antes]:.2f} → {r[despues]:.2f}",
                    va="center", fontsize=8.5, color=g.TEXTO)
        ax.set_title(f"{titulo} en prueba")
        ax.set_xlim(0, 1.12)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(resultados)), [r["modelo"] for r in resultados])
    axes[0].invert_yaxis()
    fig.legend(handles=[
        plt.Line2D([], [], marker="o", ls="", markerfacecolor=g.SUPERFICIE, markeredgecolor=TEXTO_2, label="Umbral por defecto"),
        plt.Line2D([], [], marker="o", ls="", color=TEXTO_2, label="Umbral ajustado (máx. F1 en CV)"),
    ], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    g.guardar(fig, f"{caso}_ajuste_umbral")

    # Curvas de aprendizaje
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.9), sharey=True)
    for ax, nombre in zip(axes, busquedas):
        tam, ent, val = learning_curve(extras[nombre]["mejor"], Xtr, ytr, cv=5, scoring="roc_auc",
                                       train_sizes=np.linspace(0.15, 1.0, 7), n_jobs=-1, shuffle=True,
                                       random_state=SEMILLA)
        g.curva_aprendizaje(ax, tam, ent, val, COLOR_MODELO[nombre], nombre)
    axes[0].set_ylabel("ROC-AUC")
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    g.guardar(fig, f"{caso}_curvas_aprendizaje")

    # Importancia por permutación (agnóstica al modelo: sirve también para RBF y vecinos)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    importancias = {}
    for ax, nombre in zip(axes.ravel(), busquedas):
        imp = permutation_importance(extras[nombre]["mejor"], Xte, yte, scoring="roc_auc", n_repeats=10,
                                     random_state=SEMILLA, n_jobs=-1)
        top = np.argsort(imp.importances_mean)[::-1][:10]
        g.barras_importancia(ax, [X.columns[i] for i in top], imp.importances_mean[top], imp.importances_std[top],
                             COLOR_MODELO[nombre], nombre)
        importancias[nombre] = {X.columns[i]: float(imp.importances_mean[i]) for i in top}
    fig.supxlabel("Caída media del ROC-AUC al permutar la variable (10 repeticiones)", fontsize=9.5, color=TEXTO_2)
    fig.tight_layout()
    g.guardar(fig, f"{caso}_importancia_permutacion")

    # Validación cruzada repetida (5 x 5) para comparar los modelos con su variabilidad.
    # Sólo con entrenamiento: el conjunto de prueba queda intacto para la evaluación final.
    rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEMILLA)
    puntos_cv = {n: cross_val_score(extras[n]["mejor"], Xtr, ytr, cv=rcv, scoring="roc_auc", n_jobs=-1)
                 for n in busquedas}
    pd.DataFrame(puntos_cv).to_csv(dir_caso / "cv_repetida_auc.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    rng = np.random.default_rng(SEMILLA)
    for i, (n, v) in enumerate(puntos_cv.items()):
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=22, color=COLOR_MODELO[n], alpha=0.75,
                   edgecolors=g.SUPERFICIE, linewidths=0.6)
        ax.plot([i - 0.28, i + 0.28], [v.mean()] * 2, color=g.TEXTO, lw=2)
        ax.text(i + 0.31, v.mean(), f"{v.mean():.3f} ± {v.std():.3f}", va="center", fontsize=8.5, color=g.TEXTO)
    ax.set_xticks(range(len(puntos_cv)), list(puntos_cv))
    ax.set_xlim(-0.5, len(puntos_cv) - 0.2)
    ax.set_ylabel("ROC-AUC por pliegue")
    ax.set_title("Validación cruzada repetida 5 × 5 (línea = media)")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    g.guardar(fig, f"{caso}_cv_repetida")

    _sensibilidad(caso, extras)

    resumen = tabla.to_dict(orient="records")
    for r in resumen:
        r["auc_cv_repetida_media"] = float(puntos_cv[r["modelo"]].mean())
        r["auc_cv_repetida_desv"] = float(puntos_cv[r["modelo"]].std())
        r["importancia_top10"] = importancias[r["modelo"]]
    return dict(muestras=int(X.shape[0]), variables=int(X.shape[1]), proporcion_positiva=float(y.mean()),
                clases=nombres_clase, modelos=resumen)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rapido", action="store_true", help="rejillas de hiperparámetros reducidas")
    parser.add_argument("--sin-credito", action="store_true", help="omite el caso German Credit (requiere internet la 1.ª vez)")
    args = parser.parse_args()

    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    g.aplicar_estilo()
    inicio = time.perf_counter()

    resumen, verificacion = {}, []
    for clave, experimento in (("rbf_2d", experimento_rbf), ("margen_suave_2d", experimento_margen_suave),
                               ("vecinos_2d", experimento_vecinos)):
        res, ver = experimento()
        resumen[clave] = res
        verificacion += ver
    pd.DataFrame(verificacion).to_csv(DIR_RESULTADOS / "verificacion_desde_cero.csv", index=False)
    resumen["verificacion_desde_cero"] = verificacion

    resumen["cancer"] = caso_real("cancer", cargar_cancer, args.rapido)
    if not args.sin_credito:
        try:
            resumen["credito"] = caso_real("credito", cargar_credito, args.rapido)
        except Exception as e:  # sin conexión la primera vez
            print(f"\n[aviso] No se pudo cargar German Credit ({e}). Se omite ese caso.")

    with open(DIR_RESULTADOS / "resumen.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nListo en {time.perf_counter() - inicio:.0f} s. Resultados en {DIR_RESULTADOS}")


if __name__ == "__main__":
    main()
