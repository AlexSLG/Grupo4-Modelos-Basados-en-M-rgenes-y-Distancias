"""Gráficas: fronteras de decisión 2D/3D, curvas ROC y de aprendizaje, matrices, importancias."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve

from .config import (
    CLASE_NEG, CLASE_POS, COLOR_MODELO, DIR_FIGURAS, EJES, NEUTRO, RAMPA_AZUL,
    REJILLA, SUPERFICIE, TEXTO, TEXTO_2,
)

# Divergente azul <-> gris <-> rojo para f(x): el signo es la clase, la intensidad la confianza
MAPA_DIVERGENTE = LinearSegmentedColormap.from_list(
    "polaridad", ["#86b6ef", "#cde2fb", NEUTRO, "#f6c9c8", "#ee9190"]
)
MAPA_SECUENCIAL = LinearSegmentedColormap.from_list("azul", RAMPA_AZUL)


def aplicar_estilo():
    plt.rcParams.update({
        "figure.facecolor": SUPERFICIE,
        "axes.facecolor": SUPERFICIE,
        "savefig.facecolor": SUPERFICIE,
        "axes.edgecolor": EJES,
        "axes.labelcolor": TEXTO_2,
        "axes.titlecolor": TEXTO,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": REJILLA,
        "grid.linewidth": 0.8,
        "xtick.color": TEXTO_2,
        "ytick.color": TEXTO_2,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "text.color": TEXTO,
        "font.size": 9.5,
        "lines.linewidth": 2,
        "hatch.color": EJES,
        "hatch.linewidth": 0.7,
    })


def guardar(fig, nombre):
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    ruta = DIR_FIGURAS / f"{nombre}.png"
    fig.savefig(ruta, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return ruta


# ---------------------------------------------------------------------------
# Fronteras de decisión
# ---------------------------------------------------------------------------
def malla(X, resolucion=250, holgura=0.6):
    x0, x1 = X[:, 0].min() - holgura, X[:, 0].max() + holgura
    y0, y1 = X[:, 1].min() - holgura, X[:, 1].max() + holgura
    xx, yy = np.meshgrid(np.linspace(x0, x1, resolucion), np.linspace(y0, y1, resolucion))
    return xx, yy, np.c_[xx.ravel(), yy.ravel()]


def _puntaje_malla(modelo, Z):
    """(valores centrados en 0, es_svm). SVM -> f(x); vecinos -> P(y=1|x) - 0.5."""
    if hasattr(modelo, "decision_function"):
        return modelo.decision_function(Z), True
    return modelo.predict_proba(Z)[:, 1] - 0.5, False


def dispersion_clases(ax, X, y, tam=26):
    for clase, color in ((0, CLASE_NEG), (1, CLASE_POS)):
        m = y == clase
        ax.scatter(X[m, 0], X[m, 1], s=tam, c=color, edgecolors=SUPERFICIE, linewidths=0.9, zorder=3)


def frontera_2d(ax, modelo, X, y, titulo=None, margenes=False, vectores_soporte=None,
                zona_sin_vecinos=None, resolucion=250):
    """Dibuja f(x) como fondo divergente, la frontera f(x)=0 y, opcionalmente, los márgenes ±1."""
    xx, yy, Z = malla(X, resolucion)
    f, es_svm = _puntaje_malla(modelo, Z)
    limite = 2.0 if es_svm else 0.5
    F = np.clip(f, -limite, limite).reshape(xx.shape)
    ax.contourf(xx, yy, F, levels=np.linspace(-limite, limite, 17), cmap=MAPA_DIVERGENTE, zorder=0)
    ax.contour(xx, yy, F, levels=[0], colors=TEXTO, linewidths=1.8, zorder=2)
    if margenes and es_svm:
        ax.contour(xx, yy, F, levels=[-1, 1], colors=TEXTO_2, linewidths=1.1, linestyles="--", zorder=2)
    if zona_sin_vecinos is not None:
        vacia = zona_sin_vecinos(Z).reshape(xx.shape).astype(float)
        ax.contourf(xx, yy, vacia, levels=[0.5, 1.5], colors="none", hatches=["////"], zorder=1)
        ax.contour(xx, yy, vacia, levels=[0.5], colors=TEXTO_2, linewidths=0.6, zorder=1)
    dispersion_clases(ax, X, y)
    if vectores_soporte is not None and len(vectores_soporte):
        ax.scatter(vectores_soporte[:, 0], vectores_soporte[:, 1], s=95, facecolors="none",
                   edgecolors=TEXTO, linewidths=1.1, zorder=4)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    if titulo:
        ax.set_title(titulo, fontsize=10)


def leyenda_frontera(fig, margenes=False, vectores=False, sin_vecinos=False, extra=None):
    elementos = [
        Line2D([], [], marker="o", ls="", color=CLASE_NEG, markersize=7, label="Clase 0 (−1)"),
        Line2D([], [], marker="o", ls="", color=CLASE_POS, markersize=7, label="Clase 1 (+1)"),
        Line2D([], [], color=TEXTO, lw=1.8, label="Frontera de decisión"),
    ]
    if margenes:
        elementos.append(Line2D([], [], color=TEXTO_2, lw=1.1, ls="--", label="Márgenes f(x) = ±1"))
    if vectores:
        elementos.append(Line2D([], [], marker="o", ls="", markerfacecolor="none",
                                markeredgecolor=TEXTO, markersize=9, label="Vector de soporte"))
    if sin_vecinos:
        elementos.append(plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=TEXTO_2,
                                       hatch="////", label="Sin vecinos en el radio"))
    elementos += extra or []
    fig.legend(handles=elementos, loc="lower center", ncol=len(elementos),
               bbox_to_anchor=(0.5, -0.02), handlelength=1.6)


def superficie_3d(ax, modelo, X, y, titulo=None):
    """f(x) como superficie: la frontera es el corte de la superficie con el plano z = 0."""
    xx, yy, Z = malla(X, resolucion=90)
    F = modelo.decision_function(Z).reshape(xx.shape)
    ax.plot_surface(xx, yy, F, cmap=MAPA_DIVERGENTE, vmin=-2, vmax=2, alpha=0.85,
                    linewidth=0, antialiased=True)
    ax.contour(xx, yy, F, levels=[0], colors=TEXTO, linewidths=2, offset=F.min())
    fx = modelo.decision_function(X)
    for clase, color in ((0, CLASE_NEG), (1, CLASE_POS)):
        m = y == clase
        ax.scatter(X[m, 0], X[m, 1], fx[m], c=color, s=12, depthshade=False,
                   edgecolors=SUPERFICIE, linewidths=0.5)
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_zlabel("f(x)")
    ax.view_init(elev=28, azim=-58)
    if titulo:
        ax.set_title(titulo)


# ---------------------------------------------------------------------------
# Evaluación en los casos reales
# ---------------------------------------------------------------------------
def curvas_roc(ax, resultados, y_prueba):
    ax.plot([0, 1], [0, 1], color=EJES, lw=1, ls="--")
    for r in resultados:
        fpr, tpr, _ = roc_curve(y_prueba, r["puntajes"])
        ax.plot(fpr, tpr, color=COLOR_MODELO[r["modelo"]],
                label=f"{r['modelo']} (AUC = {r['roc_auc']:.3f})")
    ax.set_xlabel("Tasa de falsos positivos (1 − especificidad)")
    ax.set_ylabel("Tasa de verdaderos positivos (recall)")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.legend(loc="lower right")


def matrices_confusion(axes, resultados, nombres_clase):
    for ax, r in zip(axes, resultados):
        M = r["matriz"]
        ax.imshow(M, cmap=MAPA_SECUENCIAL, vmin=0, vmax=M.sum(axis=1).max())
        for i in range(2):
            for j in range(2):
                oscuro = M[i, j] > 0.55 * M.sum(axis=1).max()
                ax.text(j, i, f"{M[i, j]}", ha="center", va="center", fontsize=13,
                        fontweight="bold", color=SUPERFICIE if oscuro else TEXTO)
        ax.set_xticks([0, 1], [f"Pred. {n}" for n in nombres_clase], fontsize=8)
        ax.set_yticks([0, 1], [f"Real {n}" for n in nombres_clase], fontsize=8, rotation=90, va="center")
        ax.grid(False)
        ax.set_title(r["modelo"], fontsize=10)
        for s in ax.spines.values():
            s.set_visible(False)


def curva_aprendizaje(ax, tamanos, entrenamiento, validacion, color, titulo):
    for puntos, estilo, etiqueta in ((entrenamiento, "--", "Entrenamiento"), (validacion, "-", "Validación (CV)")):
        media, desv = puntos.mean(axis=1), puntos.std(axis=1)
        ax.plot(tamanos, media, ls=estilo, color=color, marker="o", markersize=4, label=etiqueta)
        ax.fill_between(tamanos, media - desv, media + desv, color=color, alpha=0.15, lw=0)
    ax.set_title(titulo, fontsize=10)
    ax.set_xlabel("Muestras de entrenamiento")


def barras_importancia(ax, nombres, medias, desv, color, titulo):
    orden = np.argsort(medias)
    ax.barh(np.arange(len(orden)), medias[orden], xerr=desv[orden], color=color, height=0.7,
            error_kw=dict(ecolor=TEXTO_2, lw=0.8, capsize=0))
    ax.set_yticks(np.arange(len(orden)), [nombres[i] for i in orden], fontsize=8)
    ax.axvline(0, color=EJES, lw=1)
    ax.set_title(titulo, fontsize=10)
    ax.grid(axis="y", visible=False)


def barras_metricas(ax, tabla, metricas):
    """Barras agrupadas: una agrupación por métrica, una barra por modelo (eje y único 0-1)."""
    modelos = tabla["modelo"].tolist()
    n = len(modelos)
    ancho = 0.8 / n
    x = np.arange(len(metricas))
    for i, m in enumerate(modelos):
        valores = tabla.loc[tabla["modelo"] == m, metricas].to_numpy().ravel()
        ax.bar(x + (i - (n - 1) / 2) * ancho, valores, width=ancho, color=COLOR_MODELO[m],
               edgecolor=SUPERFICIE, linewidth=1.5, label=m)
    ax.set_xticks(x, [m.replace("_", "-").upper() if m == "roc_auc" else m.capitalize() for m in metricas])
    ax.set_ylim(0, 1.05)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", ncol=n, bbox_to_anchor=(0.5, -0.08))


def mapa_calor(ax, matriz, etiquetas_x, etiquetas_y, titulo, etiqueta_x, etiqueta_y):
    im = ax.imshow(matriz, cmap=MAPA_SECUENCIAL, origin="lower", aspect="auto")
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            v = matriz[i, j]
            umbral = np.nanmin(matriz) + 0.6 * (np.nanmax(matriz) - np.nanmin(matriz))
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7.5,
                    color=SUPERFICIE if v > umbral else TEXTO)
    ax.set_xticks(range(len(etiquetas_x)), etiquetas_x)
    ax.set_yticks(range(len(etiquetas_y)), etiquetas_y)
    ax.set_xlabel(etiqueta_x)
    ax.set_ylabel(etiqueta_y)
    ax.set_title(titulo)
    ax.grid(False)
    return im
