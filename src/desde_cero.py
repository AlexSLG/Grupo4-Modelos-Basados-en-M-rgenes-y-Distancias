"""Implementaciones desde cero (NumPy/SciPy) de los tres modelos.

No se usan para el caso real (ahí se usa scikit-learn, que es más rápido y robusto);
sirven para comprobar numéricamente que las ecuaciones del informe técnico
reproducen el comportamiento de las implementaciones de referencia.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
def kernel_lineal(A, B):
    """K(x, z) = x·z"""
    return A @ B.T


def kernel_rbf(A, B, gamma):
    """K(x, z) = exp(-gamma * ||x - z||²)"""
    return np.exp(-gamma * cdist(A, B, "sqeuclidean"))


# ---------------------------------------------------------------------------
# SVM de margen suave resuelta por su problema dual (programación cuadrática)
# ---------------------------------------------------------------------------
class SVMDualDesdeCero:
    """SVM de margen suave con kernel lineal o RBF.

    Primal:  min_{w,b,ξ}  ½||w||² + C Σ ξ_i
             s.a.  y_i (w·φ(x_i) + b) >= 1 - ξ_i,   ξ_i >= 0

    Dual:    max_α  Σ α_i - ½ Σ_i Σ_j α_i α_j y_i y_j K(x_i, x_j)
             s.a.  0 <= α_i <= C,   Σ α_i y_i = 0

    Se resuelve el dual con SLSQP (adecuado para n de unos cientos de puntos).
    """

    def __init__(self, C=1.0, kernel="rbf", gamma=1.0, tol=1e-5):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.tol = tol

    def _K(self, A, B):
        return kernel_rbf(A, B, self.gamma) if self.kernel == "rbf" else kernel_lineal(A, B)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        s = np.where(np.asarray(y) == 1, 1.0, -1.0)  # etiquetas {0,1} -> {-1,+1}
        n = len(s)
        Q = np.outer(s, s) * self._K(X, X)

        # SciPy minimiza: usamos el negativo del dual
        def objetivo(a):
            return 0.5 * a @ Q @ a - a.sum()

        def gradiente(a):
            return Q @ a - 1.0

        sol = minimize(
            objetivo, np.zeros(n), jac=gradiente, method="SLSQP",
            bounds=[(0.0, self.C)] * n,
            constraints=[{"type": "eq", "fun": lambda a: a @ s, "jac": lambda a: s}],
            options={"maxiter": 1000, "ftol": 1e-10},
        )
        alfa = np.clip(sol.x, 0.0, self.C)
        alfa[alfa < self.tol * self.C] = 0.0

        self.alfa_ = alfa
        self.indices_vs_ = np.flatnonzero(alfa > 0)
        self.X_vs_ = X[self.indices_vs_]
        self.coef_dual_ = alfa[self.indices_vs_] * s[self.indices_vs_]  # α_i y_i
        self.convergio_ = bool(sol.success)

        # b se obtiene de los vectores de soporte "libres" (0 < α_i < C), que cumplen
        # y_i f(x_i) = 1 exactamente (KKT); se promedia para estabilidad numérica.
        libres = np.flatnonzero((alfa > self.tol * self.C) & (alfa < self.C * (1 - self.tol)))
        ref = libres if len(libres) else self.indices_vs_
        f_sin_b = self._K(X[ref], self.X_vs_) @ self.coef_dual_
        self.b_ = float(np.mean(s[ref] - f_sin_b))

        self._Q = Q
        self._s = s
        self._X = X
        return self

    def decision_function(self, X):
        return self._K(np.asarray(X, dtype=float), self.X_vs_) @ self.coef_dual_ + self.b_

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)

    def holguras(self, X=None, y=None):
        """ξ_i = max(0, 1 - y_i f(x_i)). 0 < ξ <= 1: dentro del margen; ξ > 1: mal clasificado."""
        if X is None:
            X, s = self._X, self._s
        else:
            s = np.where(np.asarray(y) == 1, 1.0, -1.0)
        return np.maximum(0.0, 1.0 - s * self.decision_function(X))

    def norma_w2(self):
        """||w||² = Σ_i Σ_j α_i α_j y_i y_j K(x_i, x_j) = αᵀQα"""
        return float(self.alfa_ @ self._Q @ self.alfa_)

    def objetivo_primal(self):
        return 0.5 * self.norma_w2() + self.C * self.holguras().sum()

    def objetivo_dual(self):
        return float(self.alfa_.sum() - 0.5 * self.norma_w2())

    def ancho_margen(self):
        """Distancia entre los hiperplanos f(x) = +1 y f(x) = -1: 2 / ||w||."""
        return 2.0 / np.sqrt(self.norma_w2())


# ---------------------------------------------------------------------------
# Vecinos: k-NN ponderado y Radius Neighbors
# ---------------------------------------------------------------------------
def _votos_ponderados(dist, etiquetas, clases, ponderado):
    """Suma de votos por clase para un vecindario.

    Ponderación inversa a la distancia, w_i = 1/d_i. Si algún vecino está a
    distancia 0, sólo votan los vecinos idénticos (mismo criterio que scikit-learn).
    """
    if ponderado:
        ceros = dist == 0
        pesos = ceros.astype(float) if ceros.any() else 1.0 / dist
    else:
        pesos = np.ones_like(dist)
    return np.array([pesos[etiquetas == c].sum() for c in clases])


class KNNPonderadoDesdeCero:
    """ŷ(x) = argmax_c Σ_{i ∈ N_k(x)} w_i · 1[y_i = c],  con w_i = 1 / d(x, x_i)."""

    def __init__(self, k=5, p=2, ponderado=True):
        self.k = k
        self.p = p
        self.ponderado = ponderado

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y)
        self.clases_ = np.unique(self.y_)
        return self

    def predict_proba(self, X):
        D = cdist(np.asarray(X, dtype=float), self.X_, "minkowski", p=self.p)
        vecinos = np.argsort(D, axis=1, kind="stable")[:, : self.k]
        prob = np.empty((len(D), len(self.clases_)))
        for i, idx in enumerate(vecinos):
            votos = _votos_ponderados(D[i, idx], self.y_[idx], self.clases_, self.ponderado)
            prob[i] = votos / votos.sum()
        return prob

    def predict(self, X):
        return self.clases_[np.argmax(self.predict_proba(X), axis=1)]


class VecinosRadioDesdeCero:
    """ŷ(x) = argmax_c Σ_{i : d(x, x_i) <= r} w_i · 1[y_i = c].

    Si la bola de radio r no contiene ningún punto de entrenamiento, el punto se
    considera atípico y recibe ``etiqueta_atipica`` (por defecto, la clase más frecuente).
    """

    def __init__(self, radio=1.0, p=2, ponderado=True, etiqueta_atipica=None):
        self.radio = radio
        self.p = p
        self.ponderado = ponderado
        self.etiqueta_atipica = etiqueta_atipica

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y)
        self.clases_, conteos = np.unique(self.y_, return_counts=True)
        if self.etiqueta_atipica is None:
            self.etiqueta_atipica_ = self.clases_[np.argmax(conteos)]
        else:
            self.etiqueta_atipica_ = self.etiqueta_atipica
        return self

    def predict(self, X):
        D = cdist(np.asarray(X, dtype=float), self.X_, "minkowski", p=self.p)
        pred = np.empty(len(D), dtype=self.y_.dtype)
        self.sin_vecinos_ = np.zeros(len(D), dtype=bool)
        for i, fila in enumerate(D):
            idx = np.flatnonzero(fila <= self.radio)
            if len(idx) == 0:
                pred[i] = self.etiqueta_atipica_
                self.sin_vecinos_[i] = True
                continue
            votos = _votos_ponderados(fila[idx], self.y_[idx], self.clases_, self.ponderado)
            pred[i] = self.clases_[np.argmax(votos)]
        return pred
