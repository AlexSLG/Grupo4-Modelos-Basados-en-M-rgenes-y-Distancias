"""Pipelines y espacios de búsqueda de hiperparámetros de los tres modelos."""
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import uniform
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors, RadiusNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from .config import PLIEGUES_CV, SEMILLA

METRICAS_CV = {
    "roc_auc": "roc_auc",
    "f1": "f1",
    "recall": "recall",
    "precision": "precision",
    "accuracy": "accuracy",
}
# El modelo final de cada búsqueda se elige por ROC-AUC: no depende del umbral de
# decisión y es comparable entre la función de decisión de la SVM y las
# probabilidades de los vecinos.
METRICA_REFIT = "roc_auc"


def _validacion_cruzada():
    return StratifiedKFold(n_splits=PLIEGUES_CV, shuffle=True, random_state=SEMILLA)


def rango_radios(preprocesador, X_entrenamiento, k_bajo=5, cuantil_alto=0.9):
    """Rango razonable de radios en el espacio ya escalado.

    El radio útil depende de la dimensión y la escala, así que se deriva de los datos:
    va de la mediana de la distancia al 5.º vecino hasta el percentil 90 de todas
    las distancias entre pares. En alta dimensión las distancias se concentran
    (maldición de la dimensionalidad) y el radio óptimo suele quedar lejos de la
    distancia a los vecinos más cercanos.
    """
    Z = clone(preprocesador).fit_transform(X_entrenamiento)
    dist, _ = NearestNeighbors(n_neighbors=k_bajo + 1).fit(Z).kneighbors(Z)
    return float(np.median(dist[:, k_bajo])), float(np.quantile(pdist(Z), cuantil_alto))


def construir_busquedas(preprocesador, X_entrenamiento, rapido=False):
    """Devuelve {nombre: búsqueda} con un Pipeline(preprocesador -> clasificador) en cada una."""
    cv = _validacion_cruzada()
    comunes = dict(scoring=METRICAS_CV, refit=METRICA_REFIT, cv=cv, n_jobs=-1, return_train_score=True)
    n_rejilla = 4 if rapido else 6

    def tuberia(clf):
        return Pipeline([("prep", clone(preprocesador)), ("clf", clf)])

    # 1. SVM con kernel RBF: C controla la penalización de errores; gamma el alcance del kernel.
    svm_rbf = GridSearchCV(
        tuberia(SVC(kernel="rbf", random_state=SEMILLA)),
        {
            "clf__C": np.logspace(-2, 3, n_rejilla),
            "clf__gamma": np.logspace(-4, 1, n_rejilla),
            "clf__class_weight": [None, "balanced"],
        },
        **comunes,
    )

    # 2. SVM de margen suave (lineal): C es el único hiperparámetro del primal.
    svm_suave = GridSearchCV(
        tuberia(SVC(kernel="linear", random_state=SEMILLA)),
        {
            "clf__C": np.logspace(-3, 2, n_rejilla),
            "clf__class_weight": [None, "balanced"],
        },
        **comunes,
    )

    # 3a. k-NN ponderado por distancia: k y la métrica de Minkowski (p=1 Manhattan, p=2 Euclídea).
    knn = GridSearchCV(
        tuberia(KNeighborsClassifier(weights="distance")),
        {
            "clf__n_neighbors": list(range(1, 42, 4 if rapido else 2)),
            "clf__p": [1, 2],
        },
        **comunes,
    )

    # 3b. Radius Neighbors: radio continuo -> búsqueda aleatoria sobre un rango derivado de los datos.
    r_min, r_max = rango_radios(preprocesador, X_entrenamiento)
    radio = RandomizedSearchCV(
        tuberia(RadiusNeighborsClassifier(outlier_label="most_frequent")),
        {
            "clf__radius": uniform(loc=r_min, scale=r_max - r_min),
            "clf__weights": ["uniform", "distance"],
        },
        n_iter=12 if rapido else 40,
        random_state=SEMILLA,
        **comunes,
    )

    return {
        "SVM RBF": svm_rbf,
        "SVM Margen Suave": svm_suave,
        "k-NN Ponderado": knn,
        "Radius Neighbors": radio,
    }


def puntajes(modelo, X):
    """Puntaje continuo para curvas ROC: f(x) en las SVM, P(y=1|x) en los vecinos."""
    if hasattr(modelo, "decision_function"):
        return modelo.decision_function(X)
    return modelo.predict_proba(X)[:, 1]
