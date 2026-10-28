"""Textos de ayuda de la app: descripciones de controles, métricas, gráficas, pestañas y variables.

Están aquí, y no en app_simulacion.py, para que la lógica de la app quede limpia.
Todos están escritos para alguien que no conoce machine learning.
"""

AVISO_EDUCATIVO = "Herramienta educativa: no sirve para diagnóstico médico real."

# ---------------------------------------------------------------------------
# Pestañas
# ---------------------------------------------------------------------------
INTRO_PESTANAS = {
    "Simulación": (
        "Aquí los modelos aprenden en vivo sobre puntos en un plano. Cada punto es un ejemplo y su color "
        "es su clase (azul = clase 0, rojo = clase 1). El modelo debe aprender a separar los colores. "
        "Abre **Parámetros de la simulación** para cambiar los datos o el modelo, y mira cómo cambian la "
        "frontera y las métricas."
    ),
    "Fundamentos": (
        "Las fórmulas que hay detrás de cada modelo: qué función usa para decidir la clase y qué "
        "optimiza al entrenar. Es el resumen matemático del informe."
    ),
    "Casos reales": (
        "Resultados de los cuatro modelos sobre dos problemas reales: detectar tumores malignos y "
        "detectar clientes que no pagarán un crédito. Los hiperparámetros se eligieron con validación "
        "cruzada sobre el 75 % de los datos, y las métricas se miden en el 25 % restante, que el "
        "modelo nunca vio. Se generan con `python experimentos.py`."
    ),
    "Probar un caso": (
        "Ingresa los datos de un paciente o de un solicitante de crédito y mira qué predice cada modelo. "
        "Los cuatro modelos ya están entrenados con los mejores hiperparámetros de la pestaña "
        "**Casos reales**. Puedes partir de un caso real del conjunto de prueba, llenar el formulario "
        "a mano o subir un archivo CSV con varios casos."
    ),
}

# ---------------------------------------------------------------------------
# Controles de la simulación
# ---------------------------------------------------------------------------
AYUDA_CONTROLES = {
    "conjunto": (
        "Forma de los datos de prueba:\n"
        "- **Lunas:** dos medias lunas entrelazadas. Una recta no las separa; luce la SVM RBF.\n"
        "- **Círculos concéntricos:** un anillo rodea al otro. Caso clásico del kernel RBF.\n"
        "- **Nubes solapadas:** dos nubes que se mezclan en el centro. Una recta separa casi todo, "
        "pero siempre hay errores; luce la SVM de margen suave.\n"
        "- **Densidad variable + atípicos:** zonas muy densas, zonas vacías y puntos sueltos. "
        "Muestra la diferencia entre k-NN y Radius Neighbors."
    ),
    "n": (
        "Cantidad de puntos que se generan (70 % para entrenar y 30 % para medir). Con más puntos la "
        "frontera es más estable, pero todo tarda más. No aplica al conjunto de densidad variable, "
        "que tiene un tamaño fijo."
    ),
    "ruido": (
        "Cuánto se mezclan las dos clases. Con poco ruido las clases están bien separadas; con mucho "
        "ruido se invaden y ningún modelo puede acertar todo. En el conjunto de densidad variable "
        "controla la cantidad de puntos atípicos."
    ),
    "semilla": (
        "Número que fija el azar. Con la misma semilla se generan los mismos puntos y la misma "
        "división 70/30 entre entrenamiento y prueba. Cámbiala para probar con una muestra nueva."
    ),
    "modelo": (
        "- **SVM RBF:** frontera curva que deja el mayor margen posible entre las clases.\n"
        "- **SVM Margen Suave:** frontera recta que tolera algunos errores (holguras ξᵢ).\n"
        "- **k-NN Ponderado:** vota entre los k vecinos más cercanos; los más cercanos pesan más.\n"
        "- **Radius Neighbors:** vota entre todos los vecinos que están dentro de un radio."
    ),
    "C": (
        "Penalización de cada error o violación del margen (ξᵢ). Con C pequeño el margen es ancho y "
        "tolerante; con C grande el margen es estrecho y hay riesgo de sobreajuste."
    ),
    "gamma": (
        "Alcance de la influencia de cada vector de soporte. Con γ pequeño la frontera es suave, casi "
        "recta; con γ grande la frontera rodea cada punto (sobreajuste)."
    ),
    "desde_cero": (
        "Usa nuestra propia implementación de la SVM en lugar de la de scikit-learn. Resuelve el problema "
        "dual con **SLSQP**, un método general de optimización con restricciones de SciPy. Aparece la "
        "**brecha de dualidad**: la diferencia entre el valor del problema primal y del dual; si es "
        "casi 0, se encontró el óptimo. Es lento con más de ~300 puntos."
    ),
    "k": (
        "Cuántos vecinos votan. Con k pequeño la frontera sigue cada punto y se deja engañar por el "
        "ruido (sobreajuste); con k grande la frontera es suave, pero puede borrar detalles reales."
    ),
    "radio": (
        "Distancia máxima para que un punto cuente como vecino. Con radio pequeño muchos puntos no "
        "tienen vecinos (zona rayada) y reciben la clase más frecuente; con radio grande votan casi "
        "todos y la frontera se vuelve muy suave."
    ),
    "pesos": (
        "Cómo vota cada vecino. **1/d (ponderado):** el voto pesa más cuanto más cerca está el vecino. "
        "**Uniforme:** todos los vecinos valen lo mismo."
    ),
    "p": (
        "Cómo se mide la distancia. **Euclídea (p = 2):** en línea recta, la distancia de siempre. "
        "**Manhattan (p = 1):** sumando lo que se avanza en cada eje, como en las calles de una "
        "cuadrícula."
    ),
    "x1": "Primera coordenada (eje horizontal) del punto que quieres clasificar.",
    "x2": "Segunda coordenada (eje vertical) del punto que quieres clasificar.",
}

# ---------------------------------------------------------------------------
# Métricas de la simulación (clave = etiqueta de la tarjeta)
# ---------------------------------------------------------------------------
AYUDA_METRICAS = {
    "Exactitud entren.": (
        "Porcentaje de puntos de entrenamiento bien clasificados: los que el modelo usó para aprender. "
        "Compárala con la exactitud de prueba: si la de entrenamiento es mucho más alta, el modelo "
        "memorizó en lugar de aprender (sobreajuste)."
    ),
    "Exactitud prueba": (
        "Porcentaje de puntos de prueba bien clasificados. Son puntos que el modelo nunca vio, así que "
        "mide lo que pasaría con datos nuevos. Es la cifra que importa."
    ),
    "F1 prueba": (
        "Combina dos preguntas sobre la clase 1 (roja): ¿de los que marcó como rojos, cuántos lo eran? "
        "(precisión) y ¿de los rojos reales, cuántos encontró? (recall). Va de 0 a 1; 1 es perfecto."
    ),
    "ROC-AUC prueba": (
        "Probabilidad de que el modelo le dé más puntaje a un punto rojo que a uno azul elegidos al "
        "azar. 0.5 es adivinar al azar y 1 es ordenar perfecto. No depende del umbral de decisión."
    ),
    "Vectores de soporte": (
        "Puntos que definen la frontera de la SVM: los que están sobre el margen, dentro de él o mal "
        "clasificados. El resto de puntos podría borrarse sin cambiar el modelo. Muchos vectores "
        "suelen indicar clases muy mezcladas o un C pequeño."
    ),
    "Σ ξᵢ": (
        "Suma de todas las holguras: el total de \"multas\" que paga la SVM por los puntos que violan el "
        "margen o están mal clasificados. Al subir C, el modelo se esfuerza en bajarla."
    ),
    "Dentro del margen": (
        "Puntos con 0 < ξᵢ ≤ 1: están bien clasificados pero dentro del pasillo entre las líneas "
        "punteadas, así que violan el margen."
    ),
    "Mal clasificados": "Puntos con ξᵢ > 1: quedaron del lado equivocado de la frontera.",
    "Ancho del margen 2/‖w‖": (
        "Ancho del pasillo entre las dos líneas punteadas. La SVM busca un pasillo ancho porque así "
        "generaliza mejor. Con C pequeño el pasillo se ensancha."
    ),
    "Brecha de dualidad": (
        "Primal − dual. La SVM se puede plantear de dos formas equivalentes (primal y dual). En el "
        "óptimo ambas dan el mismo valor, así que una brecha ≈ 0 indica que nuestro solucionador "
        "encontró la solución (dualidad fuerte)."
    ),
    "Prueba sin vecinos": (
        "Porcentaje de puntos de prueba que no tienen ningún vecino dentro del radio. No hay nadie que "
        "vote, así que reciben la clase más frecuente del entrenamiento. Si es alto, el radio es "
        "demasiado pequeño."
    ),
}

# ---------------------------------------------------------------------------
# Leyendas de las gráficas
# ---------------------------------------------------------------------------
LEYENDA_FRONTERA = (
    "Cada punto es un ejemplo de entrenamiento, coloreado por su clase real. El fondo es lo que el "
    "modelo predice en cada zona del plano: azul para la clase 0 y rojo para la clase 1, más intenso "
    "cuanto más seguro está. La línea negra sólida es la frontera de decisión, donde el modelo cambia "
    "de opinión."
)
LEYENDA_FRONTERA_SVM = (
    " Las líneas punteadas son los márgenes f(x) = ±1, los bordes del pasillo. Los puntos con anillo "
    "son los vectores de soporte, los únicos que definen la frontera."
)
LEYENDA_FRONTERA_RADIO = (
    " La zona rayada no tiene ningún punto de entrenamiento dentro del radio: ahí el modelo asigna la "
    "clase más frecuente."
)
LEYENDA_ESTRELLA = " La estrella es el punto nuevo que ingresaste."

LEYENDA_HOLGURAS = (
    "Una barra por cada punto de entrenamiento con ξᵢ > 0, ordenadas de mayor a menor. Las azules "
    "están bien clasificadas pero dentro del margen; las naranjas, por encima de la línea punteada "
    "(ξᵢ = 1), están mal clasificadas. Los puntos sin barra (ξᵢ = 0) están fuera del pasillo y bien "
    "clasificados."
)
LEYENDA_3D = (
    "La altura es el valor de f(x) en cada punto del plano: las montañas son zonas de la clase 1 y "
    "los valles, de la clase 0. La frontera de decisión es donde la superficie corta la altura 0. "
    "Así se ve cómo el kernel RBF construye la frontera sumando campanas alrededor de los vectores "
    "de soporte."
)

LEYENDAS_CASOS = {
    "roc": (
        "Cada curva muestra, para todos los umbrales posibles, cuántos positivos detecta el modelo "
        "(eje vertical) a cambio de cuántas falsas alarmas (eje horizontal). Cuanto más pegada a la "
        "esquina superior izquierda, mejor. El área bajo la curva es el ROC-AUC; la diagonal es "
        "adivinar al azar."
    ),
    "comparativa_metricas": (
        "Las métricas principales de los cuatro modelos sobre el conjunto de prueba, con el umbral por "
        "defecto. Barras más altas son mejores. Compara sobre todo el recall: es el porcentaje de "
        "casos positivos que el modelo encuentra."
    ),
    "ajuste_umbral": (
        "Cada modelo decide \"positivo\" cuando su puntaje supera un umbral. El punto gris es el "
        "resultado con el umbral por defecto y el punto de color, con el umbral ajustado para maximizar "
        "F1 (elegido con validación cruzada sobre entrenamiento). Una línea larga indica que el umbral "
        "por defecto estaba mal puesto."
    ),
    "matrices_confusion": (
        "Una tabla por modelo. Las filas son la clase real y las columnas, la predicha. La diagonal son "
        "los aciertos; fuera de ella, los errores: falsos negativos (positivos que se escaparon) y "
        "falsos positivos (falsas alarmas)."
    ),
    "curvas_aprendizaje": (
        "Cómo cambia el rendimiento al entrenar con más datos. Si las dos curvas (entrenamiento y "
        "validación) quedan juntas y altas, el modelo generaliza bien. Si hay una brecha grande, "
        "sobreajusta y le vendrían bien más datos o más regularización."
    ),
    "sensibilidad_hiperparametros": (
        "Cómo cambia el ROC-AUC de validación cruzada al mover cada hiperparámetro. Una curva plana "
        "indica un modelo robusto; un pico estrecho indica que hay que afinar con cuidado."
    ),
    "cv_repetida": (
        "ROC-AUC en 25 particiones distintas del entrenamiento (validación cruzada 5 × 5): cada punto es "
        "una partición y la línea negra, la media. Si las nubes de puntos de dos modelos se solapan "
        "mucho, su diferencia puede deberse al azar de la partición."
    ),
    "importancia_permutacion": (
        "Cuánto empeora cada modelo si se desordena al azar una variable. Las variables con barras "
        "largas son las que el modelo más usa para decidir. Funciona igual con cualquier modelo."
    ),
}

# ---------------------------------------------------------------------------
# Columnas de la tabla de métricas de los casos reales
# ---------------------------------------------------------------------------
COLUMNAS_TABLA = {
    "modelo": "Nombre del modelo.",
    "accuracy": "Exactitud: porcentaje de casos de prueba bien clasificados.",
    "precision": "Precisión: de los casos que el modelo marcó como positivos, cuántos lo eran de verdad.",
    "recall": "Recall (sensibilidad): de los positivos reales, cuántos encontró el modelo.",
    "f1": "F1: equilibrio entre precisión y recall (media armónica). 1 es perfecto.",
    "roc_auc": "ROC-AUC en prueba: qué tan bien ordena los casos positivos por encima de los negativos.",
    "auc_cv": "ROC-AUC promedio de la validación cruzada de 5 pliegues con los mejores hiperparámetros.",
    "recall_umbral": "Recall en prueba usando el umbral ajustado para maximizar F1, en lugar del umbral por defecto.",
    "f1_umbral": "F1 en prueba usando el umbral ajustado.",
    "vectores_soporte": "Número de vectores de soporte de la SVM (no aplica a los vecinos).",
    "tiempo_busqueda_s": "Segundos que tardó la búsqueda de hiperparámetros con validación cruzada.",
    "mejores_parametros": "Hiperparámetros elegidos por la búsqueda.",
}

# ---------------------------------------------------------------------------
# Pestaña "Probar un caso"
# ---------------------------------------------------------------------------
NOMBRES_CASOS = {
    "cancer": "Diagnóstico de cáncer de mama",
    "credito": "Riesgo crediticio (German Credit)",
}

MODOS_ENTRADA = {
    "prueba": "Elegir un caso del conjunto de prueba",
    "manual": "Llenar el formulario a mano",
    "csv": "Subir un archivo CSV",
}

LEYENDA_CONFIANZA = (
    "**Cómo leer la confianza.** En las SVM es f(x), la distancia con signo a la frontera: positiva "
    "significa «{positiva}», negativa «{negativa}», y |f(x)| ≥ 1 significa que el caso queda fuera del "
    "margen, con mucha seguridad. En los vecinos es la probabilidad de «{positiva}», la proporción "
    "ponderada de vecinos de esa clase. La columna *Seguridad* lleva ambas a una escala de 0 a 100 %: "
    "en las SVM es |f(x)| hasta 1, y en los vecinos es lo lejos que está la probabilidad de 0.5."
)
AVISO_UMBRAL = (
    "La clase positiva es **{positiva}**. La columna *Predicción* usa el umbral por defecto (f(x) ≥ 0 en "
    "las SVM, probabilidad ≥ 0.5 en los vecinos). La columna *Con umbral ajustado* usa el umbral que "
    "maximiza F1 en validación cruzada, guardado por `experimentos.py`."
)
LEYENDA_GRAFICA_CONFIANZA = (
    "La barra sale del umbral por defecto (línea negra) hacia la clase que predice el modelo: hacia la "
    "derecha y en rojo es «{positiva}», hacia la izquierda y en azul es «{negativa}». La marca punteada "
    "es el umbral ajustado: si la barra no la cruza, la predicción con ese umbral es «{negativa}»."
)

# ---------------------------------------------------------------------------
# Variables del caso de cáncer de mama (Breast Cancer Wisconsin)
# ---------------------------------------------------------------------------
_CANCER_BASE = {
    "radius": ("Radio", "Distancia promedio del centro del núcleo celular a su borde."),
    "texture": ("Textura", "Variación de los tonos de gris dentro del núcleo."),
    "perimeter": ("Perímetro", "Longitud del contorno del núcleo."),
    "area": ("Área", "Superficie del núcleo, en píxeles."),
    "smoothness": ("Suavidad", "Variación local del radio: qué tan liso o irregular es el borde."),
    "compactness": ("Compacidad", "Perímetro² / área − 1: qué tanto se aleja la forma de un círculo."),
    "concavity": ("Concavidad", "Profundidad de las partes hundidas (cóncavas) del contorno."),
    "concave points": ("Puntos cóncavos", "Cantidad de partes hundidas del contorno."),
    "symmetry": ("Simetría", "Qué tan simétrico es el núcleo."),
    "fractal dimension": ("Dimensión fractal", "Qué tan rugoso y complejo es el borde."),
}
GRUPOS_CANCER = {
    "Media": ("mean {}", "Promedio de esta medida sobre todos los núcleos de la imagen."),
    "Error estándar": ("{} error", "Error estándar: cuánto varía esta medida entre los núcleos de la imagen."),
    "Peor valor": ("worst {}", "Peor valor: promedio de los 3 núcleos con los valores más altos (los más anómalos)."),
}
NOTA_CANCER = (
    "Cada variable describe los núcleos de las células en una imagen de una biopsia por aspiración con "
    "aguja fina. Los núcleos grandes, irregulares y con muchas concavidades suelen indicar un tumor maligno."
)


def variables_cancer():
    """{grupo: [(columna, etiqueta, ayuda), ...]} con las 30 variables agrupadas en media / error / peor."""
    return {
        grupo: [(plantilla.format(base), etiqueta, f"{ayuda} {ayuda_grupo}")
                for base, (etiqueta, ayuda) in _CANCER_BASE.items()]
        for grupo, (plantilla, ayuda_grupo) in GRUPOS_CANCER.items()
    }


# ---------------------------------------------------------------------------
# Variables del caso de crédito (German Credit). Montos en marcos alemanes (DM).
# ---------------------------------------------------------------------------
VARIABLES_CREDITO = {
    "checking_status": ("Cuenta corriente", "Saldo de la cuenta corriente del solicitante en el banco."),
    "duration": ("Duración (meses)", "Plazo del crédito en meses. Los plazos largos suelen asociarse a más riesgo."),
    "credit_history": ("Historial crediticio", "Cómo ha pagado el solicitante sus créditos anteriores."),
    "purpose": ("Propósito", "Para qué se pide el crédito."),
    "credit_amount": ("Monto (DM)", "Cantidad de dinero solicitada, en marcos alemanes."),
    "savings_status": ("Ahorros", "Dinero en cuentas de ahorro o bonos."),
    "employment": ("Antigüedad laboral", "Años en el empleo actual."),
    "installment_commitment": (
        "Cuota / ingreso (1–4)",
        "Peso de la cuota mensual sobre el ingreso disponible: 1 = menos del 20 %, 4 = 35 % o más.",
    ),
    "personal_status": ("Sexo y estado civil", "Sexo y estado civil del solicitante."),
    "other_parties": ("Otros deudores o avalistas", "Si alguien más firma o garantiza el crédito."),
    "residence_since": ("Años en la vivienda (1–4)", "Tiempo en la vivienda actual, en 4 tramos: 1 = menos de 1 año, 4 = 4 años o más."),
    "property_magnitude": ("Bienes", "El bien más valioso del solicitante."),
    "age": ("Edad (años)", "Edad del solicitante."),
    "other_payment_plans": ("Otros planes de pago", "Si tiene otros créditos a plazos fuera de este banco."),
    "housing": ("Vivienda", "Si la vivienda es propia, alquilada o gratuita."),
    "existing_credits": ("Créditos vigentes", "Cuántos créditos tiene ya en este banco."),
    "job": ("Tipo de empleo", "Nivel de cualificación del trabajo."),
    "num_dependents": ("Personas a cargo", "Personas que dependen económicamente del solicitante."),
    "own_telephone": ("Teléfono propio", "Si tiene un teléfono registrado a su nombre."),
    "foreign_worker": ("Trabajador extranjero", "Si el solicitante es un trabajador extranjero."),
}
GRUPOS_CREDITO = {
    "El crédito solicitado": ["duration", "credit_amount", "purpose", "installment_commitment"],
    "Historial y finanzas": ["checking_status", "savings_status", "credit_history", "existing_credits",
                             "other_payment_plans", "other_parties"],
    "Empleo y vivienda": ["employment", "job", "housing", "residence_since", "property_magnitude"],
    "Perfil personal": ["age", "personal_status", "num_dependents", "own_telephone", "foreign_worker"],
}
NOTA_CREDITO = (
    "Cada caso es una persona que pidió un crédito a un banco alemán en los años 90. Los montos están "
    "en marcos alemanes (DM). El modelo estima si será un mal pagador."
)

CATEGORIAS_CREDITO = {
    "checking_status": {"<0": "Saldo negativo (< 0 DM)", "0<=X<200": "Entre 0 y 200 DM",
                        ">=200": "200 DM o más", "no checking": "Sin cuenta corriente"},
    "credit_history": {"no credits/all paid": "Sin créditos previos o todos pagados",
                       "all paid": "Pagó todos sus créditos en este banco",
                       "existing paid": "Créditos actuales al día",
                       "delayed previously": "Se retrasó en pagos anteriores",
                       "critical/other existing credit": "Cuenta crítica o créditos en otros bancos"},
    "purpose": {"new car": "Auto nuevo", "used car": "Auto usado", "furniture/equipment": "Muebles o equipamiento",
                "radio/tv": "Radio o televisión", "domestic appliance": "Electrodomésticos",
                "repairs": "Reparaciones", "education": "Educación", "retraining": "Capacitación",
                "business": "Negocio", "other": "Otro"},
    "savings_status": {"<100": "Menos de 100 DM", "100<=X<500": "Entre 100 y 500 DM",
                       "500<=X<1000": "Entre 500 y 1000 DM", ">=1000": "1000 DM o más",
                       "no known savings": "Sin ahorros conocidos"},
    "employment": {"unemployed": "Desempleado", "<1": "Menos de 1 año", "1<=X<4": "Entre 1 y 4 años",
                   "4<=X<7": "Entre 4 y 7 años", ">=7": "7 años o más"},
    "personal_status": {"male single": "Hombre soltero", "male mar/wid": "Hombre casado o viudo",
                        "male div/sep": "Hombre divorciado o separado",
                        "female div/dep/mar": "Mujer divorciada, separada o casada"},
    "other_parties": {"none": "Ninguno", "co applicant": "Cosolicitante", "guarantor": "Avalista"},
    "property_magnitude": {"real estate": "Inmueble", "life insurance": "Seguro de vida o ahorro",
                           "car": "Auto u otro bien", "no known property": "Sin bienes conocidos"},
    "other_payment_plans": {"none": "Ninguno", "bank": "Con otro banco", "stores": "Con tiendas"},
    "housing": {"own": "Propia", "rent": "Alquilada", "for free": "Gratuita"},
    "job": {"unemp/unskilled non res": "Desempleado o no cualificado (no residente)",
            "unskilled resident": "No cualificado (residente)", "skilled": "Cualificado",
            "high qualif/self emp/mgmt": "Muy cualificado, autónomo o directivo"},
    "own_telephone": {"none": "No", "yes": "Sí"},
    "foreign_worker": {"no": "No", "yes": "Sí"},
}


def nombre_categoria(columna, valor):
    """Traducción al español de una categoría de German Credit (el valor original si no hay traducción)."""
    return CATEGORIAS_CREDITO.get(columna, {}).get(valor, valor)
