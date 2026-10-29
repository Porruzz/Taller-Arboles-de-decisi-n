"""
Synthetic Diamond Dataset module.
Provides the training dataset (12 instances) and unlabelled test instances (N1-N4)
as defined in the workshop case study.
"""

import pandas as pd
from typing import Tuple

def get_diamond_dataset() -> pd.DataFrame:
    """
    Returns the synthetic diamond dataset with 12 observations.
    
    Columns:
        ID (int): Unique identifier (not to be used as predictor).
        Quilates (float): Diamond weight in carats (continuous feature x1).
        Profundidad (float): Diamond depth percentage (continuous feature x2).
        Clase (str): Cut classification ('A' = Aceptable, 'E' = Excelente).
    """
    data = [
        (1, 0.30, 60.0, "A"),
        (2, 0.40, 61.0, "A"),
        (3, 0.50, 64.0, "A"),
        (4, 0.60, 62.0, "E"),
        (5, 0.70, 65.0, "A"),
        (6, 0.80, 63.0, "E"),
        (7, 0.90, 61.0, "E"),
        (8, 1.00, 64.0, "A"),
        (9, 1.10, 62.0, "E"),
        (10, 1.20, 65.0, "A"),
        (11, 1.30, 60.0, "E"),
        (12, 1.40, 63.0, "E")
    ]
    df = pd.DataFrame(data, columns=["ID", "Quilates", "Profundidad", "Clase"])
    return df

def get_new_cases() -> pd.DataFrame:
    """
    Returns the 4 new unlabelled cases (N1-N4) for manual and automated inference.
    """
    cases = [
        ("N1", 0.45, 62.0),
        ("N2", 1.15, 64.0),
        ("N3", 0.85, 61.0),
        ("N4", 0.55, 63.5)
    ]
    df = pd.DataFrame(cases, columns=["Caso", "Quilates", "Profundidad"])
    return df

def get_problem_formulation() -> dict:
    """
    Provides the formal structural definitions of the problem (Section 1).
    """
    return {
        "unidad_observacion": "Un diamante individual presentado para clasificación gemológica de corte.",
        "caracteristicas_X": ["Quilates (peso en quilates, x1)", "Profundidad (% de profundidad respecto al diámetro, x2)"],
        "etiqueta_y": "Clase de corte del diamante (variable cualitativa binaria)",
        "clases_posibles": ["A (Aceptable)", "E (Excelente)"],
        "clase_positiva": "A (Aceptable)",
        "justificacion_aprendizaje": (
            "Es aprendizaje supervisado porque cada observación del conjunto de entrenamiento "
            "cuenta con sus atributos predictores X y su respectiva etiqueta de verdad terreno (ground truth) "
            "y conocida previamente. Es de clasificación porque la variable dependiente y es cualitativa/categórica "
            "y discreta (A o E), a diferencia de un problema de regresión donde el objetivo es un valor numérico continuo."
        ),
        "conceptos_arbol": {
            "raiz": "Nodo inicial (profundidad 0) que contiene el 100% de los datos de entrenamiento (12 diamantes). "
                    "Allí se evalúa la primera pregunta divisoria óptima.",
            "nodo_interno": "Nodo intermedio de decisión que evalúa una condición sobre un predictor (ej. Quilates <= 0.50) "
                            "para particionar aún más un subconjunto impuro de diamantes.",
            "rama": "Enlace lógico y direccional que conecta un nodo con sus descendientes según el cumplimiento (x <= t) "
                    "o incumplimiento (x > t) de la condición evaluada.",
            "hoja": "Nodo terminal que carece de hijos. Representa la decisión final del modelo y asigna una etiqueta de clase "
                    "(A o E) por mayoría o por la regla de desempate establecida."
        },
        "analisis_fuga_datos": (
            "El campo 'resultado final de la inspección' NO debe incluirse. Esto constituye fuga de datos (data leakage), "
            "dado que es una variable generada con posterioridad o simultáneamente al juicio de clasificación. "
            "En el momento de producción/inferencia, este campo no estará disponible de antemano. Incluirlo falsearía "
            "la capacidad predictiva del modelo en el mundo real."
        ),
        "condicion_precio": (
            "Para poder incluir el 'precio', debe asegurarse estrictamente que sea una característica disponible "
            "ANTES de clasificar el corte (ej. costo de extracción o compra bruta previa), y que bajo ninguna circunstancia "
            "sea un precio de venta determinado a partir del corte ya asignado, pues de lo contrario crearía una fuga circular."
        )
    }
