"""
Metrics Evaluation module (Section 4).
Implements formulas, numerical substitutions, interpretations, and business cost analysis
for binary classification evaluation.
"""

from typing import Dict, Any

class BinaryEvaluationMetrics:
    """
    Computes and formats binary classification metrics based on confusion matrix values.
    """
    
    def __init__(self, vp: int = 6, vn: int = 9, fp: int = 3, fn: int = 2, pos_label: str = "A", neg_label: str = "E"):
        self.vp = vp
        self.vn = vn
        self.fp = fp
        self.fn = fn
        self.pos_label = pos_label
        self.neg_label = neg_label
        self.n = vp + vn + fp + fn
        
    def calculate_all(self) -> Dict[str, Any]:
        """
        Calculates Accuracy, Precision, Recall, and F1 with step-by-step substitution.
        """
        acc = (self.vp + self.vn) / self.n
        prec = self.vp / (self.vp + self.fp) if (self.vp + self.fp) > 0 else 0.0
        rec = self.vp / (self.vp + self.fn) if (self.vp + self.fn) > 0 else 0.0
        f1 = (2 * self.vp) / (2 * self.vp + self.fp + self.fn) if (2 * self.vp + self.fp + self.fn) > 0 else 0.0
        
        return {
            "VP": self.vp,
            "VN": self.vn,
            "FP": self.fp,
            "FN": self.fn,
            "N": self.n,
            "Exactitud": {
                "valor": round(acc, 4),
                "porcentaje": f"{acc * 100:.2f}%",
                "formula": "Exactitud = (VP + VN) / N",
                "sustitucion": f"({self.vp} + {self.vn}) / {self.n} = {self.vp + self.vn} / {self.n}",
                "interpretacion": (
                    f"El {acc*100:.2f}% de las predicciones globales emitidas por el modelo fueron acertadas "
                    f"({self.vp + self.vn} de {self.n} casos clasificados correctamente entre ambas clases)."
                )
            },
            "Precision": {
                "valor": round(prec, 4),
                "porcentaje": f"{prec * 100:.2f}%",
                "formula": "Precision = VP / (VP + FP)",
                "sustitucion": f"{self.vp} / ({self.vp} + {self.fp}) = {self.vp} / {self.vp + self.fp}",
                "interpretacion": (
                    f"De todos los diamantes que el clasificador etiquetó como Aceptable (clase positiva A), "
                    f"únicamente el {prec*100:.2f}% (6 de 9) eran verdaderamente Aceptables; el 33.33% restante eran falsas alarmas."
                )
            },
            "Recall": {
                "valor": round(rec, 4),
                "porcentaje": f"{rec * 100:.2f}%",
                "formula": "Recall = VP / (VP + FN)",
                "sustitucion": f"{self.vp} / ({self.vp} + {self.fn}) = {self.vp} / {self.vp + self.fn}",
                "interpretacion": (
                    f"De la totalidad de diamantes verdaderamente Aceptables (8 casos reales en el conjunto), "
                    f"el modelo fue capaz de detectar y recuperar exitosamente el {rec*100:.2f}% (6 casos)."
                )
            },
            "F1": {
                "valor": round(f1, 4),
                "porcentaje": f"{f1 * 100:.2f}%",
                "formula": "F1 = 2*VP / (2*VP + FP + FN)",
                "sustitucion": f"(2 * {self.vp}) / (2*{self.vp} + {self.fp} + {self.fn}) = {2*self.vp} / {2*self.vp + self.fp + self.fn}",
                "interpretacion": (
                    f"La media armónica entre precisión y exhaustividad para la clase A es {round(f1, 4)} ({f1*100:.2f}%), "
                    f"mostrando un balance equilibrado pero penalizado por las 3 falsas alarmas y 2 omisiones."
                )
            }
        }

def analyze_error_costs() -> Dict[str, Any]:
    """
    Detailed explanation of asymmetric error costs (Section 4b).
    """
    return {
        "error_tipo_1": {
            "nombre": "Clasificar un diamante Excelente (E) como Aceptable (A)",
            "naturaleza": "Falso Positivo (FP) respecto a la clase A / Degradación injustificada de calidad",
            "impacto": "Subestima una gema de alta gama rebajándola a estándar.",
            "escenario_mas_costoso": (
                "Escenario de Comercialización Mayorista / Venta de Inventario:\n"
                "Si la empresa comercializa las gemas con base en esta clasificación automática, vender un lote de diamantes "
                "Excelentes al precio depreciado de diamantes Aceptables causará una severa pérdida financiera directa "
                "(lucro cesante inmediato por vender un activo premium a precio de descuento)."
            )
        },
        "error_tipo_2": {
            "nombre": "Clasificar un diamante Aceptable (A) como Excelente (E)",
            "naturaleza": "Falso Negativo (FN) respecto a la clase A / Falsa sobreestimación de calidad a Excelente",
            "impacto": "Sobreestima una piedra con corte común o imperfecciones como si fuera grado supremo.",
            "escenario_mas_costoso": (
                "Escenario de Laboratorio Certificador de Lujo / Joyería de Alta Gama:\n"
                "Si un laboratorio de prestigio expide certificados de autenticidad y califica un diamante defectuoso como "
                "'Excelente', el comprador final o un perito auditor externo detectará el engaño. Las consecuencias serán "
                "demandas por publicidad engañosa, demandas comerciales, revocación de licencias y pérdida total de reputación institucional."
            )
        }
    }

def analyze_imbalanced_dataset() -> Dict[str, Any]:
    """
    Analysis of the 95 A / 5 E dataset with a majority-class predictor (Section 4c).
    """
    n = 100
    n_a = 95
    n_e = 5
    # Naive model always predicts A:
    # All 95 A are predicted as A (VP_A = 95, FN_A = 0)
    # All 5 E are predicted as A (FP_A = 5, VN_A = 0)
    acc = 95 / 100
    # For class E:
    # Real E = 5, Predicted E = 0 -> VP_E = 0, FN_E = 5
    recall_e = 0 / 5
    
    return {
        "tamano_muestra": n,
        "clase_A": n_a,
        "clase_E": n_e,
        "modelo": "Predictor trivial mayoritario (Siempre predice 'A')",
        "exactitud": {
            "valor": round(acc, 4),
            "porcentaje": "95.00%",
            "formula": "Exactitud = 95 / 100 = 0.9500"
        },
        "recall_E": {
            "valor": round(recall_e, 4),
            "porcentaje": "0.00%",
            "formula": "Recall(E) = VP_E / (VP_E + FN_E) = 0 / 5 = 0.0000"
        },
        "justificacion": (
            "Este resultado ilustra la 'paradoja de la exactitud' en conjuntos desbalanceados. Una exactitud del 95% "
            "aparenta un rendimiento excelente a nivel superficial, pero oculta que el modelo es 100% inútil para la clase "
            "minoritaria E (que con frecuencia es la clase de mayor interés o criticidad). "
            "Por esta razón es imprescindible evaluar métricas desagregadas por clase (Recall, Precisión, F1 por clase) "
            "o métricas macro-ponderadas (como Macro F1), las cuales revelan inmediatamente el colapso del clasificador."
        )
    }
