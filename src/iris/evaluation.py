"""
Iris Final Test Evaluation and Baseline Benchmark module (Section 5B - d, e).
Evaluates the winning model strictly once on the Test set,
calculates per-class metrics, and benchmarks against DummyClassifier(strategy="most_frequent").
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, Any, Tuple
from .dataset import IrisDataLoader

class IrisTestEvaluator:
    """
    Performs the single, unbiased evaluation of the chosen model on the Test set,
    alongside the DummyClassifier baseline.
    """
    
    def __init__(self, best_model: DecisionTreeClassifier, data_loader: IrisDataLoader = None):
        self.model = best_model
        self.loader = data_loader or IrisDataLoader()
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = self.loader.get_partitions()
        self.target_names = self.loader.target_names
        
    def evaluate_test_set(self) -> Dict[str, Any]:
        """
        Computes the final confusion matrix, overall accuracy, per-class metrics,
        and macro F1 on the Test set.
        """
        y_pred = self.model.predict(self.X_test)
        
        cm = confusion_matrix(self.y_test, y_pred)
        acc = accuracy_score(self.y_test, y_pred)
        f1_macro = f1_score(self.y_test, y_pred, average="macro", zero_division=0)
        
        precisions = precision_score(self.y_test, y_pred, average=None, zero_division=0)
        recalls = recall_score(self.y_test, y_pred, average=None, zero_division=0)
        f1s = f1_score(self.y_test, y_pred, average=None, zero_division=0)
        
        per_class_metrics = []
        for i, name in enumerate(self.target_names):
            per_class_metrics.append({
                "Clase": name,
                "Precisión": round(precisions[i], 4),
                "Exhaustividad (Recall)": round(recalls[i], 4),
                "F1-Score": round(f1s[i], 4),
                "Soporte": int((self.y_test == i).sum())
            })
            
        metrics_df = pd.DataFrame(per_class_metrics)
        report_str = classification_report(self.y_test, y_pred, target_names=self.target_names, digits=4, zero_division=0)
        
        return {
            "confusion_matrix": cm,
            "accuracy": round(acc, 4),
            "f1_macro": round(f1_macro, 4),
            "per_class_df": metrics_df,
            "classification_report": report_str,
            "y_pred": y_pred
        }

    def benchmark_against_dummy(self) -> pd.DataFrame:
        """
        Trains DummyClassifier(strategy="most_frequent") on X_train and evaluates on X_test.
        Returns a comparative summary table between Selected Tree vs Dummy Baseline.
        """
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(self.X_train, self.y_train)
        y_dummy_pred = dummy.predict(self.X_test)
        
        dummy_acc = accuracy_score(self.y_test, y_dummy_pred)
        dummy_f1_macro = f1_score(self.y_test, y_dummy_pred, average="macro", zero_division=0)
        
        test_eval = self.evaluate_test_set()
        
        records = [
            {
                "Modelo": "Árbol Seleccionado (max_depth=None)",
                "Rol": "Modelo Ganador del Proceso de Selección",
                "Exactitud Prueba": round(test_eval["accuracy"], 4),
                "F1 Macro Prueba": round(test_eval["f1_macro"], 4)
            },
            {
                "Modelo": "DummyClassifier (most_frequent)",
                "Rol": "Línea Base Ingenua (Referencia)",
                "Exactitud Prueba": round(dummy_acc, 4),
                "F1 Macro Prueba": round(dummy_f1_macro, 4)
            }
        ]
        
        return pd.DataFrame(records)

    @staticmethod
    def get_test_limitations_discussion() -> Dict[str, str]:
        """
        Scientific conclusion discussing what the experiment demonstrates and its statistical limitations.
        """
        return {
            "lo_que_demuestra": (
                "El experimento demuestra que un árbol de decisión es altamente efectivo para clasificar especies de Iris, "
                "superando drásticamente a la referencia ingenua (93.33% vs 33.33% de exactitud; 0.9333 vs 0.1667 F1 macro). "
                "Evidencia la separabilidad perfecta de Setosa y una excelente capacidad para distinguir entre Versicolor y Virginica "
                "usando únicamente medidas morfométricas del pétalo."
            ),
            "lo_que_limita_tamano_reducido": (
                "El conjunto de prueba consta de únicamente 30 observaciones (10 por clase). En un conjunto tan pequeño, "
                "cada flor individual mal clasificada representa una variación de 3.33% en la exactitud global y 10.0% en el recall "
                "de esa clase específica. Por ende, los intervalos de confianza estadística son amplios y no debe asumirse que "
                "el 93.33% sea exacto ante perturbaciones mayores de muestreo o datos recolectados en diferentes estaciones o regiones geográficas."
            ),
            "regla_metodologica_no_cambiar_seleccion": (
                "Es un principio inquebrantable de la ciencia de datos no modificar la selección de hiperparámetros ni reentrenar "
                "el modelo tras observar el rendimiento en el conjunto de prueba. Hacerlo contaminaría la evaluación final y "
                "anularía la condición de conjunto independiente no sesgado (leakage)."
            )
        }
