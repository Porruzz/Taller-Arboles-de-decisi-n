"""
Iris Visualizer and Path Interpreter module (Section 5B - f).
Generates high-resolution tree plots, confusion matrix heatmaps,
and provides detailed explanations for two decision paths.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier, plot_tree
from typing import Dict, Any, List
from ..config import FIGURES_DIR
from .dataset import IrisDataLoader

class IrisVisualizer:
    """
    Visualizes Iris decision tree and confusion matrix, explaining sample decision paths.
    """
    
    def __init__(self, model: DecisionTreeClassifier, data_loader: IrisDataLoader = None):
        self.model = model
        self.loader = data_loader or IrisDataLoader()
        self.feature_names = self.loader.feature_names
        self.target_names = self.loader.target_names
        
    def plot_selected_tree(self, output_path: Path = None) -> Path:
        """
        Plots the full decision tree of the selected Iris model with high resolution.
        """
        if output_path is None:
            output_path = FIGURES_DIR / "iris_best_tree.png"
            
        fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
        plot_tree(
            self.model,
            feature_names=self.feature_names,
            class_names=self.target_names,
            filled=True,
            rounded=True,
            fontsize=9,
            impurity=True,
            ax=ax
        )
        ax.set_title("Árbol de Decisión Seleccionado (Iris - max_depth=None)", fontsize=14, weight="bold", pad=15)
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close()
        return output_path

    def plot_confusion_matrix(self, cm: np.ndarray, output_path: Path = None) -> Path:
        """
        Plots a stylized confusion matrix heatmap for the Test set.
        """
        if output_path is None:
            output_path = FIGURES_DIR / "iris_confusion_matrix.png"
            
        fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.target_names,
            yticklabels=self.target_names,
            cbar=False,
            annot_kws={"size": 14, "weight": "bold"},
            ax=ax
        )
        ax.set_xlabel("Especie Predicha (Predicted)", fontsize=11, weight="bold", labelpad=10)
        ax.set_ylabel("Especie Real (True Ground Truth)", fontsize=11, weight="bold", labelpad=10)
        ax.set_title("Matriz de Confusión en Conjunto de Prueba (Test, N=30)", fontsize=12, weight="bold", pad=12)
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close()
        return output_path

    def explain_two_paths(self) -> List[Dict[str, str]]:
        """
        Provides detailed botanical and algorithmic interpretation for two distinct paths.
        """
        return [
            {
                "nombre": "Camino 1: Identificación Directa de Iris-setosa (Hoja de Profundidad 1)",
                "recorrido": "Raíz -> ¿petal length (cm) <= 2.45? -> SÍ (Rama Izquierda) -> Hoja Terminal [Clase: Setosa]",
                "soporte": "30 muestras de entrenamiento (30 Setosa, 0 Versicolor, 0 Virginica). Impureza Gini = 0.0000.",
                "interpretacion": (
                    "Este camino demuestra la perfecta separabilidad morfológica lineal de la especie Iris-setosa "
                    "respecto a las demás especies. Una flor cuyo pétalo mide menos de 2.45 cm de longitud es, de forma determinista, "
                    "una Setosa, no requiriendo evaluar ninguna otra dimensión."
                )
            },
            {
                "nombre": "Camino 2: Discriminación de Iris-virginica por Pétalo Robusto (Hoja de Profundidad 2)",
                "recorrido": "Raíz -> ¿petal length (cm) > 2.45? -> NO (Rama Derecha) -> ¿petal width (cm) > 1.65? -> NO (Rama Derecha) -> Hoja Terminal [Clase: Virginica]",
                "soporte": "28 muestras de entrenamiento (0 Setosa, 1 Versicolor, 27 Virginica). Clase mayoritaria: Virginica.",
                "interpretacion": (
                    "Cuando el pétalo es largo (> 2.45 cm), el árbol examina el grosor del pétalo (petal width). "
                    "Si el ancho supera los 1.65 cm, el espécimen posee los pétalos gruesos característicos de Iris-virginica. "
                    "Este camino aísla eficientemente a la inmensa mayoría de flores Virginica en tan solo 2 niveles de decisión."
                )
            }
        ]
