"""
Manual Decision Tree Model and Visualizer.
Implements the manual tree constructed from Section 2 & 3:
- Root: Profundidad <= 63.5
  - Left (Profundidad <= 63.5): Quilates <= 0.50
    - Left (Quilates <= 0.50): Leaf A (Pure: 2 A, 0 E)
    - Right (Quilates > 0.50): Leaf E (Pure: 0 A, 6 E)
  - Right (Profundidad > 63.5): Leaf A (Pure: 4 A, 0 E)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Dict, Any, List, Tuple
from ..config import FIGURES_DIR

class ManualDiamondTree:
    """
    Exact implementation of the manual decision tree derived through Gini impurity analysis.
    """
    
    def __init__(self):
        self.rules = [
            {
                "id": "Hoja 1",
                "condition": "SI Profundidad <= 63.5 Y Quilates <= 0.50",
                "clase": "A",
                "soporte": "2 A, 0 E (IDs: 1, 2)",
                "pureza": "100% (Gini = 0.0000)"
            },
            {
                "id": "Hoja 2",
                "condition": "SI Profundidad <= 63.5 Y Quilates > 0.50",
                "clase": "E",
                "soporte": "0 A, 6 E (IDs: 4, 6, 7, 9, 11, 12)",
                "pureza": "100% (Gini = 0.0000)"
            },
            {
                "id": "Hoja 3",
                "condition": "SI Profundidad > 63.5",
                "clase": "A",
                "soporte": "4 A, 0 E (IDs: 3, 5, 8, 10)",
                "pureza": "100% (Gini = 0.0000)"
            }
        ]
        
    def predict_one(self, quilates: float, profundidad: float) -> Tuple[str, str, str]:
        """
        Classifies a single diamond sample and returns:
        (prediction, leaf_id, path_description)
        """
        path = []
        if profundidad <= 63.5:
            path.append("Profundidad <= 63.5 (Rama Izq)")
            if quilates <= 0.50:
                path.append("Quilates <= 0.50 (Rama Izq)")
                leaf = "Hoja 1"
                pred = "A"
            else:
                path.append("Quilates > 0.50 (Rama Der)")
                leaf = "Hoja 2"
                pred = "E"
        else:
            path.append("Profundidad > 63.5 (Rama Der)")
            leaf = "Hoja 3"
            pred = "A"
            
        full_path_str = " -> ".join(["Raíz"] + path + [f"{leaf} [{pred}]"])
        return pred, leaf, full_path_str
        
    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies manual tree classification to a DataFrame containing 'Quilates' and 'Profundidad'.
        """
        df_out = df.copy()
        preds = []
        leaves = []
        paths = []
        for _, row in df.iterrows():
            p, l, path = self.predict_one(row["Quilates"], row["Profundidad"])
            preds.append(p)
            leaves.append(l)
            paths.append(path)
            
        df_out["Prediccion_Manual"] = preds
        df_out["Hoja_Asignada"] = leaves
        df_out["Recorrido_Seguido"] = paths
        return df_out

    def plot_tree_diagram(self, output_path: Path = None) -> Path:
        """
        Generates an architectural diagram of the manual decision tree using Matplotlib.
        """
        if output_path is None:
            output_path = FIGURES_DIR / "manual_tree_diagram.png"
            
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 7)
        ax.axis("off")
        
        # Styles
        box_root = dict(boxstyle="round,pad=0.5", facecolor="#E1F5FE", edgecolor="#0288D1", lw=2)
        box_internal = dict(boxstyle="round,pad=0.5", facecolor="#FFF9C4", edgecolor="#FBC02D", lw=2)
        box_leaf_A = dict(boxstyle="round,pad=0.5", facecolor="#C8E6C9", edgecolor="#388E3C", lw=2)
        box_leaf_E = dict(boxstyle="round,pad=0.5", facecolor="#FFCDD2", edgecolor="#D32F2F", lw=2)
        arrow_props = dict(arrowstyle="->", lw=1.8, color="#37474F")
        
        # Nodes coordinates
        # Root (depth 0)
        ax.text(5.0, 6.0, "¿Profundidad <= 63.5%?\n(Muestras = 12, Gini = 0.5000)\n[6 A / 6 E]",
                ha="center", va="center", bbox=box_root, fontsize=10, weight="bold")
                
        # Internal Node Left (depth 1)
        ax.text(2.5, 3.8, "¿Quilates <= 0.50?\n(Muestras = 8, Gini = 0.3750)\n[2 A / 6 E]",
                ha="center", va="center", bbox=box_internal, fontsize=9, weight="bold")
                
        # Leaf Right of Root (depth 1) - Pure A
        ax.text(7.5, 3.8, "HOJA 3 (Pura)\nClase: A (Aceptable)\n(Muestras = 4, Gini = 0.0000)\n[4 A / 0 E]\nIDs: 3, 5, 8, 10",
                ha="center", va="center", bbox=box_leaf_A, fontsize=9)
                
        # Leaf Left of Internal (depth 2) - Pure A
        ax.text(1.2, 1.2, "HOJA 1 (Pura)\nClase: A (Aceptable)\n(Muestras = 2, Gini = 0.0000)\n[2 A / 0 E]\nIDs: 1, 2",
                ha="center", va="center", bbox=box_leaf_A, fontsize=9)
                
        # Leaf Right of Internal (depth 2) - Pure E
        ax.text(3.8, 1.2, "HOJA 2 (Pura)\nClase: E (Excelente)\n(Muestras = 6, Gini = 0.0000)\n[0 A / 6 E]\nIDs: 4, 6, 7, 9, 11, 12",
                ha="center", va="center", bbox=box_leaf_E, fontsize=9)
                
        # Connections
        # Root to Left
        ax.annotate("Sí (x <= 63.5)", xy=(3.3, 4.4), xytext=(4.3, 5.5),
                    arrowprops=arrow_props, fontsize=9, color="#1565C0", weight="bold")
        # Root to Right
        ax.annotate("No (x > 63.5)", xy=(6.7, 4.4), xytext=(5.7, 5.5),
                    arrowprops=arrow_props, fontsize=9, color="#C62828", weight="bold")
                    
        # Internal to Leaf 1
        ax.annotate("Sí (x <= 0.50)", xy=(1.5, 1.8), xytext=(2.1, 3.2),
                    arrowprops=arrow_props, fontsize=9, color="#1565C0", weight="bold")
        # Internal to Leaf 2
        ax.annotate("No (x > 0.50)", xy=(3.5, 1.8), xytext=(2.9, 3.2),
                    arrowprops=arrow_props, fontsize=9, color="#C62828", weight="bold")
                    
        plt.title("Árbol de Decisión Manual - Caso Diamantes (Profundidad Máxima = 2)", fontsize=12, weight="bold", pad=20)
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close()
        
        return output_path

def get_discussion_section_3() -> Dict[str, str]:
    """
    Returns the analytical discussion points requested in Section 3.
    """
    return {
        "acierto_casos_nuevos": (
            "No es posible afirmar con certeza matemática si el modelo acertó o falló en los casos N1 a N4, "
            "debido a que dichas observaciones son no etiquetadas (carecen de ground truth). En aprendizaje automático, "
            "el rendimiento solo puede contrastarse frente a datos con etiquetas reales de referencia."
        ),
        "demostracion_exactitud_100": (
            "Una exactitud del 100% sobre las 12 filas de entrenamiento demuestra que el árbol tiene suficiente capacidad "
            "expresiva para separar perfectamente las clases del conjunto observado (ajuste perfecto de la muestra). "
            "Sin embargo, NO demuestra capacidad de generalización hacia nuevas observaciones del mundo real. "
            "De hecho, lograr 100% en una muestra tan pequeña (12 casos) conlleva un alto riesgo de sobreajuste (overfitting) "
            "o de memorización de peculiaridades muestrales no generalizables."
        )
    }
