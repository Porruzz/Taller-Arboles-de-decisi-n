"""
Iris Dataset and Partitioning module.
Handles loading Iris dataset and strictly reproducible 60%/20%/20% stratified partitioning.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Tuple, Dict, Any
from ..config import RANDOM_STATE, IRIS_DEV_SIZE, IRIS_VAL_SIZE_OF_DEV

class IrisDataLoader:
    """
    Loads and stratifies the Iris dataset according to the workshop specification.
    """
    
    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state
        self.iris = load_iris(as_frame=True)
        self.X = self.iris.data
        self.y = self.iris.target
        self.feature_names = list(self.iris.feature_names)
        self.target_names = list(self.iris.target_names)
        
    def get_partitions(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Creates stratified 60% Train / 20% Val / 20% Test partitions.
        Step 1: Separate 20% Test from remaining 80% Dev (stratified, random_state=42).
        Step 2: From Dev (80%), split 25% for Val (= 20% of total) and 75% for Train (= 60% of total).
        """
        X_dev, X_test, y_dev, y_test = train_test_split(
            self.X, self.y,
            test_size=IRIS_DEV_SIZE,
            stratify=self.y,
            random_state=self.random_state
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_dev, y_dev,
            test_size=IRIS_VAL_SIZE_OF_DEV,
            stratify=y_dev,
            random_state=self.random_state
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_partition_distribution(self) -> pd.DataFrame:
        """
        Returns a summary table of instances per class across Train, Val, and Test splits.
        """
        X_train, X_val, X_test, y_train, y_val, y_test = self.get_partitions()
        
        records = []
        for split_name, series in [("Entrenamiento (60%)", y_train), ("Validación (20%)", y_val), ("Prueba (20%)", y_test)]:
            counts = series.value_counts().sort_index()
            records.append({
                "Partición": split_name,
                "Total Muestras": len(series),
                "Setosa (0)": counts.get(0, 0),
                "Versicolor (1)": counts.get(1, 0),
                "Virginica (2)": counts.get(2, 0)
            })
            
        return pd.DataFrame(records)

    @staticmethod
    def get_justification_partitions() -> Dict[str, str]:
        """
        Provides methodological justifications for stratified splitting and fixed seed (Section 5B - a).
        """
        return {
            "descripcion_variables": (
                "El conjunto de datos Iris comprende 150 observaciones botánicas con 4 predictores continuos:\n"
                "1. Sepal length (longitud del sépalo en cm)\n"
                "2. Sepal width (ancho del sépalo en cm)\n"
                "3. Petal length (longitud del pétalo en cm)\n"
                "4. Petal width (ancho del pétalo en cm)\n"
                "La variable objetivo y corresponde a 3 especies taxonómicas: Setosa (0), Versicolor (1) y Virginica (2)."
            ),
            "justificacion_estratificacion": (
                "La estratificación garantiza que cada partición (Train: 90 flores, Val: 30 flores, Test: 30 flores) "
                "mantenga con exactitud la proporción original de las clases (33.3% por cada especie, es decir, exactamente "
                "30-30-30 en Train, 10-10-10 en Val y 10-10-10 en Test). Sin estratificación, particiones aleatorias en "
                "muestras reducidas pueden subrepresentar o sesgar la distribución de alguna de las especies."
            ),
            "justificacion_semilla_fija": (
                "El uso de una semilla fija (random_state=42) es un principio fundamental de la reproducibilidad científica. "
                "Permite que cualquier investigador o evaluador replique con exactitud matemática las particiones y "
                "resultados obtenidos, garantizando la trazabilidad experimental y evitando sesgos por re-muestreo favorable."
            )
        }
