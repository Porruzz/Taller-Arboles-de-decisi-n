"""
Script principal de ejecución del taller de árboles de decisión.
Ejecuta la pipeline completa de modelos, cálculos analíticos y experimentación.
"""

import sys
from pathlib import Path

# Agregar raíz al path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.synthetic.data import get_diamond_dataset, get_new_cases
from src.synthetic.gini import compare_root_candidates, compute_gini
from src.synthetic.manual_tree import ManualDiamondTree
from src.synthetic.metrics import BinaryEvaluationMetrics, analyze_error_costs, analyze_imbalanced_dataset
from src.synthetic.sklearn_verifier import SklearnDiamondVerifier
from src.iris.dataset import IrisDataLoader
from src.iris.experiment import IrisModelSelector
from src.iris.evaluation import IrisTestEvaluator
from src.iris.visualization import IrisVisualizer


def run_pipeline():
    print("=" * 75)
    print("   TALLER DE CLASIFICACIÓN CON ÁRBOLES DE DECISIÓN")
    print("   Estudiante: Santiago Parra Acuña | Inteligencia Artificial")
    print("=" * 75)
    
    # 1. Caso Sintético Diamantes y Cálculo Gini
    print("\n[1/4] Análisis manual y cálculo de índice Gini...")
    df_diamonds = get_diamond_dataset()
    g_root = compute_gini(df_diamonds["Clase"])
    print(f" -> Impureza de la Raíz G(S): {g_root:.4f}")
    
    summary_root, _ = compare_root_candidates(df_diamonds)
    print("\n--- Comparación Candidatos a la Raíz ---")
    print(summary_root.to_string(index=False))
    
    # Árbol manual e inferencia N1-N4
    manual_tree = ManualDiamondTree()
    fig_manual = manual_tree.plot_tree_diagram()
    print(f" -> Diagrama del árbol manual generado en: {fig_manual}")
    
    df_new = get_new_cases()
    df_preds_manual = manual_tree.predict_dataframe(df_new)
    print("\n--- Inferencia sobre Casos Nuevos (N1-N4) ---")
    print(df_preds_manual[["Caso", "Quilates", "Profundidad", "Prediccion_Manual", "Hoja_Asignada"]].to_string(index=False))

    # 2. Métricas de Evaluación Binaria
    print("\n[2/4] Evaluación de métricas binarias (20 casos de prueba)...")
    metrics_calc = BinaryEvaluationMetrics(vp=6, vn=9, fp=3, fn=2)
    m_res = metrics_calc.calculate_all()
    print(f" -> Exactitud: {m_res['Exactitud']['valor']:.4f} ({m_res['Exactitud']['porcentaje']})")
    print(f" -> Precisión: {m_res['Precision']['valor']:.4f} ({m_res['Precision']['porcentaje']})")
    print(f" -> Exhaustividad (Recall): {m_res['Recall']['valor']:.4f} ({m_res['Recall']['porcentaje']})")
    print(f" -> F1-Score: {m_res['F1']['valor']:.4f} ({m_res['F1']['porcentaje']})")

    # 3. Verificación con Scikit-Learn (Diamantes)
    print("\n[3/4] Verificación del caso sintético con Scikit-Learn...")
    verifier = SklearnDiamondVerifier()
    verifier.fit()
    fig_sk = verifier.plot_tree_sklearn()
    print(f" -> Diagrama árbol Scikit-Learn generado en: {fig_sk}")
    comp_df = verifier.compare_predictions()
    print("\n--- Comparación Manual vs Scikit-Learn ---")
    print(comp_df[["Caso", "Prediccion_Manual", "Prediccion_Sklearn", "Coincide"]].to_string(index=False))

    # 4. Experimento con Iris (60/20/20)
    print("\n[4/4] Experimento con conjunto Iris (Partición Estratificada 60/20/20)...")
    loader = IrisDataLoader()
    selector = IrisModelSelector(loader)
    cand_df = selector.run_training_and_validation()
    best_name, best_model, _ = selector.select_best_model()
    
    print("\n--- Tabla Comparativa Candidatos Iris (Entrenamiento vs Validación) ---")
    print(cand_df.to_string(index=False))
    print(f"\n -> Modelo Ganador Seleccionado: {best_name}")

    evaluator = IrisTestEvaluator(best_model, loader)
    test_res = evaluator.evaluate_test_set()
    bench_df = evaluator.benchmark_against_dummy()
    
    print("\n--- Benchmark Final en Prueba vs DummyClassifier ---")
    print(bench_df.to_string(index=False))

    visualizer = IrisVisualizer(best_model, loader)
    fig_iris = visualizer.plot_selected_tree()
    fig_cm = visualizer.plot_confusion_matrix(test_res["confusion_matrix"])
    print(f" -> Árbol Iris guardado en: {fig_iris}")
    print(f" -> Matriz de confusión guardada en: {fig_cm}")

    print("\n" + "=" * 75)
    print("   EJECUCIÓN FINALIZADA CON ÉXITO")
    print("=" * 75)


if __name__ == "__main__":
    run_pipeline()
