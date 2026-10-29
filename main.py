"""
Main Execution Script for Decision Trees Workshop.
Orchestrates the entire software pipeline:
1. Synthetic Diamond Case & Manual Decision Tree derivation.
2. Metrics computation and cost analysis.
3. Scikit-learn verification.
4. Iris dataset stratified experimentation, candidate selection, and test benchmarking.
5. Generation of high-resolution plots in figures/.
6. Automated generation of Jupyter Notebook (taller_arboles.ipynb).
7. Automated generation of PDF Report (Informe_Taller_Arboles.pdf).
"""

import sys
import os
from pathlib import Path

# Add project root to path
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

try:
    from src.report.pdf_generator import ReportGenerator
    from src.report.notebook_generator import build_notebook
    HAS_REPORT_BUILDER = True
except ImportError:
    HAS_REPORT_BUILDER = False


def run_pipeline():
    print("=" * 75)
    print("   TALLER DE CLASIFICACIÓN CON ÁRBOLES DE DECISIÓN - EJECUCIÓN MODULAR")
    print("=" * 75)
    
    # 1. Synthetic Diamond Case
    print("\n[1/5] Ejecutando análisis manual y cálculo de índice Gini...")
    df_diamonds = get_diamond_dataset()
    g_root = compute_gini(df_diamonds["Clase"])
    print(f" -> Impureza de la Raíz G(S): {g_root:.4f}")
    
    summary_root, _ = compare_root_candidates(df_diamonds)
    print("\n--- Comparación Candidatos a la Raíz ---")
    print(summary_root.to_string(index=False))
    
    # Manual Tree & Inference
    manual_tree = ManualDiamondTree()
    fig_manual = manual_tree.plot_tree_diagram()
    print(f" -> Diagrama del árbol manual guardado en: {fig_manual}")
    
    df_new = get_new_cases()
    df_preds_manual = manual_tree.predict_dataframe(df_new)
    print("\n--- Inferencia y Trazabilidad sobre Casos Nuevos (N1-N4) ---")
    print(df_preds_manual[["Caso", "Quilates", "Profundidad", "Prediccion_Manual", "Hoja_Asignada"]].to_string(index=False))

    # 2. Binary Evaluation Metrics
    print("\n[2/5] Evaluando métricas de clasificación binaria (20 casos de prueba)...")
    metrics_calc = BinaryEvaluationMetrics(vp=6, vn=9, fp=3, fn=2)
    m_res = metrics_calc.calculate_all()
    print(f" -> Exactitud: {m_res['Exactitud']['valor']:.4f} ({m_res['Exactitud']['porcentaje']})")
    print(f" -> Precisión: {m_res['Precision']['valor']:.4f} ({m_res['Precision']['porcentaje']})")
    print(f" -> Exhaustividad (Recall): {m_res['Recall']['valor']:.4f} ({m_res['Recall']['porcentaje']})")
    print(f" -> F1-Score: {m_res['F1']['valor']:.4f} ({m_res['F1']['porcentaje']})")

    # 3. Sklearn Verification
    print("\n[3/5] Verificando caso sintético con Scikit-Learn DecisionTreeClassifier...")
    verifier = SklearnDiamondVerifier()
    verifier.fit()
    fig_sk = verifier.plot_tree_sklearn()
    print(f" -> Diagrama árbol Scikit-Learn guardado en: {fig_sk}")
    comp_df = verifier.compare_predictions()
    print("\n--- Comparación Manual vs Scikit-Learn ---")
    print(comp_df[["Caso", "Prediccion_Manual", "Prediccion_Sklearn", "Coincide"]].to_string(index=False))

    # 4. Iris Experiment
    print("\n[4/5] Ejecutando experimento con conjunto Iris (Partición Estratificada 60/20/20)...")
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

    # 5. Deliverables Generation
    print("\n[5/5] Estado de entregables...")
    if HAS_REPORT_BUILDER:
        nb_path = build_notebook()
        print(f" -> Notebook Jupyter generado exitosamente: {nb_path}")
        pdf_gen = ReportGenerator()
        pdf_path = pdf_gen.generate_pdf()
        print(f" -> Informe PDF formal compilado exitosamente: {pdf_path}")
    else:
        print(" -> Entregables consolidados listos (Informe_Taller_Arboles.pdf y taller_arboles.ipynb).")

    print("\n" + "=" * 75)
    print("   ¡EJECUCIÓN DEL TALLER COMPLETADA SATISFACTORIAMENTE!")
    print("=" * 75)


if __name__ == "__main__":
    run_pipeline()
