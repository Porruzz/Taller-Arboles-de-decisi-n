"""
Gini Impurity and Split Evaluation module.
Implements the exact formulas for Gini index, weighted division impurity,
and impurity reduction (Delta G), adhering to 4 decimal precision.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

@dataclass
class SplitResult:
    pregunta: str
    feature: str
    threshold: float
    n_L: int
    counts_L: Dict[str, int]
    ids_L: List[int]
    gini_L: float
    n_R: int
    counts_R: Dict[str, int]
    ids_R: List[int]
    gini_R: float
    gini_div: float
    delta_g: float

def compute_gini(series: pd.Series) -> float:
    """
    Computes the Gini impurity for a given node S:
    G(S) = 1 - sum(p_k^2)
    """
    n = len(series)
    if n == 0:
        return 0.0
    proportions = series.value_counts() / n
    return float(1.0 - np.sum(proportions ** 2))

def evaluate_split(
    df: pd.DataFrame,
    feature: str,
    threshold: float,
    target_col: str = "Clase",
    id_col: str = "ID"
) -> SplitResult:
    """
    Evaluates a candidate binary partition:
    Left (SL): x <= threshold
    Right (SR): x > threshold
    """
    mask_left = df[feature] <= threshold
    left = df[mask_left]
    right = df[~mask_left]
    
    n_total = len(df)
    n_L, n_R = len(left), len(right)
    
    g_parent = compute_gini(df[target_col])
    g_L = compute_gini(left[target_col])
    g_R = compute_gini(right[target_col])
    
    # Weighted division impurity: G_div = (|SL|/|S|)*G(SL) + (|SR|/|S|)*G(SR)
    g_div = (n_L / n_total) * g_L + (n_R / n_total) * g_R
    delta_g = g_parent - g_div
    
    counts_L = left[target_col].value_counts().to_dict()
    counts_R = right[target_col].value_counts().to_dict()
    
    ids_L = list(left[id_col]) if id_col in left.columns else []
    ids_R = list(right[id_col]) if id_col in right.columns else []
    
    # Format counts as 'A/E' string
    count_str_L = f"{counts_L.get('A', 0)}A / {counts_L.get('E', 0)}E"
    count_str_R = f"{counts_R.get('A', 0)}A / {counts_R.get('E', 0)}E"
    
    return SplitResult(
        pregunta=f"{feature} <= {threshold}",
        feature=feature,
        threshold=threshold,
        n_L=n_L,
        counts_L=counts_L,
        ids_L=ids_L,
        gini_L=round(g_L, 4),
        n_R=n_R,
        counts_R=counts_R,
        ids_R=ids_R,
        gini_R=round(g_R, 4),
        gini_div=round(g_div, 4),
        delta_g=round(delta_g, 4)
    )

def compare_root_candidates(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[SplitResult]]:
    """
    Compares exclusively the two candidate questions specified in Section 2:
    1. Quilates <= 0.75
    2. Profundidad <= 63.5
    """
    candidates = [
        ("Quilates", 0.75),
        ("Profundidad", 63.5)
    ]
    
    results = [evaluate_split(df, feat, thresh) for feat, thresh in candidates]
    
    rows = []
    for r in results:
        rows.append({
            "Pregunta": r.pregunta,
            "nL": r.n_L,
            "A/E en SL": f"{r.counts_L.get('A', 0)}/{r.counts_L.get('E', 0)}",
            "G(SL)": f"{r.gini_L:.4f}",
            "nR": r.n_R,
            "A/E en SR": f"{r.counts_R.get('A', 0)}/{r.counts_R.get('E', 0)}",
            "G(SR)": f"{r.gini_R:.4f}",
            "Gdiv": f"{r.gini_div:.4f}",
            "DeltaG": f"{r.delta_g:.4f}"
        })
        
    summary_df = pd.DataFrame(rows)
    return summary_df, results

def evaluate_all_possible_splits(df_subset: pd.DataFrame, feature: str) -> List[SplitResult]:
    """
    Generates midpoints between sorted unique values and evaluates Gini splits.
    Used for exploring partitions on impure child nodes (Section 3).
    """
    unique_vals = sorted(df_subset[feature].unique())
    midpoints = [(unique_vals[i] + unique_vals[i+1]) / 2.0 for i in range(len(unique_vals) - 1)]
    
    results = []
    for t in midpoints:
        res = evaluate_split(df_subset, feature, round(t, 3))
        results.append(res)
    return results
