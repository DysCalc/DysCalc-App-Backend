import pandas as pd
from .C45DecisionTree import C45DecisionTree

EPSILON = 1e-9

def get_probs(tree: C45DecisionTree, X: pd.DataFrame) -> list[float]:
    """
    Extract P(at-risk) from diagnostic outputs.
    Zeroes out incomplete flags from task_importance post-prediction
    (does not affect tree split decisions).
    """
    diagnostics = tree.predict_with_diagnostics(X)
    probs = []
    for d in diagnostics:
        total = sum(d.task_importance_scores.values())
        if total > 0:
            for f in d.task_importance_scores:
                d.task_importance_scores[f] /= total

        p = d.confidence if int(d.predicted_class) == 1 else 1 - d.confidence
        probs.append(p)
    return probs

def renaming_features(df: pd.DataFrame) -> pd.DataFrame:
    name_mapping = {
        'number_comparison': 'NC',
        'dot_matching': 'DM', 
        'single_addition': 'ADD',
        'single_subtraction': 'SUB', 
        'number_series': 'NS',
        'complex_arithmetic': 'CA'
    }
    df_renamed = df.rename(columns=name_mapping)
    return df_renamed


def compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute all derived features from raw features deterministically."""
    df_der = df.copy()
    df_der['NP'] = (df_der['NC'] + df_der['DM']) / 2                           # Eq. 3.3
    df_der['SN'] = df_der['NC'] - df_der['DM']                                  # Eq. 3.4
    df_der['AF'] = (df_der['NS'] + df_der['ADD'] + df_der['SUB'] + df_der['CA']) / 4   # Eq. 3.5
    df_der['BC'] = (df_der['ADD'] + df_der['SUB']) / 2 - df_der['CA']              # Eq. 3.6
    df_der['AS'] = df_der['ADD'] - df_der['SUB']                                # Eq. 3.7
    df_der['PF'] = df_der['AF'] / (df_der['NP'] + EPSILON)                     # Eq. 3.8
    return df_der