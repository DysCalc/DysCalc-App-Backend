import pandas as pd
import sys
from .C45DecisionTree import C45DecisionTree
from dataclasses import asdict

EPSILON = 1e-9

# 1. Patch sys.modules globally (ONCE at startup)
if 'C45DecisionTree' not in sys.modules:
    import ml.C45DecisionTree
    sys.modules['C45DecisionTree'] = ml.C45DecisionTree
if 'Dataclasses' not in sys.modules:
    import ml.Dataclasses
    sys.modules['Dataclasses'] = ml.Dataclasses
# 2. Load the model globally (ONCE at startup)
# This keeps the loaded model in memory so inference is instantaneous.
MODEL_PATH = "./ml/models/v1.pkl"
LOADED_TREE, OPTIMAL_THRESHOLD, CONF_FACT, MIN_SAMPLES_LEAF, MAX_DEPTH, EPSILON = C45DecisionTree.load_model(MODEL_PATH)

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

def predict_single_diagnostic(complete_features: pd.DataFrame):
    """
    Predicts diagnostic for a single row DataFrame of complete features.
    Returns the first diagnostic result.
    """
    diagnostics = LOADED_TREE.predict_with_diagnostics(complete_features)
    return asdict(diagnostics[0])


def process_and_predict_diagnostic(raw_features: dict):
    """
    End-to-end helper: takes the raw JSON dictionary, processes it,
    and returns the predicted diagnostic result.
    """
    # 1. Convert to DataFrame
    df = pd.DataFrame([raw_features])
    
    # 2. Process features
    renamed_features = renaming_features(df)
    complete_features = compute_derived(renamed_features)
    
    # 3. Predict and return
    return predict_single_diagnostic(complete_features)
