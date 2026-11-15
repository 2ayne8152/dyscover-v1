import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import shap

from config import *
from pipeline import make_pipeline

def auto_shortlist_champion():
    """Reads the scorecard and returns the name of the best performing model."""
    scorecard_path = OUT_DIR / "scorecard_A.csv"
    if not scorecard_path.exists():
        raise FileNotFoundError("Missing scorecard_A.csv. Run generate_scorecard() first.")
        
    scorecard = pd.read_csv(scorecard_path)
    champion_name = scorecard.loc[scorecard['roc_auc_mean'].idxmax(), 'Model']
    return champion_name

def generate_explanations(X_train, y_train, X_val, y_val):
    print("\n--- Running XAI Engine ---")
    
    champion_name = auto_shortlist_champion()
    print(f"Champion Model Identified: {champion_name}")
    
    # Rebuild and fit the winning model
    model, scale_type = MODELS[champion_name]
    pipe = make_pipeline(model, scale_type)
    pipe.fit(X_train, y_train)

    print("\n1. Global Permutation Importance (Model-Agnostic)")
    result = permutation_importance(pipe, X_val, y_val, scoring='roc_auc', n_repeats=5, n_jobs=4, random_state=42)
    top_idx = result.importances_mean.argsort()[::-1][:5]
    for i in top_idx: 
        print(f"{X_val.columns[i]}: {result.importances_mean[i]:.4f} +/- {result.importances_std[i]:.4f}")

    print("\n2. Model-Specific Extraction")
    prep = pipe.named_steps['prep']
    fitted_model = pipe.named_steps['model']
    feature_names = prep.get_feature_names_out()
    
    # Route logic based on algorithm family
    if hasattr(fitted_model, 'coef_'):
        print(f"Linear model detected. Extracting coefficients...")
        coefficients = fitted_model.coef_[0]
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients,
            'Absolute_Impact': np.abs(coefficients)
        }).sort_values(by='Absolute_Impact', ascending=False)
        
        importance_df.to_csv(OUT_DIR / f"{champion_name}_coefficients.csv", index=False)
        print(f"Saved coefficients to {OUT_DIR.resolve()}/{champion_name}_coefficients.csv")

    elif "Tree" in champion_name or "Forest" in champion_name or "XGB" in champion_name:
        print(f"Tree model detected. Generating SHAP values...")
        X_val_transformed = prep.transform(X_val)
        explainer = shap.TreeExplainer(fitted_model)
        shap_values = explainer.shap_values(X_val_transformed)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        shap.summary_plot(shap_values, X_val_transformed, feature_names=feature_names, show=False)
        plt.savefig(OUT_DIR / f"{champion_name}_shap_summary.png", bbox_inches="tight")
        plt.close()
        print(f"Saved SHAP plot to {OUT_DIR.resolve()}/{champion_name}_shap_summary.png")
    else:
        print(f"No fast local XAI route built for {champion_name}. Skipping local explanations.")