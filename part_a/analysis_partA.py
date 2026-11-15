import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from config import *

def analyze_oof_predictions():
    # loads raw OOF probabilities, calculates optimal Youden thresholds (decision boundary formula for binary classification for best balance between sensitivity and specificity)
    # evaluates subgroup performance, and export a formatted csv for report

    print('\n--- Running Subgroup Analysis ---')
    oof_path = OUT_DIR/'oof_predictions.csv'
    if not oof_path.exists(): raise FileNotFoundError(f'Missing {oof_path}. Run generate_oof() first.')

    oof_df = pd.read_csv(oof_path, index_col=0)
    model_cols = [c for c in oof_df.columns if c.endswith('_prob')]
    metrics_records = []

    for prob_col in model_cols:
        model_name = prob_col.replace('_prob', '')
        y_prob = oof_df[prob_col]

        # false positive rate, true positive rate
        fpr, tpr, thresholds = roc_curve(oof_df['y_true'], y_prob)
        best_idx = np.argmax(tpr-fpr)
        best_thresh = thresholds[best_idx]

        oof_df[f'{model_name}_pred'] = (y_prob >= best_thresh).astype(int)
        overall_auc = roc_auc_score(oof_df['y_true'], y_prob)

        print(f'\n{model_name} Optimal Threshold: {best_thresh:.4f}')

        for group in SUBGROUPS:
            for level in oof_df[group].dropna().unique():
                subset = oof_df[oof_df[group] == level]

                if len(subset['y_true'].unique()) < 2: continue

                sub_auc = roc_auc_score(subset['y_true'], subset[prob_col])
                tn, fp, fn, tp = confusion_matrix(subset['y_true'], subset[f'{model_name}_pred']).ravel()

                fpr_rate = fp/ (fp+tn) if (fp+tn) > 0 else 0
                fnr_rate = fn/ (fn+tp) if (fn+tp) > 0 else 0

                complement = oof_df[oof_df[group] != level]
                complement_auc = roc_auc_score(complement['y_true'], complement[prob_col])
                auc_gap = sub_auc - complement_auc

                metrics_records.append({
                    "Model": model_name,
                    "Demographic": group,
                    "Level": level,
                    "N": len(subset),
                    "Threshold": best_thresh,
                    "AUC": sub_auc,
                    "AUC_Gap": auc_gap,
                    "FPR": fpr_rate,
                    "FNR": fnr_rate
                })
    metrics_df = pd.DataFrame(metrics_records)
    metrics_df.to_csv(OUT_DIR/'subgroup_metrics.csv', index=False, float_format='%.4f')
    print(f"Saved subgroup metrics to {OUT_DIR.resolve()}/subgroup_metrics.csv")