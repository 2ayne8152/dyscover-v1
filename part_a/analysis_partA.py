from os.path import split

import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from config import *

def analyze_oof_predictions():
    # loads raw OOF probabilities, calculates optimal Youden thresholds (decision boundary formula for binary classification for best balance between sensitivity and specificity)
    # evaluates subgroup performance, and export a formatted csv for report

    print('\n--- Running Cross-Fitted Subgroup Analysis ---')
    oof_path = OUT_DIR/'oof_predictions.csv'
    if not oof_path.exists(): raise FileNotFoundError(f'Missing {oof_path}. Run generate_oof() first.')

    oof_df = pd.read_csv(oof_path, index_col=0)
    model_cols = [c for c in oof_df.columns if c.endswith('_prob')]
    metrics_records = []
    threshold_records = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for prob_col in model_cols:
        model_name = prob_col.replace('_prob', '')
        y_prob = oof_df[prob_col]
        y_true = oof_df['y_true']

        # false positive rate, true positive rate for in-sample set
        fpr_is, tpr_is, thresh_is = roc_curve(y_true, y_prob)
        best_idx_is = np.argmax(tpr_is - fpr_is)
        global_thresh = thresh_is[best_idx_is]
        oof_df[f'{model_name}_is_pred'] =(y_prob >= global_thresh).astype(int)
        overall_auc = roc_auc_score(y_true, y_prob)

        # print(f'\n{model_name} Optimal Threshold: {best_thresh:.4f}')

        cv_preds = np.zeros(len(oof_df))
        fold_thresholds = []

        for train_idx, test_idx in skf.split(oof_df, y_true):
            fpr_cv, tpr_cv, thresh_cv = roc_curve(y_true.iloc[train_idx], y_prob.iloc[train_idx])
            best_idx_cv = np.argmax(tpr_cv - fpr_cv)
            fold_thresh = thresh_cv[best_idx_cv]
            fold_thresholds.append(fold_thresh)

            cv_preds[test_idx] = (y_prob.iloc[test_idx] >= fold_thresh).astype(int)

        oof_df[f'{model_name}_cv_pred'] = cv_preds

        thresh_mean = np.mean(fold_thresholds)
        thresh_std = np.std(fold_thresholds)
        print(f'\n{model_name}:')
        print(f' Global thresh: {global_thresh:.4f}')
        print(f' CV thresh:     {thresh_mean:.4f} +- {thresh_std:.4f}')

        threshold_records.append({
            "Model": model_name,
            "Global_Thresh": global_thresh,
            "CV_Thresh_Mean": thresh_mean,
            "CV_Thresh_Std": thresh_std
        })

        for group in SUBGROUPS:
            for level in oof_df[group].dropna().unique():
                subset = oof_df[oof_df[group] == level]
                if len(subset['y_true'].unique()) < 2: continue

                sub_auc = roc_auc_score(subset['y_true'], subset[prob_col])

                # Cross-fitted Metrics
                tn_cv, fp_cv, fn_cv, tp_cv = confusion_matrix(subset['y_true'], subset[f'{model_name}_cv_pred']).ravel()
                fpr_cv_rate = fp_cv/ (fp_cv+tn_cv) if (fp_cv+tn_cv) > 0 else 0
                fnr_cv_rate = fn_cv/ (fn_cv+tp_cv) if (fn_cv+tp_cv) > 0 else 0

                # In-Sample Metrics
                tn_is, fp_is, fn_is, tp_is = confusion_matrix(subset['y_true'], subset[f'{model_name}_is_pred']).ravel()
                fpr_is_rate = fp_is/ (fp_is+tn_is) if (fp_is+tn_is) > 0 else 0
                fnr_is_rate = fn_is/ (fn_is+tp_is) if (fn_is+tp_is) > 0 else 0

                auc_gap = sub_auc - overall_auc

                metrics_records.append({
                    "Model": model_name,
                    "Demographic": group,
                    "Level": level,
                    "N": len(subset),
                    "AUC": sub_auc,
                    "AUC_Gap": auc_gap,
                    "FPR_CV": fpr_cv_rate,
                    "FNR_CV": fnr_cv_rate,
                    "FPR_InSample": fpr_is_rate,
                    "FNR_InSample": fnr_is_rate,
                    "FPR_Leakage_Diff": fpr_is_rate - fpr_cv_rate,
                    "FNR_Leakage_Diff": fnr_is_rate - fnr_cv_rate
                })
    pd.DataFrame(metrics_records).to_csv(OUT_DIR / 'subgroup_metrics_cv.csv', index=False, float_format='%.4f')
    pd.DataFrame(threshold_records).to_csv(OUT_DIR / 'threshold_stability.csv', index=False, float_format='%.4f')
    print(f"\nSaved cross-fitted metrics to {OUT_DIR.resolve()}/subgroup_metrics_cv.csv")
    print(f"Saved threshold stability to {OUT_DIR.resolve()}/threshold_stability.csv")