# part_a/scores_partA.py
import traceback
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, precision_recall_curve
)
from sklearn.utils.class_weight import compute_sample_weight

from config import *
from pipeline import make_pipeline


def best_f1_threshold(y_true, y_prob):
    prec, rec, thresh = precision_recall_curve(y_true, y_prob)
    f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec + 1e-12), 0)
    best_idx = np.argmax(f1[:-1])  # last PR point has no matching threshold
    return thresh[best_idx]


def generate_scorecard(X_train, y_train):
    # main.py passes y_train as a numpy array; the per-fold loop below needs .iloc,
    # so wrap it back into a Series aligned to X_train's index.
    y_train = pd.Series(np.asarray(y_train), index=X_train.index, name='target')

    results, all_folds, failed = [], [], []
    cv_strategy = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)

    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    MODELS["XGBoost"][0].set_params(scale_pos_weight=pos_weight)

    for model_name, (model, scale_type) in MODELS.items():
        print(f"\n--- Evaluating {model_name} ---")
        fold_rows = []
        try:
            for train_idx, test_idx in cv_strategy.split(X_train, y_train):
                X_tr, X_te = X_train.iloc[train_idx], X_train.iloc[test_idx]
                y_tr, y_te = y_train.iloc[train_idx], y_train.iloc[test_idx]

                # fit on the full training fold
                pipe = make_pipeline(model, scale_type)
                if model_name == "GaussianNB":
                    weights = compute_sample_weight('balanced', y_tr)
                    pipe.fit(X_tr, y_tr, model__sample_weight=weights)
                else:
                    pipe.fit(X_tr, y_tr)

                prob_te = pipe.predict_proba(X_te)[:, 1]
                roc = roc_auc_score(y_te, prob_te)
                pr_auc = average_precision_score(y_te, prob_te)

                # pick the F1-optimal threshold using an inner split of the TRAINING fold only
                X_tr_sub, X_thresh, y_tr_sub, y_thresh = train_test_split(
                    X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=RANDOM_STATE
                )
                pipe_inner = make_pipeline(model, scale_type)
                if model_name == "GaussianNB":
                    weights_sub = compute_sample_weight('balanced', y_tr_sub)
                    pipe_inner.fit(X_tr_sub, y_tr_sub, model__sample_weight=weights_sub)
                else:
                    pipe_inner.fit(X_tr_sub, y_tr_sub)

                prob_thresh = pipe_inner.predict_proba(X_thresh)[:, 1]
                thresh = best_f1_threshold(y_thresh, prob_thresh)

                y_pred_te = (prob_te >= thresh).astype(int)
                fold_rows.append({
                    'roc_auc': roc,
                    'average_precision': pr_auc,
                    'precision': precision_score(y_te, y_pred_te, zero_division=0),
                    'recall': recall_score(y_te, y_pred_te, zero_division=0),
                    'f1': f1_score(y_te, y_pred_te, zero_division=0),
                    'threshold_used': thresh,
                })
        except Exception as exc:
            print(f'-> Failed: {model_name}: {type(exc).__name__} : {exc}')
            traceback.print_exc()
            failed.append(model_name)
            continue

        fold_df = pd.DataFrame(fold_rows)
        fold_df.insert(0, 'Model', model_name)
        all_folds.append(fold_df)

        row = {'Model': model_name}
        for m in ['roc_auc', 'average_precision', 'precision', 'recall', 'f1']:
            row[f'{m}_mean'] = fold_df[m].mean().round(4)
            row[f'{m}_std'] = fold_df[m].std().round(4)
        results.append(row)

        print(f"--> ROC-AUC = {row['roc_auc_mean']:.4f} | F1 = {row['f1_mean']:.4f} "
              f"| Precision = {row['precision_mean']:.4f} | avg threshold ≈ {fold_df['threshold_used'].mean():.3f}")

    if results:
        scorecard = pd.DataFrame(results).sort_values(by='roc_auc_mean', ascending=False)
        scorecard.to_csv(OUT_DIR / 'scorecard_A.csv', index=False)
        pd.concat(all_folds, ignore_index=True).to_csv(OUT_DIR / 'per_fold_scores.csv', index=False)
        print(f"\nSaved 'scorecard_A.csv' and 'per_fold_scores.csv' to {OUT_DIR.resolve()}")

    if failed:
        print(f"Models that failed: {failed}")