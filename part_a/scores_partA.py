import traceback, os
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

from config import *
from pipeline import make_pipeline

def generate_scorecard(X_train, y_train):
    # evaluates all models in the registry using Repeated Stratified K-Fold.
    # Aggregated scorecard and per-fold metrics will be exported.

    results, all_folds, failed = [], [], []

    cv_strategy = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
    scoring = ['roc_auc', 'average_precision', 'precision', 'recall', 'f1']

    for model_name, (model, scale_type) in MODELS.items():
        print(f"\n--- Evaluating{model_name} ---")
        try:
            pipe = make_pipeline(model, scale_type)

            cv_scores = cross_validate(
                pipe, X_train, y_train,
                cv=cv_strategy, scoring=scoring,
                n_jobs=max(1, os.cpu_count() - 2),
                error_score='raise',
            )
        except Exception as exc:
            print(f'-> Failed: {model_name}: {type(exc).__name__} : {exc}')
            traceback.print_exc()
            failed.append(model_name)
            continue

        roc_auc_mean = cv_scores["test_roc_auc"].mean()
        roc_auc_std = cv_scores["test_roc_auc"].std()
        pr_auc_mean = cv_scores["test_average_precision"].mean()
        pr_auc_std = cv_scores["test_average_precision"].std()

        print(f"--> Result: ROC-AUC = {roc_auc_mean:.4f} (+/- {roc_auc_std:.4f}) | PR-AUC = {pr_auc_mean:.4f} (+/- {pr_auc_std:.4f})")

        row = {'Model':model_name}
        for m in scoring:
            vals = cv_scores[f'test_{m}']
            row[f'{m}_mean'] = vals.mean().round(4)
            row[f'{m}_std'] = vals.std().round(4)
        results.append(row)

        fold_scores = pd.DataFrame({m:cv_scores[f'test_{m}'] for m in scoring})
        fold_scores.insert(0, 'Model',model_name)
        all_folds.append(fold_scores)

    if results:
        scorecard = pd.DataFrame(results).sort_values(by='roc_auc_mean', ascending=False)
        scorecard.to_csv(OUT_DIR/'scorecard_A.csv', index=False)
        pd.concat(all_folds, ignore_index=True).to_csv(OUT_DIR/'per_fold_scores.csv', index=False)
        print(f"\nSaved 'scorecard_A.csv' and 'per_fold_scores.csv' to {OUT_DIR.resolve()}")

    if failed: print(f"Models that failed: {failed}")