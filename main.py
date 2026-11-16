from nhanes import NHANES
from part_a.scores_partA import generate_scorecard
from part_a.oof_partA import generate_oof
from part_a.analysis_partA import analyze_oof_predictions
from part_a.explain_partA import generate_explanations

if __name__ == "__main__":
    nh = NHANES()
    X_train, X_test, y_train, y_test, df_1720, _ = nh.load_data_a(force_rebuild=False)
    y_train = y_train.astype(int).values.ravel()
    y_test = y_test.astype(int).values.ravel()
    
    # Upfront Leakage Check
    leaky = [c for c in X_train.columns if any(s in c.lower() for s in ("hba1c", "glucose", "fasting"))]
    if leaky:
        raise ValueError(f"Leakage columns found in X_train: {leaky}")
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}, train prevalence: {y_train.mean():.3f}")

    # Phase 1: Heavy Computation (Only run these if data/models change)
    generate_scorecard(X_train, y_train)
    generate_oof(X_train, y_train, df_1720)
    
    # Phase 2: Instant Analysis & XAI
    analyze_oof_predictions()
    generate_explanations(X_train, y_train, X_test, y_test)