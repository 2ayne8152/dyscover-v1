import traceback
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from config import *
from pipeline import make_pipeline

def generate_oof(X_train, y_train, df_1720):
    # generates Out-ofFold probabilities for all models
    print('\n---Generating OOF predictions ---')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    oof_df = pd.DataFrame({'y_true':y_train}, index=X_train.index)
    oof_df['is_asian_descent'] = df_1720.loc[X_train.index, 'is_asian_descent']

    race_map = {
        1.0: "Mexican American", 2.0: "Other Hispanic", 3.0: "Non-Hispanic White",
        4.0: "Non-Hispanic Black", 6.0: "Non-Hispanic Asian", 7.0: "Other Race"
    }

    oof_df['race_ethnicity'] = df_1720.loc[X_train.index, 'race_ethnicity'].map(race_map)
    oof_df['sex'] = X_train['sex']
    oof_df['age_group'] = pd.cut(df_1720.loc[X_train.index, 'age'], bins=AGE_BINS, labels=AGE_LABELS)

    for model_name, (model, scale_type) in MODELS.items():
        print(f"Predicting with {model_name}")
        try:
            pipe = make_pipeline(model, scale_type)
            cv_scores = cross_val_predict(pipe, X_train, y_train, cv=skf, method='predict_proba')[:,1]
            oof_df[f'{model_name}_prob'] = cv_scores
        except Exception as exc:
            print(f'-> Failed: {model_name}: {type(exc).__name__} : {exc}')
            traceback.print_exc()
            continue

    oof_df.to_csv(OUT_DIR/'oof_predictions.csv' , index=True, float_format='%.4f')
    print(f'Saved oof predictions to {OUT_DIR.resolve()}/oof_predictions.csv')