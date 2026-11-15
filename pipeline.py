from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from config import *

def build_preprocessor(scale_type):
    # Imputer and encoder translation
    num_steps = [('imputer', SimpleImputer(strategy='median'))]
    if scale_type == 'standard':
        num_steps.append(('scaler', StandardScaler()))

    numeric_transformer = Pipeline(steps=num_steps)

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_COLS),
            ('cat', categorical_transformer, CATEGORICAL_COLS)
        ],
        remainder='drop'
    )


def make_pipeline(model, scale_type):
    # wrap the model with correct preprocessor
    preprocessor = build_preprocessor(scale_type=scale_type)
    return Pipeline([
        ('prep', preprocessor),
        ('model', model)
    ])