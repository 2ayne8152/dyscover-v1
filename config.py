from pathlib import Path
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ========================================
# Configurations
# ========================================

RANDOM_STATE = 42

DATA_DIR = Path('./data/raw')

CACHE_DIR = Path('./data/cache')
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_1720 = CACHE_DIR/'1720.feather'
CACHE_2123 = CACHE_DIR/'2123.feather'

OUT_DIR = Path("./results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

AGE_BINS = [17, 39, 59, 120]
AGE_LABELS = ['18-39', '40-59', '60+']
SUBGROUPS = ['sex', 'is_asian_descent', 'age_group']

COLUMNS = {
    'DEMO': ['SEQN', 'RIDAGEYR', 'RIAGENDR', 'RIDRETH3', 'DMDEDUC2', 'INDFMPIR', 'RIDEXPRG', 'DMDMARTZ'],
    'BMX' : ['SEQN', 'BMXWT', 'BMXHT', 'BMXBMI', 'BMXWAIST'],
    'BPXO': ['SEQN', 'BPXOSY1', 'BPXOSY2', 'BPXOSY3', 'BPXODI1', 'BPXODI2', 'BPXODI3'],
    'BPQ': ['SEQN', 'BPQ020', 'BPQ080', 'BPQ050A', 'BPQ150', 'BPQ100D', 'BPQ101D', 'BPQ040A', 'BPQ090D'],
    'SMQ': ['SEQN', 'SMQ020', 'SMQ040'],
    'ALQ': ['SEQN', 'ALQ111', 'ALQ121', 'ALQ130'],
    'PAQ': ['SEQN', 'PAD680'],
    'SLQ': ['SEQN', 'SLD012'],
    'MCQ': ['SEQN', 'MCQ160B', 'MCQ160C', 'MCQ160D', 'MCQ160E', 'MCQ160F'],
    'DIQ': ['SEQN', 'DIQ010', 'DIQ050', 'DIQ070', 'DIQ160'],
    'GHB': ['SEQN', 'LBXGH'],
    'GLU': ['SEQN', 'LBXGLU'],
    'FASTQX': ['SEQN', 'PHAFSTHR']
}

EXPECTED_MISSING = {
    'P_BPQ.XPT': ['BPQ150', 'BPQ101D'],                             # 2017-2020
    'BPQ_L.XPT': ['BPQ050A', 'BPQ100D', 'BPQ040A', 'BPQ090D'],      # 2021-2023
}

FILE_MAP_1720 = {
    'DEMO': 'P_DEMO.XPT',
    'BMX': 'P_BMX.XPT',
    'BPXO': 'P_BPXO.XPT',
    'BPQ': 'P_BPQ.XPT',
    'SMQ': 'P_SMQ.XPT',
    'ALQ': 'P_ALQ.XPT',
    'PAQ': 'P_PAQ.XPT',
    'SLQ': 'P_SLQ.XPT',
    'MCQ': 'P_MCQ.XPT',
    'DIQ': 'P_DIQ.XPT',
    'GHB': 'P_GHB.XPT',
    'GLU': 'P_GLU.XPT',
    'FASTQX': 'P_FASTQX.XPT'
}

FILE_MAP_2123 = {
    'DEMO': 'DEMO_L.XPT',
    'BMX': 'BMX_L.XPT',
    'BPXO': 'BPXO_L.XPT',
    'BPQ': 'BPQ_L.XPT',
    'SMQ': 'SMQ_L.XPT',
    'ALQ': 'ALQ_L.XPT',
    'PAQ': 'PAQ_L.XPT',
    'SLQ': 'SLQ_L.XPT',
    'MCQ': 'MCQ_L.XPT',
    'DIQ': 'DIQ_L.XPT',
    'GHB': 'GHB_L.XPT',
    'GLU': 'GLU_L.XPT',
    'FASTQX': 'FASTQX_L.XPT'
}

RENAME_DICT =  {
    # Metadata
    "SEQN": "respondent_id",
    
    # Demographics
    "RIDAGEYR": "age",
    "RIAGENDR": "sex",
    "RIDRETH3": "race_ethnicity",
    "is_asian": "is_asian_descent",
    "DMDEDUC2": "education_level",
    "INDFMPIR": "income_poverty_ratio",

    # Body Measures
    "BMXWT": "weight_kg",
    "BMXBMI": "body_mass_index",
    "BMXWAIST": "waist_circumference_cm",
    "waist_height_ratio": "waist_height_ratio",

    # Vitals & Medical History
    "sbp_mean": "systolic_bp_mean",
    "dbp_mean": "diastolic_bp_mean",
    "BPQ020": "history_high_bp",
    "bp_med": "taking_bp_meds",
    "BPQ080": "history_high_cholesterol",
    "chol_med": "taking_cholesterol_meds",
    "cvd_history": "history_cvd",

    # Lifestyle & Habits
    "smoking_status": "smoking_status",
    "alcohol_drinks_on_drinking_days": "alcohol_drinks_on_drinking_days",
    "PAD680": "sedentary_minutes_per_day",
    "SLD012": "sleep_hours_weekday",

    # Target
    "target": "undiagnosed_dysglycemia",
    
    # Label-Defining Labs
    "LBXGH": "hba1c_percent",
    "LBXGLU": "fasting_glucose_mg_dl",
    "PHAFSTHR": "fasting_hours"
}

TARGET = 'undiagnosed_dysglycemia'

DROP_COLS = [
    # --- Cycle-specific raw columns (only exist in one cycle) ---
    'BPQ050A', 'BPQ040A', 'BPQ100D', 'BPQ090D',  # 2017-2020 only
    'BPQ150', 'BPQ101D',  # 2021-2023 only

    'BPXOSY1', 'BPXOSY2', 'BPXOSY3',  # -> systolic_bp_mean
    'BPXODI1', 'BPXODI2', 'BPXODI3',  # -> diastolic_bp_mean
    'MCQ160B', 'MCQ160C', 'MCQ160D', 'MCQ160E', 'MCQ160F',  # -> history_cvd
    'SMQ020', 'SMQ040',  # -> smoking_status
    'ALQ111', 'ALQ121', 'ALQ130',  # -> alcohol_drinks_on_drinking_days
    'DMDMARTZ',  # -> marital_status (duplicate copy)
    'BMXHT',  # -> waist_height_ratio

    'DIQ010', 'DIQ050', 'DIQ070', 'DIQ160' # Diabates-related survry questions, used for label
]

META_COLS = [
    'respondent_id', 'cycle', 'race_ethnicity', 'is_asian_descent',
    'hba1c_percent', 'fasting_glucose_mg_dl', 'fasting_hours',
]

NUMERIC_COLS = [
    'age', 'weight_kg', 'body_mass_index', 'waist_circumference_cm', 
    'sedentary_minutes_per_day', 'sleep_hours_weekday', 'waist_height_ratio', 
    'systolic_bp_mean', 'diastolic_bp_mean', 'income_poverty_ratio', 
    'alcohol_drinks_on_drinking_days'
]

CATEGORICAL_COLS = [
    'sex', 'education_level', 'history_high_bp', 'history_high_cholesterol', 
    'marital_status', 'history_cvd', 'taking_bp_meds', 
    'taking_cholesterol_meds', 'smoking_status'
]

MODELS = {
    "LogisticRegression": (LogisticRegression(random_state=RANDOM_STATE, verbose=0, max_iter=1000), "standard"),
    "MLPClassifier": (MLPClassifier(random_state=RANDOM_STATE, max_iter=1000, early_stopping=True), "standard"),
    "GaussianNB": (GaussianNB(), "standard"),
    "KNeighbors": (KNeighborsClassifier(), "standard"),
    "CalibratedSVC": (CalibratedClassifierCV(SVC(verbose=0, random_state=RANDOM_STATE), ensemble=False), "standard"),
    "DecisionTree": (DecisionTreeClassifier(random_state=RANDOM_STATE), "none"),
    "RandomForest": (RandomForestClassifier(random_state=RANDOM_STATE, verbose=0), "none"),
    "XGBoost": (XGBClassifier(random_state=RANDOM_STATE, n_jobs=4), "none")
}