import numpy as np
import pandas as pd
from config import *

class NHANES:
    def __init__(self):
        self.flow = []

    def _log(self, cycle, step, df):
        self.flow.append({'cycle': cycle, 'step':step, 'n':len(df)})

    def read_path(self, path, file_type):
        try:
            df = pd.read_sas(DATA_DIR/path, encoding='utf-8')
        except FileNotFoundError:
            raise FileNotFoundError(f"Missing raw file: {DATA_DIR/path}.")

        df.columns = [col.decode('utf-8') if isinstance(col, bytes) else str(col) for col in df.columns]
        num = df.select_dtypes('number').columns
        df[num] = df[num].mask(df[num].abs() < 1e-30, 0.0)

        missing = [col for col in COLUMNS[file_type] if col not in df.columns]
        unexpected = [c for c in missing if c not in EXPECTED_MISSING.get(path, set())]
        if unexpected: raise ValueError(f"Unexpected missing columns {unexpected} in {path}")

        for col in missing: df[col] = np.nan

        keep_cols = COLUMNS[file_type].copy()
        return df[keep_cols].copy()

    def replace_missing_codes(self, df):
        """Converts NHANES Refused (7) and Don't Know (9) codes to np.nan."""
        # 1-digit codes
        cols_1_digit = ['DMDEDUC2', 'BPQ020', 'BPQ050A', 'BPQ150', 'BPQ080', 'BPQ100D', 'BPQ101D', 'BPQ090D', 'BPQ040A',
                        'SMQ020', 'SMQ040', 'ALQ111', 'MCQ160B', 'MCQ160C',
                        'MCQ160D', 'MCQ160E', 'MCQ160F', 'DIQ010', 'DIQ050', 'DIQ070', 'DIQ160']
        for col in cols_1_digit:
            if col in df.columns:
                df[col] = df[col].replace({7.0: np.nan, 9.0: np.nan})

        # 2-digit codes
        cols_2_digit = ['ALQ121', 'SLD012', 'DMDMARTZ']
        for col in cols_2_digit:
            if col in df.columns:
                df[col] = df[col].replace({77.0: np.nan, 99.0: np.nan})

        # 3-digit and 4-digit codes
        if 'ALQ130' in df.columns:
            df['ALQ130'] = df['ALQ130'].replace({777.0: np.nan, 999.0: np.nan})
        if 'PAD680' in df.columns:
            df['PAD680'] = df['PAD680'].replace({7777.0: np.nan, 9999.0: np.nan})

        return df

    def merge_files(self, dfs_dict):
        """Left merges all dataframes sequentially onto the DEMO base file."""
        df_master = dfs_dict['DEMO'].copy()

        for key, df_right in dfs_dict.items():
            if key == 'DEMO':
                continue
            # Strict 1-to-1 Left Join ensures no row duplication and no orphaned records
            df_master = df_master.merge(df_right, on='SEQN', how='left', validate='one_to_one')

        return df_master

    def derive_demographics(self, df):
        # 6:Non-hispanic Asian
        df['is_asian'] = np.where(df['RIDRETH3'] == 6, 1, 0)
        df['waist_height_ratio'] = df['BMXWAIST'] / df['BMXHT']

        if 'DMDMARTZ' in df.columns:
            df['marital_status'] = df['DMDMARTZ']
        else:
            df['marital_status'] = np.nan

        return df

    def derive_medical_history(self, df):
        # Blood Pressure Means
        df['sbp_mean'] = df[['BPXOSY1', 'BPXOSY2', 'BPXOSY3']].mean(axis=1)
        df['dbp_mean'] = df[['BPXODI1', 'BPXODI2', 'BPXODI3']].mean(axis=1)

        # Composite cardiovascular (CVD) history (1 if any are 1)
        cvd_cols = ['MCQ160B', 'MCQ160C', 'MCQ160D', 'MCQ160E', 'MCQ160F']
        all_cvd_missing = df[cvd_cols].isna().all(axis=1)
        any_cvd_yes = (df[cvd_cols] == 1).any(axis=1)

        df['cvd_history'] = np.select(
            [all_cvd_missing, any_cvd_yes],
            [np.nan, 1.0],
            default=0.0  # Not missing AND Not True
        )
        return df

    def derive_medical_condition(self, df):
        # Blood Pressure Medication
        bp_med_cond = [
            (df['BPQ020'] == 2),  # Unmedicated (Never told)
            (df['cycle'] == '2021-2023') & (df['BPQ150'] == 1),
            (df['cycle'] == '2021-2023') & (df['BPQ150'] == 2),
            (df['cycle'] == '2017-2020') & (df['BPQ050A'] == 1),
            (df['cycle'] == '2017-2020') & ((df['BPQ040A'] == 2) | (df['BPQ050A'] == 2))
        ]
        df['bp_med'] = np.select(bp_med_cond, [0, 1, 0, 1, 0], default=np.nan)

        # Cholesterol Medication
        chol_med_cond = [
            (df['BPQ080'] == 2),  # Unmedicated (Never told)
            (df['cycle'] == '2021-2023') & (df['BPQ101D'] == 1),
            (df['cycle'] == '2021-2023') & (df['BPQ101D'] == 2),
            (df['cycle'] == '2017-2020') & (df['BPQ100D'] == 1),
            (df['cycle'] == '2017-2020') & ((df['BPQ090D'] == 2) | (df['BPQ100D'] == 2))
        ]
        df['chol_med'] = np.select(chol_med_cond, [0, 1, 0, 1, 0], default=np.nan)

        return df

    def derive_habit(self, df):
        # Smoking Status
        smk_conditions = [
            (df['SMQ020'] == 1) & (df['SMQ040'].isin([1, 2])),
            (df['SMQ020'] == 1) & (df['SMQ040'] == 3),
            (df['SMQ020'] == 2)
        ]
        df['smoking_status'] = np.select(smk_conditions, [1, 2, 3], default=np.nan)

        # Alcohol
        df['alcohol_drinks_on_drinking_days'] = np.where((df['ALQ111'] == 2) | (df['ALQ121'] == 0), 0.0, df['ALQ130'])
        return df

    def derive_targets(self, df):
        # 1. Define Laboratory Truths
        lab_dm = (df['LBXGH'] >= 6.5) | ((df['LBXGLU'] >= 126) & (df['PHAFSTHR'] >= 8))
        lab_pre = (~lab_dm) & (
                ((df['LBXGH'] >= 5.7) & (df['LBXGH'] < 6.5)) |
                ((df['LBXGLU'] >= 100) & (df['LBXGLU'] < 126) & (df['PHAFSTHR'] >= 8))
        )

        # 2. Create the Binary Target: 1 = Undiagnosed Dysglycemia, 0 = Normal
        # We only assign 1 or 0 if they explicitly answered "No" to having diabetes (DIQ010 == 2)
        eligible = (df['DIQ010'] == 2) & (df['DIQ160'] != 1) & (df['DIQ050'] != 1) & (df['DIQ070'] != 1)
        conditions = [
            eligible & (lab_dm | lab_pre),  # 1: Undiagnosed (Pre-DM or DM)
            eligible & (~lab_dm) & (~lab_pre)  # 0: Strictly Normal
        ]

        # Call it 'target' (or 'y_undx_dysgly' if you prefer)
        df['target'] = np.select(conditions, [1.0, 0.0], default=np.nan)

        # 3. Drop the known cases and people with missing DIQ010 responses
        # By dropping NaNs in 'target', we isolate the SCREEN-ELIGIBLE population!
        cycle = df['cycle'].iloc[0]
        df = df.dropna(subset=['target']).copy()
        df['target'] = df['target'].astype(int)
        self._log(cycle, 5, df)
        return df

    def derive_all(self, df):
        df = self.derive_demographics(df)
        df = self.derive_medical_history(df)
        df = self.derive_medical_condition(df)
        df = self.derive_habit(df)
        df = self.derive_targets(df)
        df = df.drop(columns=DROP_COLS, errors='ignore')
        df = df.rename(columns=RENAME_DICT)
        return df

    def load_raw_data(self, path_dict, cycle):
        dfs = {}
        for name, path in path_dict.items():
            dfs[name] = self.read_path(path, name)

        # Merge all dataframes into a single master df
        df = self.merge_files(dfs)
        df['cycle'] = cycle
        df = self.replace_missing_codes(df)
        self._log(cycle, 1, df)

        # Excluding respondents under 18 and pregnant respondents
        df = df[df['RIDAGEYR'] >= 18].copy()
        self._log(cycle, 2, df)

        if 'RIDEXPRG' in df.columns:
            df = df[df['RIDEXPRG'] != 1.0].copy()
        self._log(cycle, 3, df)

        # Drop missing HbA1c
        df = df.dropna(subset=['LBXGH']).copy()
        self._log(cycle, 4, df)

        # Drop pregnancy flag as it is no longer needed
        df = df.drop(columns=['RIDEXPRG'], errors='ignore')

        # Cache the result
        if cycle == '2017-2020':
            df.reset_index(drop=True).to_feather(CACHE_1720)
        elif cycle == '2021-2023':
            df.reset_index(drop=True).to_feather(CACHE_2123)
        else:
            raise ValueError(f"Unknown cycle: {cycle}")

        return df

    def get_xy(self, df):
        features = [
            c for c in df.columns
            if c!= TARGET
            and c not in META_COLS
        ]
        return df[features].copy(), df[TARGET].copy()

    def load_data_a(self, force_rebuild=False):
        if CACHE_1720.exists() and not force_rebuild:
            print("--- Loading 2017-2020 data from cache")
            df_1720 = pd.read_feather(CACHE_1720)
            self.flow = pd.read_json(CACHE_DIR / 'flow.json').to_dict('records')

        else:
            print("--- Loading 2017-2020 RAW data")
            df_1720 = self.load_raw_data(FILE_MAP_1720, '2017-2020')

        if CACHE_2123.exists() and not force_rebuild:
            print("--- Loading 2021-2023 data from cache")
            df_2123 = pd.read_feather(CACHE_2123)
            self.flow = pd.read_json(CACHE_DIR / 'flow.json').to_dict('records')

        else:
            print("--- Loading 2021-2023 RAW data")
            df_2123 = self.load_raw_data(FILE_MAP_2123, '2021-2023')

        # Apply derivations sequentially to 2017-2020 & 2021-2023
        df_1720 = self.derive_all(df_1720)
        df_2123 = self.derive_all(df_2123)

        (pd.DataFrame(self.flow)
         .drop_duplicates(['cycle', 'step'], keep='last')
         .to_json(CACHE_DIR / 'flow.json', orient='records')
         )
        X_train, y_train = self.get_xy(df_1720)
        X_test, y_test = self.get_xy(df_2123)

        return X_train, X_test, y_train, y_test, df_1720, df_2123

if __name__ == '__main__':
    """nh = NHANES()
    X_train, X_test, y_train, y_test, df_1720, df_2123 = nh.load_data_a(force_rebuild=True)

    print(X_train.info())
    print(X_test.info())"""

    X_train, X_test, y_train, y_test, df_1720, df_2123 = NHANES().load_data_a(True)
    print(X_train.shape, X_test.shape)                  # both should have 20 columns
    print(set(META_COLS) <= set(df_1720.columns))       # True: kept in the full frame
    print(set(META_COLS).isdisjoint(X_train.columns))   # True: not in X
    print(df_1720["is_asian_descent"].sum(), df_1720.loc[df_1720["is_asian_descent"] == 1, TARGET].sum())

    print()
    print(df_1720.info())
    #print(df_1720.describe())