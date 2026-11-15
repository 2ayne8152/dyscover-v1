import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from contextlib import redirect_stdout
from config import *


class EDA:
    """Checks: structure, special codes, leakage, missingness, duplicates, target balance,
    outliers, distributions, correlation. Text goes to eda/eda_log.md, plots to eda/*.png."""

    # ---- helpers for a readable markdown log --------------------------------
    def _section(self, title):
        print(f"\n## {title}\n")

    def _table(self, data):
        """Print a DataFrame (or list of dicts / Series) as an aligned table in a code fence."""
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        print("```")
        print(data.to_string())
        print("```")

    def __init__(self, df: pd.DataFrame, target_col: str):
        self.df = df
        self.target = target_col
        self.num_cols = [c for c in df.columns if df[c].nunique() > 10 and c != target_col]
        self.cat_cols = [c for c in df.columns if df[c].nunique() <= 10 and c != target_col]

    # ---- 1. structure --------------------------------------------------------
    def structural_overview(self):
        self._section("Structural overview")
        print(f"Shape: {self.df.shape}  "
              f"({len(self.num_cols)} numeric, {len(self.cat_cols)} categorical, 1 target)\n")

        print("Data types:\n")
        self._table(self.df.dtypes.astype(str).value_counts().rename("n_columns").to_frame())

        print("\nSummary of numeric features (with skew):\n")
        desc = self.df[self.num_cols].describe().T
        desc['skew'] = self.df[self.num_cols].skew()
        self._table(desc.round(2))

    # ---- 2. special codes ----------------------------------------------------
    def special_code_check(self):
        self._section("Leftover special codes (7, 9, 77, 99, 777, 999, 7777, 9999)")
        special_codes = [7, 9, 77, 99, 777, 999, 7777, 9999]
        records = []
        for col in self.cat_cols:
            n = int(self.df[col].isin(special_codes).sum())
            if n > 0:
                records.append({'feature': col, 'count': n})
        if records:
            self._table(pd.DataFrame(records))
        else:
            print("None found in categorical columns.")

    # ---- 3. leakage and missingness -----------------------------------------
    def leakage_check(self):
        self._section("Leakage check")
        suspects = ['hba1c', 'glucose', 'fasting', 'DIQ']
        leaks = [col for col in self.df.columns if any(s.lower() in col.lower() for s in suspects)]
        print(f"Potential leakage columns: {leaks if leaks else 'None'}")

    def missingness(self):
        self._section("Missingness")
        missing = pd.DataFrame({
            'count': self.df.isna().sum(),
            'pct': (self.df.isna().mean() * 100).round(2),
        }).sort_values('pct', ascending=False)
        self._table(missing)

        heavy = missing[missing['pct'] > 50.0]
        print()
        if heavy.empty:
            print("No columns are missing more than 50% of their values.")
        else:
            print("Columns missing more than 50% (consider dropping):")
            self._table(heavy)

    # ---- 4. duplicates and target -------------------------------------------
    def duplication(self):
        self._section("Duplicates and leftover ID columns")
        n_dupes = int(self.df.duplicated().sum())
        sus_id = [c for c in self.df.columns if 'id' in c.lower() or 'seqn' in c.lower()]
        print(f"Duplicate rows: {n_dupes}")
        print(f"Leftover ID columns: {sus_id if sus_id else 'None'}")

    def target_balance(self):
        self._section("Target distribution")
        counts = self.df[self.target].value_counts()
        balance = pd.DataFrame({
            'count': counts,
            'pct': (counts / counts.sum() * 100).round(2),
        })
        self._table(balance)

    # ---- 5. outliers ---------------------------------------------------------
    def outliers(self):
        self._section("Outliers (IQR rule)")
        records = []
        for col in self.num_cols:
            s = self.df[col].dropna()
            if len(s) == 0:
                continue
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_out = int(((s < lower) | (s > upper)).sum())
            records.append({'feature': col, 'count': n_out,
                            'pct': round(n_out / len(self.df) * 100, 2)})
        self._table(pd.DataFrame(records).sort_values('pct', ascending=False))

    # ---- 6. plots ------------------------------------------------------------
    def numeric_distributions_plot(self):
        cols = 3
        rows = math.ceil(len(self.num_cols) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), squeeze=False)
        axes = axes.flatten()

        for i, col in enumerate(self.num_cols):
            sns.histplot(self.df[col].dropna(), kde=True, bins=20, color='steelblue', ax=axes[i])
            axes[i].set_title(f"Distribution of {col}")

        for j in range(len(self.num_cols), len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.savefig(OUT_DIR / "numeric_distribution.png", dpi=200, bbox_inches='tight')
        plt.close(fig)

    def categorical_distributions_plot(self):
        cols = 3
        rows = math.ceil(len(self.cat_cols) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), squeeze=False)
        axes = axes.flatten()

        for i, col in enumerate(self.cat_cols):
            data = self.df[col].value_counts(dropna=False, normalize=True)
            sns.barplot(x=data.index.astype(str), y=data.values, color='steelblue', ax=axes[i])
            axes[i].set_title(f"Distribution of {col}")
            axes[i].tick_params(axis='x', rotation=45)

        for j in range(len(self.cat_cols), len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.savefig(OUT_DIR / "categorical_distribution.png", dpi=200, bbox_inches='tight')
        plt.close(fig)

    def plot_correlation(self):
        plt.figure(figsize=(12, 8))
        corr = self.df[self.num_cols].corr(method='spearman')
        sns.heatmap(corr, cmap="Spectral", center=0, annot=True, vmin=-1, vmax=1,
                    fmt='.3f', annot_kws={'size': 9})
        plt.title("Numeric Feature Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "correlation_heatmap.png", bbox_inches='tight')
        plt.close()

    # ---- run everything ------------------------------------------------------
    def run_all(self):
        with open(OUT_DIR / "eda_log.md", "w", encoding="utf-8") as f:
            with redirect_stdout(f):
                print("# EDA log")
                self.structural_overview()
                self.special_code_check()
                self.leakage_check()
                self.missingness()
                self.duplication()
                self.target_balance()
                self.outliers()
        # plots are saved as files, so they don't need to be inside the redirect
        self.numeric_distributions_plot()
        self.categorical_distributions_plot()
        self.plot_correlation()
        print(f"EDA done. See {OUT_DIR.resolve()}")


if __name__ == '__main__':
    from NHANES import NHANES
    nh = NHANES()
    X_train, _, y_train, _, _, _ = nh.load_data_a()
    eda = EDA(X_train.assign(target=y_train), target_col='target')
    eda.run_all()