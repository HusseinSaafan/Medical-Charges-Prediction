from pathlib import Path
from typing import List, Optional

import pandas as pd
from scipy.stats import pearsonr, ttest_ind

from src.utils.config import PROJECT_ROOT, log

TARGET_COLUMN = "charges"
DATASET_PATH = PROJECT_ROOT / "database" / "raw" / "insurance.csv"
ALPHA = 0.05


def load_dataset(file_path: Path) -> pd.DataFrame:
	log("Loading dataset from:", file_path)
	df = pd.read_csv(file_path)
	log("Dataset loaded successfully. Shape:", df.shape)
	return df


def pearson_tests(df: pd.DataFrame, target_col: str = TARGET_COLUMN, alpha: float = ALPHA) -> None:
	continuous_features = [
		col
		for col in df.select_dtypes(include=["number"]).columns
		if col != target_col
	]

	log("\nPearson Correlation Tests (continuous features vs target):")
	for feature in continuous_features:
		valid_rows = df[[feature, target_col]].dropna()
		corr, p_value = pearsonr(valid_rows[feature], valid_rows[target_col])
		significant = p_value < alpha

		log(
			f"Feature: {feature} | Pearson r: {corr:.4f} | p-value: {p_value:.6g} | "
			f"Significant (alpha={alpha}): {significant}"
		)


def t_tests_for_binary_categoricals(
	df: pd.DataFrame,
	target_col: str = TARGET_COLUMN,
	categorical_features: Optional[List[str]] = None,
	alpha: float = ALPHA,
) -> None:
	if categorical_features is None:
		categorical_features = ["sex", "smoker"]

	log("\nIndependent Two-Sample t-tests (binary categorical features vs target):")

	for feature in categorical_features:
		if feature not in df.columns:
			log(f"Skipping {feature}: not found in dataset columns")
			continue

		valid_rows = df[[feature, target_col]].dropna()
		groups = valid_rows.groupby(feature)[target_col]

		if len(groups) != 2:
			log(
				f"Skipping {feature}: expected exactly 2 groups, found {len(groups)} groups"
			)
			continue

		group_names = list(groups.groups.keys())
		sample_a = groups.get_group(group_names[0])
		sample_b = groups.get_group(group_names[1])

		t_stat, p_value = ttest_ind(sample_a, sample_b, equal_var=False)
		significant = p_value < alpha

		log(
			f"Feature: {feature} | Groups: {group_names[0]} vs {group_names[1]} | "
			f"t-stat: {t_stat:.4f} | p-value: {p_value:.6g} | "
			f"Significant (alpha={alpha}): {significant}"
		)


def run_statistical_tests() -> None:
	df = load_dataset(DATASET_PATH)
	pearson_tests(df)
	t_tests_for_binary_categoricals(df)


if __name__ == "__main__":
	run_statistical_tests()
