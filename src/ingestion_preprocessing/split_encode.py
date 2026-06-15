from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from src.utils.config import PROJECT_ROOT, log

RAW_DATA_PATH = PROJECT_ROOT / "database" / "raw" / "insurance.csv"
OUTPUT_DIR = PROJECT_ROOT / "database"
TARGET_COLUMN = "charges"


def load_data(file_path: Path) -> pd.DataFrame:
	log("Loading raw dataset from:", file_path)
	df = pd.read_csv(file_path)
	log("Dataset loaded. Shape:", df.shape)
	return df


def map_binary_categoricals(df: pd.DataFrame) -> pd.DataFrame:
	encoded_df = df.copy()

	# Map binary categorical features to 0/1.
	encoded_df["sex"] = encoded_df["sex"].map({"female": 0, "male": 1})
	encoded_df["smoker"] = encoded_df["smoker"].map({"no": 0, "yes": 1})

	if encoded_df[["sex", "smoker"]].isna().any().any():
		raise ValueError("Unexpected values found in sex/smoker columns during mapping")

	return encoded_df


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
	if TARGET_COLUMN not in df.columns:
		raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset")

	X = df.drop(columns=[TARGET_COLUMN])
	y = df[TARGET_COLUMN]

	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=0.2,
		random_state=42,
		shuffle=True,
	)

	return X_train, X_test, y_train, y_test


def normalize_numeric_features(
	X_train: pd.DataFrame,
	X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
	scaler = MinMaxScaler()

	X_train = X_train.copy()
	X_test = X_test.copy()

	X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
	X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

	return X_train, X_test


def one_hot_encode_region(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    encoder = OneHotEncoder(sparse_output=False, drop="first")

    X_train = X_train.copy()
    X_test = X_test.copy()

    region_train = X_train[["region"]]
    region_test = X_test[["region"]]

    region_train_encoded = encoder.fit_transform(region_train)
    region_test_encoded = encoder.transform(region_test)

    # Get feature names for the encoded columns
    feature_names = encoder.get_feature_names_out(["region"])

    # Create dataframe for encoded region features
    region_train_df = pd.DataFrame(
        region_train_encoded, columns=feature_names, index=region_train.index
    )
    region_test_df = pd.DataFrame(
        region_test_encoded, columns=feature_names, index=region_test.index
    )

    # Drop original region column and add encoded columns
    X_train = X_train.drop(columns=["region"])
    X_train = pd.concat([X_train, region_train_df], axis=1)

    X_test = X_test.drop(columns=["region"])
    X_test = pd.concat([X_test, region_test_df], axis=1)

    return X_train, X_test


def split_encode_normalize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
	X_train, X_test, y_train, y_test = split_dataset(df)

	# Apply categorical mapping after split as requested.
	X_train = map_binary_categoricals(X_train)
	X_test = map_binary_categoricals(X_test)

	# Apply one-hot encoding to region (fit on train only)
	X_train, X_test = one_hot_encode_region(X_train, X_test)

	# Fit scaler on train only, then transform both splits.
	X_train, X_test = normalize_numeric_features(X_train, X_test)

	train_encoded = X_train.copy()
	train_encoded[TARGET_COLUMN] = y_train.values

	test_encoded = X_test.copy()
	test_encoded[TARGET_COLUMN] = y_test.values

	return train_encoded, test_encoded


def save_outputs(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	train_path = OUTPUT_DIR / "train_encoded.csv"
	test_path = OUTPUT_DIR / "test_encoded.csv"

	train_df.to_csv(train_path, index=False)
	test_df.to_csv(test_path, index=False)

	log("Saved encoded train dataset to:", train_path)
	log("Saved encoded test dataset to:", test_path)
	log("Train shape:", train_df.shape)
	log("Test shape:", test_df.shape)


def run_pipeline() -> None:
	df = load_data(RAW_DATA_PATH)
	train_encoded, test_encoded = split_encode_normalize(df)
	save_outputs(train_encoded, test_encoded)


if __name__ == "__main__":
	run_pipeline()
