"""
Main pipeline entry point.

Execution order:
  1. Preprocessing  – split & encode the raw dataset
  2. Modeling       – train LR, EN, RFR, XGBR and select the champion
"""

from src.utils.config import log
from src.ingestion_preprocessing.split_encode import run_pipeline as run_split_encode
from src.modeling.lr import run_linear_regression
from src.modeling.en import run_elastic_net
from src.modeling.rfr import run_random_forest
from src.modeling.xgbr import run_xgboost_regressor
from src.modeling.champ import run_champion_selection


def main() -> None:
    # ------------------------------------------------------------------ #
    # Step 1 – Preprocessing
    # ------------------------------------------------------------------ #
    log("=" * 60)
    log("STEP 1: Preprocessing – split & encode dataset")
    log("=" * 60)
    run_split_encode()

    # ------------------------------------------------------------------ #
    # Step 2 – Model training
    # ------------------------------------------------------------------ #
    log("=" * 60)
    log("STEP 2: Training individual models")
    log("=" * 60)

    log("--- Linear Regression ---")
    run_linear_regression()

    log("--- Elastic Net ---")
    run_elastic_net()

    log("--- Random Forest Regressor ---")
    run_random_forest()

    log("--- XGBoost Regressor ---")
    run_xgboost_regressor()

    # ------------------------------------------------------------------ #
    # Step 3 – Champion selection & test evaluation
    # ------------------------------------------------------------------ #
    log("=" * 60)
    log("STEP 3: Champion selection & test-set evaluation")
    log("=" * 60)
    result = run_champion_selection()
    log(
        "Pipeline complete. Champion:",
        result["champion"]["model_name"],
        "| AMSE:", result["champion"]["AMSE"],
    )


if __name__ == "__main__":
    main()
