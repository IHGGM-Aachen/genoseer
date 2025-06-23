import os
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import argparse

import logging
from utils.xgboost_utils import (
    prepare_data_for_fold,
    create_folds,
    optimize_hyperparameters,
    objective,
    prepare_data,
)
from utils.parse_config import _parse_config

from xgboost import XGBClassifier


def setup_logging():
    log_dir = "./src/logs/"
    os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger_handler = logging.FileHandler(os.path.join(log_dir, "xgboost.log"))
    logger_handler.setLevel(logging.INFO)
    logger_handler.setFormatter(formatter)

    xgboost_logger = logging.getLogger("XGBoostLogger")
    xgboost_logger.setLevel(logging.INFO)
    xgboost_logger.addHandler(logger_handler)

    return xgboost_logger


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train and validate XGBoost Model.")
    parser.add_argument("--config_path", type=str, help="Path to the config yaml")
    return parser.parse_args()


if __name__ == "__main__":

    # Parse arguments and config file
    args = parse_arguments()
    config = _parse_config(args.config_path)

    # Setup logger to monitor the training process
    logger = setup_logging()

    features_all_folds = []
    labels_all_folds = []

    fold_files = [
        os.path.join(config["input_dir_folds"], file)
        for file in os.listdir(config["input_dir_folds"])
    ]

    logger.info("Loading folds!")
    for i, fold_file in enumerate(fold_files):
        df_fold = pd.read_parquet(fold_file)

        logger.info(f"Distribution of pathogenic variants in fold {i+1}")
        dist_patho = np.round(
            len(df_fold[df_fold["target"] == 1]) / len(df_fold) * 100,
            2,
        )
        logger.info(f"{dist_patho} %")

        logger.info(f"Distribution of benign variants in fold {i+1}")
        dist_benign = np.round(
            len(df_fold[df_fold["target"] == 0]) / len(df_fold) * 100, 2
        )
        logger.info(f"{dist_benign} %")

        X = df_fold[
            [
                "cosine_distance",
                "structural_embeddings_wt",
                "structural_embeddings_mt",
                "cadd",
                "node_embedding_affected_wt",
                "node_embedding_affected_mt",
                "cosine_distance_node_embedding_affected",
            ]
        ]
        y = df_fold["target"]

        features_all_folds.append(X)
        labels_all_folds.append(y)

    # Define the folds indices
    folds_indices = [
        {"train": [0, 1, 2], "val": 3, "test": 4},
        {"train": [0, 1, 4], "val": 2, "test": 3},
        {"train": [0, 3, 4], "val": 1, "test": 2},
        {"train": [2, 3, 4], "val": 0, "test": 1},
        {"train": [1, 2, 3], "val": 4, "test": 0},
    ]

    # Prepare data for all folds
    logger.info("Preparing folds!")
    data_folds = []
    for fold_indices in folds_indices:
        # Generate datasets for the current fold
        (
            folding_training,
            folding_validation,
            folding_testing,
            folding_training_y,
            folding_validation_y,
            folding_testing_y,
        ) = create_folds(features_all_folds, labels_all_folds, fold_indices)
        # Prepare and store results for the current fold
        data_fold = prepare_data_for_fold(
            folding_training,
            folding_validation,
            folding_testing,
            folding_training_y,
            folding_validation_y,
            folding_testing_y,
        )
        data_folds.append(data_fold)

    logger.info(f"Length of data folds: {len(data_folds)}")
    best_params_per_fold = []

    # Hyperparameter optimization for each fold
    for i, data_fold in enumerate(data_folds):
        logger.info(f"Hyperparameter optimization for fold {i+1} started!")
        best_params, best_accuracy = optimize_hyperparameters(
            data_fold,
            objective=objective,
            n_trials=config["trials"],
        )
        best_params_per_fold.append(best_params)

        logger.info(f"Best parameters for fold {i+1}: {best_params}")
        logger.info(f"Best accuracy for fold {i+1}: {best_accuracy}")
        
        if i == 1:
            break

    # Calculate average hyperparameters across all folds
    average_params_across_folds = {}
    for param in best_params_per_fold[0].keys():
        if isinstance(best_params_per_fold[0][param], str):
            average_params_across_folds[param] = best_params_per_fold[0][param]
        elif param in ["n_estimators", "max_depth"]:
            average_params_across_folds[param] = int(
                np.mean([fold[param] for fold in best_params_per_fold])
            )
        else:
            average_params_across_folds[param] = np.mean(
                [fold[param] for fold in best_params_per_fold]
            )

    logger.info(f"Average parameters across folds: {average_params_across_folds}")


    ######################## Final Testing #######################
    # Load hold out dataset for testing
    logger.info("Starting to test XGBoost Model on hold out testset.")
    
    df_test_hold_out = pd.read_parquet(config["hold_out_testset"])

    X_test_hold_out = df_test_hold_out[
        [
            "cosine_distance",
            "structural_embeddings_wt",
            "structural_embeddings_mt",
            "cadd",
            "node_embedding_affected_wt",
            "node_embedding_affected_mt",
            "cosine_distance_node_embedding_affected",
        ]
    ]
    y_test_hold_out = df_test_hold_out["target"]

    # Prepare hold out dataset for testing
    X_test_hold_out, y_test_hold_out = prepare_data(X_test_hold_out, y_test_hold_out)

    # Prepare data from folds for training and validation of averaged XGBoost model
    X_train = pd.concat(features_all_folds[:-1])
    X_val = features_all_folds[-1]

    y_train = pd.concat(labels_all_folds[:-1])
    y_val = labels_all_folds[-1]
    
    X_train, y_train = prepare_data(X_train, y_train)
    X_val, y_val = prepare_data(X_val, y_val)

    xgb_model_average_params = XGBClassifier(
        **average_params_across_folds, device="cuda", predictor="gpu_predictor"
    )
    xgb_model_average_params.fit(
        X_train, y_train, eval_set=[(X_val, y_val)], verbose=True
    )

    # Get the model predictions for the test set
    y_preds = xgb_model_average_params.predict(X_test_hold_out)
    logger.info("Classification Report for Model Predictions on Test Set:")
    sklearn.metrics.classification_report(y_test_hold_out, y_preds)

    # Calculate probabilities
    y_preds_proba = xgb_model_average_params.predict_proba(X_test_hold_out)

    # Calculate the false positive rate, true positive rate, and thresholds
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(
        y_test_hold_out, y_preds_proba[:, 1]
    )

    # Calculate and log the area under the ROC curve (AUC)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    logger.info(f"Area Under the ROC Curve (AUC): {roc_auc}")

    # Calculate and log the Matthews Correlation Coefficient
    mcc_value = sklearn.metrics.matthews_corrcoef(y_test_hold_out, y_preds)
    logger.info(f"Matthews Correlation Coefficient: {mcc_value}")

    # Calculate and log the accuracy
    accuracy = sklearn.metrics.accuracy_score(y_test_hold_out, y_preds)
    logger.info(f"Accuracy Score {accuracy}")

    precision, recall, _ = sklearn.metrics.precision_recall_curve(
        y_test_hold_out, y_preds_proba[:, 1]
    )
    average_precision = sklearn.metrics.average_precision_score(
        y_test_hold_out, y_preds_proba[:, 1]
    )
    logger.info(f"Average Precision Score: {average_precision}")

    fig_title = f"ROC Curve ({config['scope']} {config['embedding_size']})"
    fig_filename = f"roc_curve_{config['scope']}_{config['embedding_size']}.png"
    # Plot all the ROC curves
    plt.figure(figsize=(8, 6))

    plt.plot(
        fpr,
        tpr,
        color="#332288",
        label=f"ROC curve (AUC = {roc_auc:.2f})",
    )

    plt.plot([0, 1], [0, 1], color="red", linestyle="--", label="Random guess")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(fig_title)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(config["output_dir"], fig_filename), dpi=400)

    fig_title = f"Precision-Recall Curve ({config['scope']} {config['embedding_size']})"
    fig_filename = (
        f"precision_recall_curve_{config['scope']}_{config['embedding_size']}.png"
    )
    # Plot all the ROC curves
    plt.figure(figsize=(8, 6))

    plt.plot(
        recall,
        precision,
        color="#332288",
        label=f"{config['scope']} {config['embedding_size']}",
    )

    plt.plot([0, 1], [0.5, 0.5], color="red", linestyle="--", label="Random guess")
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="upper right")

    plt.savefig(os.path.join(config["output_dir"], fig_filename), dpi=400)
