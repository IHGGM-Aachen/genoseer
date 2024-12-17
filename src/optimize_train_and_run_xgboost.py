import os
import numpy as np
import pandas as pd
import sklearn
import statistics
import matplotlib.pyplot as plt
import argparse

import logging
from utils.xgboost_utils import (
    prepare_data_for_fold,
    create_folds,
    optimize_hyperparameters,
    objective,
    get_roc_auc,
)
from utils.parse_config import _parse_config


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
    parser = argparse.ArgumentParser(description="Create protein graph dataset.")
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
        # Create dictionaries for encoding categorical features and tartget
        aa_encoding_dict = {k: v for v, k in enumerate(df_fold.ref_AA.unique())}
        # nuc_encoding_dict = {k:v for v, k in enumerate(df_clinvar.ref.unique())}
        pg_patho_encoding_dict = {k: v for v, k in enumerate(["Benign", "Pathogenic"])}
        # Encoding of categorical features
        df_fold["aa_ref_encoded"] = df_fold["ref_AA"].map(aa_encoding_dict)
        df_fold["aa_alt_encoded"] = df_fold["alt_AA"].map(aa_encoding_dict)
        df_fold["target"] = df_fold["DMS_bin_score"].map(pg_patho_encoding_dict)

        logger.info(f"Distribution of pathogenic variants in fold {i+1}")
        dist_patho = np.round(
            len(df_fold[df_fold["DMS_bin_score"] == "Pathogenic"]) / len(df_fold) * 100,
            2,
        )
        logger.info(f"{dist_patho} %")

        logger.info(f"Distribution of benign variants in fold {i+1}")
        dist_benign = np.round(
            len(df_fold[df_fold["DMS_bin_score"] == "Benign"]) / len(df_fold) * 100, 2
        )
        logger.info(f"{dist_benign} %")

        X = df_fold[
            [
                "aa_ref_encoded",
                "aa_alt_encoded",
                "AA_position",
                "cosine_distance_res",
                "structural_embeddings_res_mt",
                "structural_embeddings_res_wt",
                "AF",
                "mis.oe",
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

    (
        fpr_average_per_fold,
        tpr_average_per_fold,
        roc_auc_average_per_fold,
        y_pred_proba_average_per_fold,
    ) = [[] for _ in range(4)]

    # Calculate ROC AUC for each fold using average parameters
    for i, data_fold in enumerate(data_folds):
        logger.info(f"Starting to evaluate ROC AUC for fold {i+1}!")
        fpr_average, tpr_average, roc_auc_average, y_pred_proba_average = get_roc_auc(
            average_params_across_folds, data_fold
        )
        fpr_average_per_fold.append(fpr_average)
        tpr_average_per_fold.append(tpr_average)
        roc_auc_average_per_fold.append(roc_auc_average)
        y_pred_proba_average_per_fold.append(y_pred_proba_average)

    # Concatenating predictions and calculating the mean ROC curve across all folds
    test_data_accross_all_folds_concat = np.concatenate(
        [data_fold["test"][1] for data_fold in data_folds]
    )
    y_pred_proba_concat = np.concatenate(y_pred_proba_average_per_fold, axis=0)
    fpr_mean_accross_all_folds, tpr_mean_accross_all_folds, _ = (
        sklearn.metrics.roc_curve(
            test_data_accross_all_folds_concat, y_pred_proba_concat[:, 1]
        )
    )

    # Calculate the area under the ROC curve (AUC)
    roc_auc_average_mean = sklearn.metrics.auc(
        fpr_mean_accross_all_folds, tpr_mean_accross_all_folds
    )

    colors = ["#332288", "#117733", "#44AA99", "#88CCEE", "#CC6677"]
    fig_title = f"ROC Curve ({config["scope"]} {config["embedding_size"]})"
    fig_filename = f"roc_curve_{config["scope"]}_{config["embedding_size"]}.png"
    # Plot all the ROC curves
    plt.figure(figsize=(8, 6))
    for i, (fpr_average, tpr_average, roc_auc_average, color) in enumerate(
        zip(
            fpr_average_per_fold, tpr_average_per_fold, roc_auc_average_per_fold, colors
        ),
        start=1,
    ):
        plt.plot(
            fpr_average,
            tpr_average,
            color=color,
            label=f"ROC curve for fold {i} (AUC = {roc_auc_average:.2f})",
        )

    plt.plot(
        fpr_mean_accross_all_folds,
        tpr_mean_accross_all_folds,
        color="red",
        label=f"ROC curve total (AUC = {roc_auc_average_mean:.2f})",
    )
    plt.plot([0, 1], [0, 1], color="red", linestyle="--", label="Random guess")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(fig_title)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(config["output_dir"], fig_filename), dpi=400)

    mean_roc_auc = statistics.mean(roc_auc_average_per_fold)
    std_roc_auc = np.std(roc_auc_average_per_fold)
    logger.info(f"Mean ROC AUC across all folds: {mean_roc_auc:.2f}")
    logger.info(f"Standard Deviation across all folds: {std_roc_auc:.2f}")
