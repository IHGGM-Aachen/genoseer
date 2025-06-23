import numpy as np
import pandas as pd
import optuna
from xgboost import XGBClassifier
from functools import partial
from sklearn.metrics import accuracy_score
import sklearn

import typing as ty


def prepare_data(x: pd.DataFrame, y: pd.DataFrame) -> ty.Tuple[np.ndarray]:
    """
    Prepares the data for the XGBoost model by encoding and stacking feature columns and reshaping target columns.

    Args:
        x (pd.DataFrame): DataFrame containing feature columns.
        y (pd.DataFrame): DataFrame or Series containing the target variable.

    Returns:
        tuple: A tuple containing the features and target arrays.
    """
    # Encode and reshape feature columns
    cosine_distance = np.array(x["cosine_distance"]).reshape(-1, 1)
    structural_embeddings_mut = np.vstack(x["structural_embeddings_mt"].to_numpy())
    structural_embeddings_wt = np.vstack(x["structural_embeddings_wt"].to_numpy())

    node_cosine_distance = np.array(
        x["cosine_distance_node_embedding_affected"]
    ).reshape(-1, 1)
    node_embeddings_mut = np.vstack(x["node_embedding_affected_mt"].to_numpy())
    node_embeddings_wt = np.vstack(x["node_embedding_affected_wt"].to_numpy())

    cadd = np.array(x["cadd"]).reshape(-1, 1)

    # Combine all features
    features = np.hstack(
        (
            cadd,
            structural_embeddings_mut,
            structural_embeddings_wt,
            cosine_distance,
            node_embeddings_mut,
            node_embeddings_wt,
            node_cosine_distance,
        )
    )

    target = np.array(y).reshape(-1, 1)

    return features, target


# Function to create training, validation, and testing datasets based on folding
def create_folds(
    features_all_folds: ty.List[pd.DataFrame],
    labels_all_folds: ty.List[pd.DataFrame],
    fold_indices: ty.Dict[str, ty.Union[ty.List[int], int]],
) -> ty.Tuple[pd.DataFrame]:
    """
    Create training, validation, and testing datasets based on provided fold indices.

    Args:
    features_all_folds (List[pd.DataFrame]): DataFrames containing features for each fold.
    labels_all_folds (List[pd.DataFrame]): Series containing labels for each fold.
    fold_indices (Dict[str, Union[List[int], int]]): Indices for 'train', 'val', and 'test' folds.

    Returns:
    Tuple[pd.DataFrame]: The training, validation, and testing datasets and their corresponding labels.
    """
    folding_training = pd.concat(
        [features_all_folds[i] for i in fold_indices["train"]]
    ).reset_index(drop=True)
    folding_validation = features_all_folds[fold_indices["val"]].reset_index(drop=True)
    folding_testing = features_all_folds[fold_indices["test"]].reset_index(drop=True)

    folding_training_y = pd.concat(
        [labels_all_folds[i] for i in fold_indices["train"]]
    ).reset_index(drop=True)
    folding_validation_y = labels_all_folds[fold_indices["val"]].reset_index(drop=True)
    folding_testing_y = labels_all_folds[fold_indices["test"]].reset_index(drop=True)

    return (
        folding_training,
        folding_validation,
        folding_testing,
        folding_training_y,
        folding_validation_y,
        folding_testing_y,
    )


def prepare_data_for_fold(
    folding_training: pd.DataFrame,
    folding_validation: pd.DataFrame,
    folding_testing: pd.DataFrame,
    folding_training_y: pd.DataFrame,
    folding_validation_y: pd.DataFrame,
    folding_testing_y: pd.DataFrame,
) -> ty.Dict[str, ty.List[np.ndarray]]:
    """
    Prepare and return data for the specified fold by calling the _prepare_data helper function.

    Args:
    folding_training (pd.DataFrame): DataFrame of training features.
    folding_validation (pd.DataFrame): DataFrame of validation features.
    folding_testing (pd.DataFrame): DataFrame of testing features.
    folding_training_y (pd.DataFrame): Series of training labels.
    folding_validation_y (pd.DataFrame): Series of validation labels.
    folding_testing_y (pd.DataFrame): Series of testing labels.

    Returns:
    Dict[str, List[np.ndarray]]: Prepared training, validation, and testing data.
    """
    train_X, train_y = prepare_data(folding_training, folding_training_y)
    val_X, val_y = prepare_data(folding_validation, folding_validation_y)
    test_X, test_y = prepare_data(folding_testing, folding_testing_y)

    return {
        "train": [train_X, train_y],
        "val": [val_X, val_y],
        "test": [test_X, test_y],
    }


def objective(trial: optuna.trial.Trial, data: pd.DataFrame) -> float:
    """
    Optuna objective function for hyperparameter optimization of XGBoost.

    Args:
        trial (optuna.trial.Trial): A single evaluation of hyperparameters.
        data (pd.DataFrame): Data dictionary containing 'train', 'val', and 'test' datasets.

    Returns:
        float: Accuracy score for the trial.
    """
    # Suggest hyperparameters
    param = {
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.1),
        "n_estimators": trial.suggest_int("n_estimators", 1000, 2500),
        "max_depth": trial.suggest_int("max_depth", 10, 20),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_loguniform("gamma", 0.01, 1),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 0.01, 100),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 0.01, 100),
        "objective": "binary:logistic",
        "random_state": 42,
        "early_stopping_rounds": 100,
    }

    # Fit model with suggested hyperparameters
    model = XGBClassifier(**param, device="cuda", predictor="gpu_predictor")
    model.fit(
        data["train"][0],
        data["train"][1],
        eval_set=[(data["val"][0], data["val"][1])],
        verbose=False,
    )

    # Predict and evaluate accuracy
    preds = model.predict(data["test"][0])
    accuracy = accuracy_score(data["test"][1], preds)

    return accuracy


def get_roc_auc(params: ty.Dict[str, ty.Any], data: ty.Dict[str, np.array]):
    """
    Evaluates the ROC AUC of the trained XGBoost model.

    Args:
        params (dict): Hyperparameters of the XGBoost model.
        data (dict): Data dictionary containing 'train', 'val', and 'test' datasets.

    Returns:
        tuple: False positive rate, true positive rate, ROC AUC score, and predicted probabilities.
    """
    # Fit model with provided parameters
    xgboost = XGBClassifier(**params, device="cuda", predictor="gpu_predictor")
    xgboost.fit(
        data["train"][0],
        data["train"][1],
        eval_set=[(data["val"][0], data["val"][1])],
        verbose=False,
    )

    # Calculate predicted probabilities
    y_pred_proba_average = xgboost.predict_proba(data["test"][0])

    # Calculate ROC curve metrics
    fpr_average, tpr_average, _ = sklearn.metrics.roc_curve(
        data["test"][1], y_pred_proba_average[:, 1]
    )

    # Calculate ROC AUC
    roc_auc_average = sklearn.metrics.auc(fpr_average, tpr_average)

    return fpr_average, tpr_average, roc_auc_average, y_pred_proba_average


def optimize_hyperparameters(data, objective, n_trials: int = 10):
    """
    Optimizes hyperparameters using Optuna and returns the best parameters and accuracy.

    Args:
        data (dict): Data dictionary containing 'train', 'val', and 'test' datasets.
        objective (function): Objective function to be optimized.
        n_trials (int): Number of optimization trials. Default is 10.

    Returns:
        tuple: Best hyperparameters and accuracy score.
    """
    objective = partial(objective, data=data)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    # Fit model with best found parameters
    xgb_model_best = XGBClassifier(**study.best_params)
    xgb_model_best.fit(
        data["train"][0],
        data["train"][1],
        eval_set=[(data["val"][0], data["val"][1])],
        verbose=True,
    )

    # Predict and evaluate accuracy
    preds = xgb_model_best.predict(data["test"][0])
    best_accuracy = accuracy_score(data["test"][1], preds)

    return study.best_params, best_accuracy
