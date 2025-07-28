"""Utility functions used for evaluating models"""

import pandas as pd
from numpy import ndarray
from sklearn.metrics import (
    accuracy_score,
    d2_tweedie_score,
    f1_score,
    mean_absolute_error,
    median_absolute_error,
    r2_score,
)

from utility_functions.utils import reg_to_classif


def get_metrics(
    y_true: ndarray | list,
    y_pred: ndarray | list,
    do_classif: bool = True,
    return_classes: bool = False,
) -> tuple[float, float, float, float, float, float]:
    """Generate regression and reg-to-classif metrics

    Note : See scikit-learn docs for details on each metric.

    Note : the gamma deviance d2 score is not defined when `y_true` or `y_pred`
    contains a zero or a negative value. In that case, we set it to 0.0.

    - MAE = Mean Absolute Error
    - MedAE = Median Absolute Error
    - r2 = r squared score (coef of determination)
    - Gamma d2 = Fraction of gamma deviance explained

    TODO Add Quantile regression metrics

    Parameters
    ----------
    y_true : ndarray | list
        Ground truth regression targets
    y_pred : ndarray | list
        Predicted (by any model) regression values
    do_classif : bool, optional
        Whether to also return classification metrics, by default True
    return_classes : bool, optional
        Whether to also return the predicted and true classes, by default False

    Returns
    -------
    tuple[float, float, float, float]
        (MAE, MedAE, r2, Gamma d2) if not `do_classif`
    tuple[float, float, float, float, float, float]
        (MAE, MedAE, r2, Gamma d2, classif accuracy, classif f1 score) if `do_classif`
    tuple[tuple[float, float, float, float, float, float], list[int], list[int]]
        ((MAE, MedAE, r2, Gamma d2, classif accuracy, classif f1 score), y_true_classif, y_pred_classif)
        if `do_classif` and `return_classes`
    """
    if do_classif:
        y_true_classif = reg_to_classif(y_true)
        y_pred_classif = reg_to_classif(y_pred)

    try:
        d2 = d2_tweedie_score(y_true, y_pred, power=2.0)
    except ValueError:
        # if y_true or y_pred is not strictly positive
        d2 = 0.0

    if do_classif:
        if return_classes:
            return (
                (
                    mean_absolute_error(y_true, y_pred),
                    median_absolute_error(y_true, y_pred),
                    r2_score(y_true, y_pred),
                    d2,
                    accuracy_score(y_true_classif, y_pred_classif),
                    f1_score(y_true_classif, y_pred_classif, average="weighted"),
                ),
                y_true_classif,
                y_pred_classif,
            )
        return (
            mean_absolute_error(y_true, y_pred),
            median_absolute_error(y_true, y_pred),
            r2_score(y_true, y_pred),
            d2,
            accuracy_score(y_true_classif, y_pred_classif),
            f1_score(y_true_classif, y_pred_classif, average="weighted"),
        )
    else:
        return (
            mean_absolute_error(y_true, y_pred),
            median_absolute_error(y_true, y_pred),
            r2_score(y_true, y_pred),
            d2,
        )


def add_mean_std(df: pd.DataFrame) -> None:
    """
    Add column-wise means and stds to the given pandas DataFrame
    """
    df["Mean"] = df.mean(axis=1)
    df["Std"] = df.std(axis=1, ddof=0)
