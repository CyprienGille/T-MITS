"""Evaluates a trained T-MITS model."""

import json
from os import makedirs
from os.path import basename, splitext

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import EvalConfig, TrainingConfig
from dataset_classes.dataset_base import read_data
from dataset_classes.dataset_regression import (
    ICU_Reg,
    get_ds_from_key,
    get_ds_shorthand,
    padded_collate_fn,
)
from models.tmits import T_MITS
from utility_functions.eval_utils import add_mean_std, get_metrics
from utility_functions.utils import denorm_list, descale_list

# This configuration is used if this script is ran directly
cfg = EvalConfig(
    exp_name="29",
    do_train=False,
    save_confusion_matrix=True,
    save_true_pred=True,
    save_xlsx=True,
)


def get_y_true_y_pred(
    model: T_MITS, dl: DataLoader, device="cpu", verbose=True, quantiles: bool = True
) -> tuple[list[float], list[float]]:
    """
    Generate arrays of ground truth values and predicted values
    for all elems of the dl DataLoader.

    If verbose, display progression stats.

    If quantiles, assume that the second quantile is the regression prediction.
    """
    model.eval()

    y_true = []
    y_pred = []

    if verbose:
        # ncols=0 -> no prog bar, only stats
        pbar = tqdm(dl, ncols=0)
        pbar.set_description("Getting predictions...")
    else:
        pbar = dl

    for demog, values, times, variables, tgt, _, mask in pbar:
        try:
            y_true.extend(tgt.flatten().tolist())

            demog = demog.to(device)
            values = values.to(device)
            times = times.to(device)
            variables = variables.to(device)
            mask = mask.to(device)
            out: torch.Tensor = model(
                demog=demog,
                values=values,
                times=times,
                variables=variables,
                mask=mask,
            )
            if quantiles:
                # Note: this assumes the 2nd quantile is the regression one (0.5)
                y_pred.extend(out[:, 1].flatten().tolist())
            else:
                y_pred.extend(out.tolist())
        except IndexError as e:
            raise IndexError(
                f"{e}\n\nNote: this error may happen if this stay is empty."
            )

    return y_true, y_pred


def save_conf_matrix(y_true, y_pred, title: str, saving_path: str):
    plt.figure()
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, normalize="all")
    plt.title(title)
    plt.savefig(saving_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def perform_evaluation(
    cfg: EvalConfig,
    exp_cfg: TrainingConfig,
    ds: ICU_Reg,
    df: pd.DataFrame,
    model: T_MITS,
    data_suffix: str,
    device: str = "cpu",
    train: bool = False,
    weights_path: str | None = None,
):
    """TODO: docstring"""
    set_name = "train" if train else "test"
    saving_dir_prefix = f"results/{cfg.exp_name}/"

    # Note : this is manually set so one can change the denomination of a given metric,
    # but it not linked in any way to the metrics generation.
    # If you modify get_metrics to return a new metric/return less metrics, then
    # There might be misalignment in the final dataframe.
    metric_names = ["mae", "medae", "r2", "d2", "acc", "f1"]
    res_columns = [f"Fold_{i}" for i in range(exp_cfg.n_folds)] + ["Mean", "Std"]

    # Init results
    results = pd.DataFrame(index=metric_names, columns=res_columns)

    for fold_i in range(exp_cfg.n_folds):
        # get the indexes of each set to reproduce the splitting and avoid leakage
        idx = np.load(f"{saving_dir_prefix}T_MITS_{set_name}_idx_{fold_i}.npy")

        ds.restrict_to_indexes(idx)

        ds.normalize()

        if ds.normed_vars:
            # Get the target mean and std (or min and max) for denormalization later
            if ds.mode == "standard":
                tgt_mean = ds.means[ds.var_id]
                tgt_std = ds.stds[ds.var_id]
            elif ds.mode == "minmax":
                tgt_min = ds.mins[ds.var_id]
                tgt_max = ds.maxs[ds.var_id]

        dl = DataLoader(
            ds,
            batch_size=10,
            shuffle=False,
            collate_fn=padded_collate_fn,
            drop_last=False,
        )

        if weights_path is None:
            model.load_state_dict(
                torch.load(f"{saving_dir_prefix}T_MITS_{fold_i}.pth", weights_only=True)
            )
        else:
            model.load_state_dict(
                torch.load(f"{weights_path}T_MITS_{fold_i}.pth", weights_only=True)
            )

        model = model.to(device)

        y_true, y_pred = get_y_true_y_pred(
            model, dl, device, exp_cfg.verbose, quantiles=(exp_cfg.n_quantiles > 1)
        )

        # denormalize before computing metrics
        if ds.normed_vars:
            if ds.mode == "standard":
                y_true = denorm_list(y_true, tgt_mean, tgt_std)
                y_pred = denorm_list(y_pred, tgt_mean, tgt_std)
            elif ds.mode == "minmax":
                y_true = descale_list(y_true, tgt_min, tgt_max)
                y_pred = descale_list(y_pred, tgt_min, tgt_max)

        if cfg.save_true_pred:
            # Save regression arrays
            makedirs(f"{saving_dir_prefix}arrays/", exist_ok=True)
            np.save(
                f"{saving_dir_prefix}arrays/y_true_{set_name}{data_suffix}_fold{fold_i}.npy",
                y_true,
            )
            np.save(
                f"{saving_dir_prefix}arrays/y_pred_{set_name}{data_suffix}_fold{fold_i}.npy",
                y_pred,
            )

        results[f"Fold_{fold_i}"], y_true_classif, y_pred_classif = get_metrics(
            y_true, y_pred, return_classes=True
        )

        if cfg.save_true_pred:
            # Save classification arrays
            makedirs(f"{saving_dir_prefix}arrays_classif/", exist_ok=True)
            np.save(
                f"{saving_dir_prefix}arrays_classif/y_true_{set_name}{data_suffix}_fold{fold_i}.npy",
                y_true_classif,
            )
            np.save(
                f"{saving_dir_prefix}arrays_classif/y_pred_{set_name}{data_suffix}_fold{fold_i}.npy",
                y_pred_classif,
            )

        if cfg.save_xlsx:
            df_xlsx = pd.DataFrame(
                data={
                    "stay_id": idx,
                    "target": y_true,
                    "prediction": y_pred,
                    "target_class": y_true_classif,
                    "prediction_class": y_pred_classif,
                }
            )

            df_xlsx.to_excel(
                f"{saving_dir_prefix}true_pred_values_{set_name}{data_suffix}_{fold_i}.xlsx",
                index=False,
            )

        if cfg.save_confusion_matrix and fold_i == 0:
            # In case of kkfold crossval, only save for first fold
            saving_path = f"{saving_dir_prefix}confusion_{set_name}{data_suffix}.png"

            # Remove underscore from override suffix with [1:]
            title_suffix = (
                f" evaluated on {data_suffix[1:]}" if data_suffix != "" else ""
            )

            save_conf_matrix(
                y_true_classif,
                y_pred_classif,
                title=f"{set_name.capitalize()} confusion matrix for {cfg.exp_name}{title_suffix}",
                saving_path=saving_path,
            )

        # reset the data held by the Dataset before going to next fold
        ds.reset_df(df)

    # Mean and std over all folds
    add_mean_std(results)

    results.to_csv(f"{saving_dir_prefix}metrics_{set_name}{data_suffix}.csv")


def main(cfg: EvalConfig, data_override: str | None = None):
    """
    Evaluate the models from each fold of the experiment pointed at by cfg.

    This will create two .csv in cfg.exp_dir, with the training and testing
    metrics respectively.

    To see which metrics will be computed, see `eval_utils.get_metrics`.

    Note: If you change the csv used for evaluation by using the `data_override` argument,
    make sure that the other csv is compatible (same variable names, for example) because every other parameter
    from the training config will be reused.

    Note 2: Using `data_override` will save the result `.csv` files with a suffix.

    TODO decide if data override should go in EvalConfig

    Optional Parameters
    -------------------
    data_override: str|None
        An path to a csv that was not the one used for training, to be used for evaluation. By default None (use original data)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load the training config
    with open(f"results/{cfg.exp_name}/{cfg.config_file_name}") as f:
        exp_cfg = TrainingConfig(**json.load(f))

    if data_override is not None and data_override != exp_cfg.data_path:
        # Read from a different path than the one used in training
        df = read_data(data_override)
        # Get the file name, no extension and no parent dirs, for later use in output file names
        data_suffix = "_" + splitext(basename(data_override))[0]
    else:
        df = read_data(exp_cfg.data_path)
        data_suffix = ""

    if exp_cfg.dataset_preset is not None:
        ds = get_ds_shorthand(
            loaded_df=df, preset=exp_cfg.dataset_preset, key_path=exp_cfg.dataset_key
        )
    else:
        ds = get_ds_from_key(
            loaded_df=df,
            key_path=exp_cfg.dataset_key,
            tgt_var_label=exp_cfg.tgt_var_label,
            masked_var_labels=exp_cfg.masked_var_labels,
            files_dir=exp_cfg.files_dir,
            back_interval=exp_cfg.back_interval,
            horizon=exp_cfg.horizon,
            histories=exp_cfg.histories,
            max_measures=exp_cfg.max_measures,
        )

    # Determine the number of variables that will be embedded from the id of the time token,
    # which has been set as max(variable_ids) + 1 by get_ds_from_key.
    exp_cfg.n_var_embs = ds.time_token_var

    # Instantiate the model, will load the weights later (per-fold)
    model = T_MITS(
        n_var_embs=exp_cfg.n_var_embs,
        dim_embed=exp_cfg.dim_embed,
        n_layers=exp_cfg.n_layers,
        n_heads=exp_cfg.n_heads,
        dropout=exp_cfg.dropout,
        n_quantiles=exp_cfg.n_quantiles,
        activation="gelu",
    )

    # Generate and save test metrics
    perform_evaluation(
        cfg=cfg,
        exp_cfg=exp_cfg,
        ds=ds,
        df=df,
        model=model,
        data_suffix=data_suffix,
        device=device,
    )

    if cfg.do_train:
        # Same but for training set
        ds.reset_df(df)  # Just in case, start fresh
        perform_evaluation(
            cfg=cfg,
            exp_cfg=exp_cfg,
            ds=ds,
            df=df,
            model=model,
            data_suffix=data_suffix,
            device=device,
            train=True,
        )


if __name__ == "__main__":
    main(cfg)
