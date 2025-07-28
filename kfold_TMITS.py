"""
Performs training of a T_MITS model (optionally with cross-validation).

Note: at the moment, we use stratified train/test spliting by default when n_folds==1
"""

import json
import os
import warnings
from copy import deepcopy
from os.path import splitext

import numpy as np
import torch
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR, LRScheduler
from tqdm import tqdm

from config import TrainingConfig, EvalConfig
from dataset_classes.dataset_base import read_data
from dataset_classes.dataset_regression import (
    get_ds_from_key,
    get_ds_shorthand,
    padded_collate_fn,
)
from models.loss import QuantileLoss
from models.tmits import T_MITS
from utility_functions.utils import init_logger
from eval_kfold_TMITS import main as eval

# This config is used if this script gets ran directly
cfg = TrainingConfig(
    exp_name="29",
    data_path="generated/29_culled_168max.csv",
    dataset_key="generated/mimic_BU_key.csv",
    tgt_var_label="Creatinine (serum)",
    masked_var_labels=["Creatinine (serum)"],
    back_interval=None,
    max_measures=2500,
    horizon=None,
    histories=None,
    n_var_embs=0,  # Overriden by the info in dataset_key
    n_folds=1,
    train_batch_size=10,
    test_batch_size=10,
    lr=6e-5,
    lr_threshold=1e-5,
    n_epochs=30,
    dropout=0.2,
    dim_embed=208,
    n_layers=2,
    n_heads=4,
    n_quantiles=1,
    sched_factor=0.5,
    sched_patience=3,
    sched_per_batch=False,
    verbose=True,
    resume=False,
    do_eval_after=True,
)


def prep_folders(exp_name: str, exp_dir: str = "results/") -> str:
    """Makes the directory `{exp_dir}{exp_name}/` if it does not exist and returns it"""
    saving_path = f"{exp_dir}{exp_name}/"
    os.makedirs(saving_path, exist_ok=True)
    return saving_path


def train_one_epoch(
    model: T_MITS,
    train_dl: DataLoader,
    optim: torch.optim.Optimizer,
    loss_fn,
    sched: LRScheduler | None = None,
    device: str = "cpu",
    verbose: bool = True,
) -> float:
    """Train `model` over all batches in `train_dl`.

    Parameters
    ----------
    model : T_MITS
        The model to train.
    train_dl : DataLoader
        The DataLoader to train with.
    optim : torch.optim.Optimizer
        The optimizer to update the model weights.
    sched : torch.optim.lr_scheduler.LRScheduler, optional
        If not None, a learning rate scheduler that should be called after every optimizer step. By default None.
    loss_fn
        A pytorch regression loss function
    device : str, optional
        A device on which to perform forward and backwards passes, by default "cpu".
    verbose : bool, optional
        Whether to show progress stats, by default True.

    Returns
    -------
    float
        The average training loss over all batches of `train_dl`.
    """
    avg_train_loss = 0.0

    # Training mode
    model.train()

    if verbose:
        # ncols=0 means that there is no progress bar, only the progress stats (percentage, time, etc)
        pbar = tqdm(train_dl, ncols=0)
        pbar.set_description("Training...")
    else:
        pbar = train_dl

    for demog, values, times, variables, tgt, _, masks in pbar:
        optim.zero_grad()

        demog = demog.to(device)
        values = values.to(device)
        times = times.to(device)
        variables = variables.to(device)
        tgt = tgt.to(device)
        masks = masks.to(device)
        pred = model(demog, values, times, variables, masks)

        loss = loss_fn(pred, tgt)
        avg_train_loss += loss.item()

        loss.backward()
        optim.step()
        if sched is not None:
            sched.step()

    return avg_train_loss / len(train_dl)


# Testing step
def test_model(
    model: T_MITS,
    test_dl: DataLoader,
    loss_fn,
    device: str = "cpu",
    verbose: bool = True,
) -> float:
    """Test `model` over all batches in `test_dl`.

    Parameters
    ----------
    model : T_MITS
        The model to test.
    test_dl : DataLoader
        The DataLoader to test with.
    loss_fn
        A pytorch regression loss function
    device : str, optional
        A device on which to perform forward and backwards passes, by default "cpu"
    verbose : bool, optional
        Whether to show the progress stats, by default True

    Returns
    -------
    float
        The average testing loss over all batches of `test_dl`
    """
    avg_test_loss = 0.0

    # Testing mode
    model.eval()

    if verbose:
        pbar = tqdm(test_dl, ncols=0)
        pbar.set_description("Testing...")
    else:
        pbar = test_dl

    for demog, values, times, variables, tgt, _, masks in pbar:
        demog = demog.to(device)
        values = values.to(device)
        times = times.to(device)
        variables = variables.to(device)
        tgt = tgt.to(device)
        masks = masks.to(device)

        pred = model(demog, values, times, variables, masks)

        loss = loss_fn(pred, tgt)
        avg_test_loss += loss.item()
    return avg_test_loss / len(test_dl)


def main(cfg: TrainingConfig) -> None:
    """
    Perform the whole cross-validation routine using parameters in `cfg`.

    This will
        - log progress in the specified folder in a .log file.
        - Save the config to a .json file.
        - Save training and testing indexes for inference for each fold.
        - Save best model checkpoints for each fold.

    Note: can be cleanly interrupted with a keyboard interrupt.
    """

    saving_path = prep_folders(exp_name=cfg.exp_name)

    logger = init_logger(cfg.exp_name, f"{saving_path}T_MITS.log")
    if cfg.resume:
        logger.info(
            f"Restarting experiment {cfg.exp_name} from fold {cfg.n_completed_folds}, epoch {cfg.n_completed_epochs}"
        )
    else:
        logger.info(f"***Starting experiment {cfg.exp_name}***")

    # Save the config early to minimize risk of an early failure leading to no config file being saved.
    # As a side note, this might mean that n_var_embs is saved with the wrong value,
    # but it should be overriden at dataset instantiation anyway
    cfg.save_to_json(f"{saving_path}{cfg.config_file_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ## Datasets
    # Pre-load dataframe to only have to read it once (expensive operation)
    df = read_data(cfg.data_path)

    if cfg.dataset_preset is not None:
        # Get Datasets using the shorthand presets
        train_ds = get_ds_shorthand(
            loaded_df=df, preset=cfg.dataset_preset, key_path=cfg.dataset_key
        )
        test_ds = deepcopy(train_ds)
        warnings.warn(
            message="Using a dataset shorthand preset means other dataset attributes (max_measures, files_dir...) are not set automatically. Set them explicitly after instantiation if you need to change them from their defaults."
        )
    else:
        # Get Datasets using info contained in the key
        try:
            train_ds = get_ds_from_key(
                loaded_df=df,
                key_path=cfg.dataset_key,
                tgt_var_label=cfg.tgt_var_label,
                masked_var_labels=cfg.masked_var_labels,
                files_dir=cfg.files_dir,
                back_interval=cfg.back_interval,
                horizon=cfg.horizon,
                histories=cfg.histories,
                max_measures=cfg.max_measures,
            )
            test_ds = deepcopy(train_ds)  # we restrict the indexes later
        except Exception as e:
            logger.error(f"Error loading dataset from key file: {e}")
            raise RuntimeError(
                f"{e}\n\nNote that instantiating from `dataset_key` needs a valid `tgt_var_label` (case sensitive)."
            )

    # Determine the number of variables that will be embedded from the id of the time token,
    # which has been set as max(variable_ids) + 1 by get_ds_from_key
    # Also convert to native python int
    cfg.n_var_embs = int(train_ds.time_token_var)

    logger.info("Initialized datasets")

    if cfg.n_folds >= 2:
        # Init kfold indexes generator
        # Note : we need to set the random state in case of resuming
        # to have the same folds as before interruption
        kf = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.random_state)
        folds_iterator = kf.split(train_ds.indexes)
    else:
        # One fold == No cross-validation
        # This way of creating the folds_iterator is just to have the same
        # type as kf.split so the rest of the code can function as normal when iterating
        # (but we will only go through the cross-val loop once)
        try:
            with open(f"{splitext(cfg.data_path)[0]}_labels.json") as f:
                labels_dict = json.load(f)
            folds_iterator = (
                train_test_split(
                    train_ds.indexes,
                    test_size=0.2,
                    shuffle=True,
                    random_state=cfg.random_state,
                    stratify=[
                        labels_dict[str(stay_ind)] for stay_ind in train_ds.indexes
                    ],
                )
                for _ in range(1)
            )
        except FileNotFoundError:
            # If we can't find the labels (they are normally generated at culling time)
            logger.warning(
                "Could not find labels.json file to stratify indexes splitting! Falling back to regular splitting"
            )
            folds_iterator = (
                train_test_split(
                    train_ds.indexes,
                    test_size=0.2,
                    shuffle=True,
                    random_state=cfg.random_state,
                )
                for _ in range(1)
            )

    try:
        for fold_i, (train_idx, test_idx) in enumerate(folds_iterator):
            if cfg.resume and fold_i < cfg.n_completed_folds:
                # if we're resuming and this fold has already been completed, skip to next fold
                logger.info(f"Skipped fold {fold_i} as {cfg.n_completed_folds=}")
                continue

            logger.info(f"Starting fold {fold_i}/{cfg.n_folds - 1}")

            # Save indexes for inference (e.g. evaluation scripts) later
            np.save(f"{saving_path}T_MITS_train_idx_{fold_i}.npy", train_idx)
            np.save(f"{saving_path}T_MITS_test_idx_{fold_i}.npy", test_idx)

            # Set the indexes for each dataset
            train_ds.restrict_to_indexes(train_idx)
            test_ds.restrict_to_indexes(test_idx)

            # Normalize variables and times
            # Note : we need to do this after restricting indexes to avoid info leakage
            train_ds.normalize()
            test_ds.normalize()

            # preload all stays in memory
            # (so we don't have to do the filtering operations (horizon, n_measures...) every loop)
            train_ds.load_all_stays(verbose=cfg.verbose)
            test_ds.load_all_stays(verbose=cfg.verbose)

            train_dl = DataLoader(
                train_ds,
                batch_size=cfg.train_batch_size,
                shuffle=True,
                collate_fn=padded_collate_fn,
                drop_last=True,
            )
            test_dl = DataLoader(
                test_ds,
                batch_size=cfg.test_batch_size,
                shuffle=False,
                collate_fn=padded_collate_fn,
                drop_last=True,
            )

            model = T_MITS(
                n_var_embs=cfg.n_var_embs,
                dim_embed=cfg.dim_embed,
                n_layers=cfg.n_layers,
                n_heads=cfg.n_heads,
                dropout=cfg.dropout,
                activation="gelu",
                n_quantiles=cfg.n_quantiles,  # note: changing this might break other scripts for now
            )

            if (
                cfg.resume
                and fold_i == cfg.n_completed_folds
                and cfg.n_completed_epochs > 0
            ):
                # If this is the resumed fold, and we completed at least one epoch last time,
                # load saved model weights

                # Note : because we save the best model weights and not the last ones,
                # doing 15+10 epochs with resuming might lead to worse results than
                # doing 25 epochs in one sitting
                # For example if the best model in the first 15 epochs was at epoch 6
                # then with resuming the training will be equivalent to 6+10 epochs total

                model.load_state_dict(
                    torch.load(f"{saving_path}T_MITS_{fold_i}.pth", weights_only=True)
                )
                logger.info("Loaded model weights")

            model = model.to(device)

            if cfg.resume and fold_i == cfg.n_completed_folds:
                optim = torch.optim.Adam(model.parameters(), lr=cfg.resuming_lr)
            else:
                optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)

            if cfg.n_quantiles == 1:
                loss_fn = torch.nn.HuberLoss(delta=cfg.huber_delta)
            elif cfg.n_quantiles == 3:
                # Special case (using auto_quantiles here would lead to 0.25, 0.5 and 0.75 as the quantiles)
                loss_fn = QuantileLoss(manual_quantiles=[0.1, 0.5, 0.9])
            else:
                loss_fn = QuantileLoss(auto_quantiles=cfg.n_quantiles)

            # Reducing the learning rate when the loss stops decreasing
            if cfg.sched_per_batch:
                # Annealing the learning rate from max/25 to max in 30% of epochs, then back down to max/1e4 in the rest of the epochs
                sched = OneCycleLR(
                    optim, max_lr=1e-4, total_steps=cfg.n_epochs * len(train_dl)
                )
            else:
                sched = ReduceLROnPlateau(
                    optim, factor=cfg.sched_factor, patience=cfg.sched_patience
                )

            # Compute a starting test loss to have a reference point
            best_test_loss = test_model(
                model, test_dl, loss_fn, device, verbose=cfg.verbose
            )
            logger.info(f"Starting test loss : {best_test_loss:.6f}")

            # Note : even when resuming, we do not know the previous epoch of best loss
            # (but it will be in the logs)
            # TODO use the logs parser to know previous epoch of best loss
            epoch_best_loss = -1

            # Note : because there is a single parameter group, get_last_lr() gives a length 1 list
            if cfg.resume and fold_i == cfg.n_completed_folds:
                running_lr = [cfg.resuming_lr]
            else:
                running_lr = [cfg.lr]

            for epoch in range(cfg.n_epochs):
                if (
                    cfg.resume
                    and fold_i == cfg.n_completed_folds
                    and epoch < cfg.n_completed_epochs
                ):
                    # Skip this epoch if it was already done in this fold
                    logger.info(f"Skipped epoch {epoch} as {cfg.n_completed_epochs=}")
                    continue

                logger.info(f"Starting {epoch=}/{cfg.n_epochs - 1}")
                avg_train_loss = train_one_epoch(
                    model=model,
                    train_dl=train_dl,
                    optim=optim,
                    loss_fn=loss_fn,
                    sched=sched if cfg.sched_per_batch else None,
                    device=device,
                    verbose=cfg.verbose,
                )
                logger.info(f"Average training loss : {avg_train_loss:.6f}")

                avg_test_loss = test_model(
                    model, test_dl, loss_fn, device, verbose=cfg.verbose
                )
                logger.info(f"Average testing loss : {avg_test_loss:.6f}")

                if avg_test_loss < best_test_loss:
                    # If best test loss yet
                    best_test_loss = avg_test_loss
                    epoch_best_loss = epoch
                    torch.save(model.state_dict(), f"{saving_path}T_MITS_{fold_i}.pth")
                    logger.info(f"New best test loss @{epoch=}, saved weights")

                if not cfg.sched_per_batch:
                    sched.step(avg_test_loss)

                current_lr = sched.get_last_lr()
                if current_lr != running_lr:
                    logger.info(f"Updated lr from {running_lr[0]} to {current_lr[0]}")
                    running_lr = current_lr

                if (not cfg.sched_per_batch) and (running_lr[0] < cfg.lr_threshold):
                    logger.info(
                        f"Stopped training because lr got below {cfg.lr_threshold}"
                    )
                    break  # stop the training loop, go to next fold

            logger.info(
                f"Best test loss : {best_test_loss:.6f} at epoch {epoch_best_loss}"
            )

            # reset the datasets to before index restriction and normalization
            train_ds.reset_df(df)
            test_ds.reset_df(df)

        logger.info("***END***")

    except KeyboardInterrupt:
        # Exit cleanly, and log the progress to facilitate resuming
        logger.info(
            f"KeyboardInterrupt. Completed {fold_i} folds, completed {epoch} epochs in current fold"
        )

    if cfg.do_eval_after:
        eval(
            EvalConfig(
                exp_name=cfg.exp_name,
                do_train=False,
                save_confusion_matrix=True,
                save_true_pred=True,
                save_xlsx=True,
            )
        )


if __name__ == "__main__":
    main(cfg)
