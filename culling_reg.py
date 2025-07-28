"""
This script removes from a given preprocessed .csv :
    - Sections of each stay that are after the last observation of the target variable
    (since this last obs will be used as the regression target)
    - Stays with no measure for the target variable
    - Stays with only measures for skipped variables
    - (And more if some options are set)

This script can also split the culled DataFrame into individual stay files.
If there are a lot of lines, the I/O cost of loading the desired stay file can be lower
than having to filter through the whole loaded DataFrame.
"""

import json
import os
from os.path import splitext

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import CullingConfig
from utility_functions.utils import get_id_from_label, string_to_thresholding

# This config will be saved (as json) along the culled csv for experiment tracking purposes
cfg = CullingConfig(
    data_path="generated/29.csv",
    key_path="generated/mimic_BU_key.csv",
    out_path="generated/29_culled.csv",
    tgt_var_label="Creatinine (serum)",
    skippable_vars_labels=["Creatinine (serum)"],
    must_include_labels=None,
    must_exclude_labels=None,
    kept_entry_stages=None,
    kept_exit_stages=None,
    entry_above_exit=None,
    val_to_stage="KDIGO",
    pred_horizon=0,
    pred_history=0,
    max_hours=24 * 7,
    attr_in_paths=True,
    split_into_files=False,
    save_classif_labels=True,
)


class Tracker:
    """A class meant to help track various properties of a stay in a preprocessed DataFrame,
    specifically in the context of deciding if (part of) that stay should be droppped or not.


    See the CullingConfig docs for more details on the arguments."""

    def __init__(
        self,
        tgt_id: int,
        skippable_ids: list[int],
        cfg: CullingConfig,
        include_ids: list[int] | None = None,
        exclude_ids: list[int] | None = None,
    ):
        self.tgt_id = tgt_id
        self.skippable_ids = skippable_ids
        self.include_ids = include_ids or []
        self.exclude_ids = exclude_ids or []

        if (
            (cfg.kept_entry_stages is not None)
            or (cfg.kept_exit_stages is not None)
            or (cfg.entry_above_exit is not None)
            or (cfg.save_classif_labels)
        ):
            assert cfg.val_to_stage is not None, (
                "If aany of the stage-related args are set, you need to provide a val_to_stage function name."
            )
            self.stage_function = string_to_thresholding(cfg.val_to_stage)
        else:
            # This will be used to check if we care about tgt var thresholds
            self.stage_function = None

        self.cfg = cfg

    def ready(self, upcoming_stay_id: int = None):
        """Set all tracking variables to their initial states (before any line is processed)

        ``stay_id` only needs to be provided to identify the upcoming stay if `self.cfg.save_classif_labels`.
        """

        if self.cfg.save_classif_labels and (upcoming_stay_id is None):
            raise ValueError(
                "This tracker needs a stay_id to save classification labels later."
            )
        self.stay_id = upcoming_stay_id
        self.entry_stage = None
        self.exit_stage = None
        self.time_last_tgt_val = None
        self.n_tgt = 0
        self.n_non_skipped = 0
        # if include_ids is empty, ok_included is always True
        # Note: in python, empty lists are False
        self.ok_included = True if not self.include_ids else False
        self.ok_excluded = True
        self.passed_times = []
        self.passed_idx = []

    def update_from_row(self, row):
        """Update all internal variables of the Tracker with info from a new row

        Note: the row is expected to be a namedtuple with at least the following fields:
        Index, itemid, time, value
        """
        if row.itemid == self.tgt_id:
            # This row is an observation for the target variable
            self.time_last_tgt_val = row.time
            self.n_tgt += 1

            if self.stage_function is not None:
                # If stage_function is None we are not tracking entry and exit stages
                if self.n_tgt == 1:
                    # First tgt var obs
                    self.entry_stage = self.stage_function(row.value)

                # update at each tgt obs
                self.exit_stage = self.stage_function(row.value)

        if row.itemid not in self.skippable_ids:
            self.n_non_skipped += 1

        if row.itemid in self.include_ids:
            self.ok_included = True

        if row.itemid in self.exclude_ids:
            self.ok_excluded = False

        self.passed_times.append(row.time)
        self.passed_idx.append(row.Index)

    def exists_measure_without_horizon(self) -> bool:
        """
        Returns
        -------
        bool
            True if there exists input observations at least [pred_horizon] hours away from the target obs
        """
        if self.n_tgt == 0:
            return False

        # Since passed times is sorted, the first time is the smallest one
        return (
            self.passed_times[0] < self.time_last_tgt_val - self.cfg.pred_horizon * 60
        )

    def exists_measure_outside_history(self) -> bool:
        """

        Returns
        -------
        bool
            True if there exists a target observation outside (ie after the end of) the kept part of the input stay
        """
        if self.n_tgt == 0:
            return False
        return self.time_last_tgt_val > self.cfg.pred_history * 60

    def should_drop(self) -> bool:
        """
        if we only saw values from variables that will/can be skipped during training,
        or if that stay had no measure for the target variable,
        or if the available history before last tgt obs isn't long enough,
        or if the last tgt obs is within the history,
        or if the entry/exit stage(s) is/are not the one(s) we want,
        or if the trajectory of the tgt variable is not the one we want,
        or if the stay does not include a required variable;
        or if the stay includes an excluded variable;
        that stay is unusable/unwanted, we should drop all of its idx

        """
        # TODO Structure this docstring

        correct_entry = True
        correct_exit = True
        correct_traj = True

        # With no target value there would be no entry or exit stage
        if self.n_tgt != 0:
            # Logic: if we care about [entry/exit/trajectory],
            # Check and adjust the corresponding boolean
            if self.cfg.kept_entry_stages is not None:
                correct_entry = self.entry_stage in self.cfg.kept_entry_stages

            if self.cfg.kept_exit_stages is not None:
                correct_exit = self.exit_stage in self.cfg.kept_exit_stages

            if self.cfg.entry_above_exit is not None:
                went_up = self.entry_stage < self.exit_stage
                went_down = self.entry_stage > self.exit_stage
                correct_traj = went_down if self.cfg.entry_above_exit else went_up

        return (
            (self.n_non_skipped == 0)
            or (self.n_tgt == 0)
            or (not self.exists_measure_without_horizon())
            or (not self.exists_measure_outside_history())
            or (not correct_entry)
            or (not correct_exit)
            or (not correct_traj)
            or (not self.ok_included)
            or (not self.ok_excluded)
        )


def resolve_stay_info(
    tracker: Tracker, to_drop_indexes: list[int]
) -> tuple[int, int] | None:
    """Updates `to_drop_indexes` in-place with the indexes seen by a Tracker.
    If the stay should be dropped, all passed indexes are added to `to_drop_indexes`.
    If the stay should be used, only the indexes after the last target obs are added to `to_drop_indexes`.

    Note: If returned, the stay_label is equivalent to the exit stage.

    Returns
    ---------
    tuple[int, int] | None
        If the stay is kept, (stay_id, stay_label)

    """
    if tracker.should_drop():
        # drop the whole stay
        to_drop_indexes.extend(tracker.passed_idx)
    else:
        # keep the part of the stay before (inclusive) the last tgt obs
        idx = np.array(tracker.passed_idx)
        times = np.array(tracker.passed_times)
        to_drop_indexes.extend(idx[times > tracker.time_last_tgt_val])

        # Note: exit_stage is always defined, even if we're not tracking stages (in which case it will be None)
        return (tracker.stay_id, tracker.exit_stage)


def main(cfg: CullingConfig, prog_bar: bool = False):
    print("Reading data...")
    df_key = pd.read_csv(cfg.key_path)

    # Turn labels into ids
    tgt_id = get_id_from_label(df_key, cfg.tgt_var_label)
    if cfg.skippable_vars_labels is not None:
        skippable_ids = [
            get_id_from_label(df_key, label) for label in cfg.skippable_vars_labels
        ]
    else:
        skippable_ids = []

    if cfg.must_include_labels is not None:
        include_ids = [
            get_id_from_label(df_key, label) for label in cfg.must_include_labels
        ]
    else:
        include_ids = None

    if cfg.must_exclude_labels is not None:
        exclude_ids = [
            get_id_from_label(df_key, label) for label in cfg.must_exclude_labels
        ]
    else:
        exclude_ids = None

    df = pd.read_csv(cfg.data_path)

    to_drop_indexes = []
    if cfg.save_classif_labels:
        # directory that will map a stay index to its classification label
        stay_to_label = {}

    rows_as_tuples = df.itertuples()
    current_ind = -1
    tracker = Tracker(tgt_id, skippable_ids, cfg, include_ids, exclude_ids)

    print("Culling stays...")
    if prog_bar:
        pbar = tqdm(rows_as_tuples)
    else:
        pbar = rows_as_tuples

    for row in pbar:
        if current_ind != row.ind:
            # We just started a new stay (we are assuming that stays are contiguous in the csv)

            # If we're not saving labels or we haven't started yet:
            if current_ind == -1:
                # Haven't started
                # Set relevant variables
                tracker.ready(
                    upcoming_stay_id=row.ind if cfg.save_classif_labels else None
                )
            elif not cfg.save_classif_labels:
                # Not saving labels
                # First, resolve previous stay info
                resolve_stay_info(tracker, to_drop_indexes)
                # Second, reset relevant variables
                tracker.ready()
            else:
                resolve_out = resolve_stay_info(tracker, to_drop_indexes)
                try:
                    stay, label = resolve_out
                    stay_to_label[stay] = label
                except TypeError:
                    # Cannot unpack the None returned by resolve_stay_info when the stay should be dropped
                    pass

                tracker.ready(upcoming_stay_id=row.ind)

            # Then start dealing with new stay info
            current_ind = row.ind

        if tracker.cfg.max_hours is None:
            tracker.update_from_row(row)
        else:
            if row.time > tracker.cfg.max_hours * 60:
                # if we're beyond the max_hours, drop this row and don't update the tracker anymore
                to_drop_indexes.append(row.Index)
            else:
                tracker.update_from_row(row)

    # This is just to resolve the last stay
    resolve_out = resolve_stay_info(tracker, to_drop_indexes)
    if cfg.save_classif_labels:
        try:
            stay, label = resolve_out
            stay_to_label[stay] = label
        except TypeError:
            # Cannot unpack the None returned by resolve_stay_info when the stay should be dropped
            pass

    df.drop(pd.Index(to_drop_indexes), inplace=True)

    full_path, extension = splitext(cfg.out_path)

    print(f"Writing data to {full_path}...")
    cfg.save_to_json()

    if cfg.save_classif_labels:
        with open(full_path + "_labels.json", "w") as f:
            json.dump(stay_to_label, f, indent=4)

    df.to_csv(cfg.out_path)
    if cfg.split_into_files:
        # write one file for each stay
        out_dir = full_path + "/"
        os.makedirs(out_dir, exist_ok=True)

        all_inds = df["ind"].unique()

        for ind in tqdm(all_inds):
            df[df["ind"] == ind].to_csv(out_dir + f"ind_{ind}.csv")

    print(
        f"Done. Current database contains {df['ind'].nunique()} stays for {len(df.index)} lines."
    )


if __name__ == "__main__":
    main(cfg, prog_bar=True)
