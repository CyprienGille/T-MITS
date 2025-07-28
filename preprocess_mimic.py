"""
Preprocess MIMIC-IV data.

This script will :
    - Selectively load items (i.e. signals) of interest
    - Remove outliers
    - Add a 'time since admission' column
    - Remove short and very long stays
    - Add demographic (age, sex) information
    - Write the .csv to disk

Parameters to control the output of this script are at the start of it.
"""

# Imports and definitions
import os

import numpy as np
import polars as pl

from utility_functions.preprocessing_utils import (
    add_index_from_col,
    get_item_from_full,
    get_item_lazy,
    remove_outliers,
    load_chartevents,
    bottom_up_items,
    top_down_items,
)

## Parameters
data_dir = "../mimic-iv-2.2/"
output_dir = "generated/"
# False means bottom-up (hand-picked 29 variables)
# True means top-down (frequency-based 206 variables)
top_down = True

# Whether to load all of chartevents.csv in memory at once (Faster, needs enough RAM)
# If False, loads it by streaming only the needed lines for each item (Can be much slower, needs less RAM)
load_once = True

# Minimum duration of a stay, in hours
min_hours = 0

# Filtering stays by patient origin
# origins_to_keep = ["EMERGENCY ROOM", "TRANSFER FROM HOSPITAL"]

if top_down:
    output_csv_name = "206.csv"
    output_key_name = "mimic_TD_key.csv"
    items_to_include = top_down_items
    min_measures, max_measures = 10, 30_000
else:
    # Default values for bottom up (you can change these)
    output_csv_name = "29.csv"
    output_key_name = "mimic_BU_key.csv"
    items_to_include = bottom_up_items
    # Min and max number of measures in a stay (both inclusive)
    min_measures, max_measures = 10, 5000

if __name__ == "__main__":
    all_dfs_list = []

    if load_once:
        print("Loading chartevents...")
        df_full = load_chartevents(data_dir)

        print("Filtering items and removing outliers...")
        for label, itemid in items_to_include.items():
            all_dfs_list.append(remove_outliers(get_item_from_full(df_full, itemid)))

    else:
        print("Loading chartevents selectively and removing outliers...")
        for label, itemid in items_to_include.items():
            all_dfs_list.append(remove_outliers(get_item_lazy(itemid)))

    print("Concatenating variables...")
    df_ev = pl.concat(all_dfs_list)

    # reindex the item ids starting from zero
    print("Re-indexing items...")
    df_ev, indexes_key = add_index_from_col(
        df_ev, col_name="itemid", index_name="itemid", return_key=True
    )

    # Save the key
    labels = []
    itemids = []
    reindexed_ids = []

    for label, itemid in items_to_include.items():
        # we use a for loop to make sure everything is aligned (since there is no order in dicts)
        labels.append(label)
        itemids.append(itemid)
        reindexed_ids.append(indexes_key[itemid])

    pl.DataFrame({"label": labels, "itemid": itemids, "key": reindexed_ids}).write_csv(
        output_dir + output_key_name
    )

    # Load the admissions dataframe, parse dates
    print("Loading admissions...")
    df_hadm = pl.read_csv(
        data_dir + "hosp/admissions.csv",
        columns=["admittime", "hadm_id", "deathtime", "admission_location"],
    ).with_columns(pl.col("admittime").str.strptime(pl.Datetime, format="%Y-%m-%d %T"))

    # Add the admission info to the event dataframe
    print("Joining events and admission data...")
    df_ev_hadm = df_ev.join(df_hadm, on="hadm_id")

    # Only keep the stays of patients that come from the specified origins
    # print("Removing stays by origin...")
    # df_ev_hadm = df_ev_hadm.filter(pl.col("admission_location").is_in(origins_to_keep))

    # Add a column with relative event time (since admission) in minutes
    print("Creating and sorting by 'Time since admission'...")
    df_ev_hadm = df_ev_hadm.with_columns(
        (pl.col("charttime") - pl.col("admittime")).dt.total_minutes().alias("time")
    )

    df_ev_hadm = df_ev_hadm.sort([pl.col("stay_id"), pl.col("time")])

    # Add column that indexes the stays starting from zero
    print("Indexing stays...")
    df_ev_hadm = add_index_from_col(df_ev_hadm, col_name="stay_id", index_name="ind")

    print("Removing short stays...")
    if min_hours > 0:
        # Remove stays shorter than min_hours
        # Note: this only works because the data has been sorted by time
        time_bounds = df_ev_hadm.group_by("ind").agg(
            [pl.col("time").last().alias("last"), pl.col("time").first().alias("first")]
        )
        long_stay_inds = time_bounds.select(
            pl.col("ind"), (pl.col("last") - pl.col("first")) > min_hours * 60
        )
        df_ev_hadm = df_ev_hadm.filter(
            pl.col("ind").is_in(long_stay_inds.select("ind").to_numpy().flatten())
        )
    # Keep only the stays with a valid amount of measures across all variables
    to_keep_inds = (
        df_ev_hadm.group_by(pl.col("ind"))
        .agg(pl.col("ind").len().alias("count"))
        .filter(pl.col("count").is_between(min_measures, max_measures, closed="both"))
    )

    df_ev_hadm = df_ev_hadm.filter(pl.col("ind").is_in(to_keep_inds.select("ind").to_numpy().flatten()))

    # Add demographic data
    print("Adding demographic data...")
    df_demog = pl.read_csv(
        data_dir + "hosp/patients.csv", columns=["subject_id", "gender", "anchor_age"]
    ).with_columns(pl.col("anchor_age").alias("age"))

    df_ev_hadm_demog = df_ev_hadm.join(df_demog, on="subject_id")

    print("Writing to disk...")
    os.makedirs(output_dir, exist_ok=True)

    # drop unused cols
    df_ev_hadm_demog.drop(
        [
            "charttime",
            "admission_location",
            "hadm_id",
            "stay_id",
            "subject_id",
            "admittime",
            "anchor_age",
        ]
    ).write_csv(output_dir + output_csv_name)
    print(f"Done. Wrote {df_ev_hadm_demog.select(pl.len()).item()} lines to csv.")
    print(
        f"Current database contains {len(np.unique(df_ev_hadm_demog.select('ind')))} stays."
    )
