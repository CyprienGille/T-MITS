"""
Preprocess eICU data.

Parameters to control the output of this script are at the start of it.
"""

import os

import numpy as np
import polars as pl

from utility_functions.preprocessing_utils import (
    add_index_from_col,
    remove_outliers,
    eicu_included_labs,
    eicu_included_periodic,
)

data_dir = "../eicu-crd-2.0/"
output_dir = "generated/"
output_csv_name = "eicu.csv"
output_key_name = "eicu_key.csv"


def get_item_from_lab(df: pl.DataFrame, variable: str):
    """Simply filters the 'variable' column of `df` to look for `variable`."""
    return df.filter(pl.col("variable").is_in([variable]))


if __name__ == "__main__":
    print("Loading and filtering labs data, removing outliers...")
    df_lab_orig = pl.read_csv(
        data_dir + "lab.csv",
        columns=["patientunitstayid", "labresultoffset", "labname", "labresult"],
    ).rename({"labresultoffset": "time", "labname": "variable", "labresult": "value"})

    all_labs = []

    for variable in eicu_included_labs:
        all_labs.append(remove_outliers(get_item_from_lab(df_lab_orig, variable)))

    print("Concatenating labs variables...")
    df_lab = pl.concat(all_labs)

    print("Loading periodic vitals...")
    df_periodic = (
        pl.scan_csv(data_dir + "vitalPeriodic.csv")
        .select(["patientunitstayid", "observationoffset"] + eicu_included_periodic)
        .rename({"observationoffset": "time"})
        .unpivot(index=["patientunitstayid", "time"])  # This is expensive
        .drop_nulls()
        # skip rows to decrease sampling frequency and reduce size of dataset
        .gather_every(3)
        .collect(engine="streaming")
    )

    print("Concatenating labs and periodic vitals...")
    df_measures = pl.concat([df_periodic, df_lab], how="vertical_relaxed")

    print("Re-indexing variables...")
    df_measures, indexes_key = add_index_from_col(
        df_measures, col_name="variable", index_name="itemid", return_key=True
    )

    # Save the key
    all_labels = eicu_included_labs + eicu_included_periodic
    reindexed_ids = [indexes_key[label] for label in all_labels]

    pl.DataFrame({"label": all_labels, "key": reindexed_ids}).write_csv(
        output_dir + output_key_name
    )

    print("Sorting stays...")
    df_measures = df_measures.sort([pl.col("patientunitstayid"), pl.col("time")])

    print("Indexing stays...")
    df_measures = add_index_from_col(
        df_measures, col_name="patientunitstayid", index_name="ind"
    )

    print("Removing short and long stays...")
    uniques, counts = np.unique(
        df_measures.select("ind").to_numpy().flatten(), return_counts=True
    )

    val_counts_dict = dict(zip(uniques, counts))

    df_measures = df_measures.with_columns(
        pl.col("ind").replace(val_counts_dict).alias("count")
    )
    df_measures = df_measures.filter((pl.col("count") > 9) & (pl.col("count") <= 20000))

    print("Adding demographic data...")
    df_demog = pl.read_csv(
        data_dir + "patient.csv", columns=["patientunitstayid", "gender", "age"]
    ).with_columns(pl.col("age").replace({"> 89": 95, "": 50}).cast(pl.Int32))

    df_measures_demog = df_measures.join(df_demog, on="patientunitstayid")

    print("Writing to disk...")
    os.makedirs(output_dir, exist_ok=True)

    df_measures.drop(["patientunitstayid", "count"])
    df_measures_demog.write_csv(output_dir + output_csv_name)
    print(f"Done. Wrote {df_measures_demog.select(pl.len()).item()} lines to csv.")
    print(
        f"Current database contains {len(np.unique(df_measures_demog.select('ind')))} stays."
    )

# Done. Wrote 275 604 817 lines to csv.
# Current database contains 197 042 stays.
