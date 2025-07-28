import warnings

import pandas as pd
from numpy import float64, int64
from torch.utils.data import Dataset
from tqdm import tqdm

from utility_functions.utils import norm, scale


class ICU(Dataset):
    def __init__(
        self,
        data_path: str | None = None,
        loaded_df: pd.DataFrame | None = None,
        files_dir: str | None = None,
    ):
        """Load ICU (MIMIC-IV or eICU) data into a torch Dataset object

        Note: one of data_path or loaded_df has to be provided.

        Note : if files_dir is not None, the df at data_path or in loaded_df will only
        be used to compute normalization stats and valid indexes.

        Note : This class assumes that the individual stay files (if files_dir is not None) are
        of the form 'ind_{ind}.csv'

        Parameters
        ----------
        data_path : str, optional
            Path to a .csv containing data (check preprocessing code to see what that data looks like), by default None
        loaded_df : DataFrame, optional
            if provided, supersedes the data found at data_path with an already loaded DataFrame, by default None
        files_dir : str, optional
            If not None, indicates the directory where the per-stay files corresponding to data_path can be found, by default None
        """
        super().__init__()

        assert not (data_path is None and loaded_df is None), (
            "One of data_path or loaded_df has to be provided"
        )

        if loaded_df is not None:
            self.df = loaded_df.copy()
        else:
            self.df = read_data(data_path)

        self.indexes = self.df["ind"].unique()

        self.split_into_files = files_dir is not None
        self.files_dir = files_dir

        self.normed_vars = False
        self.normed_times = False

    def reset_df(self, df: pd.DataFrame) -> None:
        """Resets in-place the internal dataframe with the provided one.
        Resets indexes and normalization flags.
        """
        self.df = df.copy()
        self.indexes = self.df["ind"].unique()
        self.normed_times = False
        self.normed_vars = False

    def restrict_to_indexes(self, indexes: list[int]) -> None:
        """Restricts in-place the internal dataframe to the provided indexes (using the `ind` column)"""
        current_set, new_set = set(self.indexes), set(indexes)
        remaining = current_set.intersection(new_set)

        if len(remaining) == 0:
            raise ValueError(
                "The new set of indexes is strictly absent from the internal dataframe and would thus restrict it down to zero stays."
            )

        self.indexes = list(remaining)
        self.df = self.df.loc[self.df["ind"].isin(self.indexes)]

    def normalize(
        self,
        normalize_vars=True,
        normalize_times=True,
        mode="standard",
        verbose=False,
    ) -> None:
        """
        Normalizes the values (per variable) and the times.

        If `verbose==True`, displays a progress bar for per-variable normalization.

        Note : This should be called *after* any index restriction to avoid information leakage

        Note: `mode` should be one of ['standard', 'minmax']

        Note : if data is split into files, normalization will happen at read time
        using the computed means and stds (or mins and maxs) from this method (so you should still call it once)
        """
        if mode == "minmax":
            warnings.warn(
                "Min-Max normalization is relatively new and thus may raise errors from other scripts in this codebase - use with caution."
            )
        if normalize_vars:
            if verbose:
                print("Normalizing variables...")
                itemids = tqdm(self.df["itemid"].unique())
            else:
                itemids = self.df["itemid"].unique()
            if mode == "standard":
                # per-variable standardization (0 mean, 1 std)
                self.means = {}
                self.stds = {}
                for unique_itemid in itemids:
                    data = self.df.loc[
                        self.df["itemid"] == unique_itemid, "value"
                    ].copy()

                    mean = data.mean()
                    std = data.std(ddof=0)

                    self.means[unique_itemid] = mean
                    self.stds[unique_itemid] = std
                    if not self.split_into_files:
                        self.df.loc[self.df["itemid"] == unique_itemid, "value"] = norm(
                            data, mean, std
                        )

                # age normalization
                data = self.df["age"].copy()
                self.age_mean = data.mean()
                self.age_std = data.std(ddof=0)
                if not self.split_into_files:
                    self.df["age"] = norm(data, self.age_mean, self.age_std)
            elif mode == "minmax":
                # per-variable min-max scaling
                self.mins = {}
                self.maxs = {}
                for unique_itemid in itemids:
                    data = self.df.loc[
                        self.df["itemid"] == unique_itemid, "value"
                    ].copy()

                    item_max = data.max()
                    item_min = data.min()

                    self.maxs[unique_itemid] = item_max
                    self.mins[unique_itemid] = item_min
                    if not self.split_into_files:
                        self.df.loc[self.df["itemid"] == unique_itemid, "value"] = (
                            scale(data, item_min, item_max)
                        )
                # age scaling
                data = self.df["age"].copy()
                self.age_min = data.min()
                self.age_max = data.max()
                if not self.split_into_files:
                    self.df["age"] = scale(data, self.age_min, self.age_max)

            else:
                raise ValueError(
                    f"mode must be one of ['standard', 'minmax'], got '{mode}'"
                )

        if normalize_times:
            if verbose:
                print("Normalizing times...")
            if mode == "standard":
                # Global time standardization
                data = self.df["time"].copy()
                self.time_mean = data.mean()
                self.time_std = data.std(ddof=0)
                if not self.split_into_files:
                    self.df["time"] = norm(data, self.time_mean, self.time_std)
            elif mode == "minmax":
                data = self.df["time"].copy()
                self.time_min = data.min()
                self.time_max = data.max()
                if not self.split_into_files:
                    self.df["time"] = scale(data, self.time_min, self.time_max)
            else:
                raise ValueError(
                    f"mode must be one of ['standard', 'minmax'], got '{mode}'"
                )

        self.normed_vars = normalize_vars
        self.normed_times = normalize_times
        self.mode = mode

    def __get_stay_data__(self, index: int) -> pd.DataFrame:
        """
        if self.split_into_files == True:
            Read stay data from a single file, and normalize it if needed
        else:
            Return stay data from the global DataFrame
        """
        if not self.split_into_files:
            # Just return the relevant section of the internal dataframe
            base_data = self.df.loc[self.df["ind"] == self.indexes[index]].copy()
        else:
            # If reading from a single file, we need to do normalization every time
            stay_path = self.files_dir + f"ind_{self.indexes[index]}.csv"

            base_data = read_data(stay_path)

            if self.normed_vars:
                for unique_itemid in base_data["itemid"].unique():
                    # Norm each variable
                    if self.mode == "standard":
                        base_data.loc[base_data["itemid"] == unique_itemid, "value"] = (
                            norm(
                                base_data.loc[
                                    base_data["itemid"] == unique_itemid, "value"
                                ],
                                self.means[unique_itemid],
                                self.stds[unique_itemid],
                            )
                        )
                    elif self.mode == "minmax":
                        base_data.loc[base_data["itemid"] == unique_itemid, "value"] = (
                            scale(
                                base_data.loc[
                                    base_data["itemid"] == unique_itemid, "value"
                                ],
                                self.mins[unique_itemid],
                                self.maxs[unique_itemid],
                            )
                        )
                # Age normalization
                if self.mode == "standard":
                    base_data["age"] = norm(
                        base_data["age"], self.age_mean, self.age_std
                    )
                elif self.mode == "minmax":
                    base_data["age"] = scale(
                        base_data["age"], self.age_min, self.age_max
                    )

            if self.normed_times:
                # Norm times
                if self.mode == "standard":
                    base_data["time"] = norm(
                        base_data["time"], self.time_mean, self.time_std
                    )
                elif self.mode == "minmax":
                    base_data["time"] = scale(
                        base_data["time"], self.time_min, self.time_max
                    )

        return base_data

    def __len__(self) -> int:
        """Number of stays in this dataset."""
        return len(self.indexes)


def read_data(data_path: str) -> pd.DataFrame:
    """Read data from `data_path` using only the relevant columns (itemid, value, time, ind, gender, age)
    and setting the correct dtypes for torch later"""
    return pd.read_csv(
        data_path,
        usecols=[
            "itemid",
            "value",
            "time",
            "ind",
            "gender",
            "age",
        ],
        dtype={
            "itemid": int64,
            "value": float64,
            "time": float64,
            "ind": int64,
            "age": float64,
        },
    )
