import pandas as pd
from torch import Tensor, cat, float32, ones_like, tensor
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from dataset_classes.dataset_base import ICU
from utility_functions.utils import denorm, descale, get_id_from_label, norm, scale


class ICU_Reg(ICU):
    def __init__(
        self,
        data_path: str | None = None,
        loaded_df: pd.DataFrame | None = None,
        files_dir: str | None = None,
        var_id: int = 0,
        time_token_var: int = 30,
        crop_back_interval: int | None = None,
        max_measures: int | None = None,
        horizon: int | None = None,
        histories: list[int] | None = None,
        masked_features: list[int] | None = None,
    ):
        """Load ICU data into a Dataset object intended for regression.

        Note : time_token_var should be 1 more than the largest variable id in the data
        Note : large values/None for crop_back_interval can lead to long sequences and large complexity.
        Note : crop_back_interval is ignored if either max_measures or horizon are not None

        Parameters
        ----------
        data_path, loaded_df, files_dir
            See superclass docstring
        var_id : int, optional
            id of the target variable, by default 0
        time_token_var
            id of the variable dedicated to the time token, by default 30
        crop_back_interval : int, optional
            If not None, how many minutes before the target time to keep, by default None
        max_measures : int, optional
            How many points to leave in the input at most (if over limit, remove oldest measures), by default None
        horizon : int, optional
            If not None, how many hours should the last input measure be before the target observation, by default None
        histories : list[int], optional
            a list of history lengths (from start of seq) to try and keep (greedy), by default None
        masked_features : list[int], optional
            a list of feature ids to remove from inputs, by default None
        """
        super().__init__(data_path, loaded_df, files_dir)
        self.var_id = var_id
        self.time_token_var = time_token_var
        self.back = crop_back_interval
        self.max_measures = max_measures
        self.horizon = horizon
        if histories is not None:
            self.histories = sorted(histories, reverse=True)
        else:
            self.histories = histories
        self.masked_features = masked_features

        self.has_loaded_all = False

    def reset_df(self, df):
        super().reset_df(df)
        self.has_loaded_all = False

    def filter_by_back_time(self, values, times, variables, tgt_time):
        """Filter the provided tensors by removing observations older than a given lookback time.

        Note: the lookback is done from the target observation.

        Parameters
        ----------
        values : torch.Tensor
            All observed values
        times : torch.Tensor
            All observation times
        variables : torch.Tensor
            All observed variables
        tgt_time : torch.Tensor
            Time of the target observation

        Returns
        -------
        Same types as input
            Same as input (values, times, variables) but without the filtered measures
        """
        if not self.normed_times:
            cutoff_time = tgt_time.item() - self.back
        else:
            # Since self.back is in minutes, we need to denorm and norm
            # to get the normalized cutoff time out
            if self.mode == "standard":
                cutoff_time = norm(
                    denorm(tgt_time.item(), self.time_mean, self.time_std) - self.back,
                    self.time_mean,
                    self.time_std,
                )
            elif self.mode == "minmax":
                cutoff_time = scale(
                    descale(tgt_time.item(), self.time_min, self.time_max) - self.back,
                    self.time_min,
                    self.time_max,
                )

        # boolean mask
        to_keep = times >= cutoff_time

        additional_backs = 1
        while len(values[to_keep]) == 0:
            # Try to avoid empty sequences
            # While there are no kept datapoints, widen the window by [back] minutes
            if not self.normed_times:
                cutoff_time = tgt_time.item() - (additional_backs + 1) * self.back
            else:
                if self.mode == "standard":
                    cutoff_time = norm(
                        denorm(tgt_time.item(), self.time_mean, self.time_std)
                        - (additional_backs + 1) * self.back,
                        self.time_mean,
                        self.time_std,
                    )
                elif self.mode == "minmax":
                    cutoff_time = scale(
                        descale(tgt_time.item(), self.time_min, self.time_max)
                        - (additional_backs + 1) * self.back,
                        self.time_min,
                        self.time_max,
                    )
            to_keep = times >= cutoff_time

            additional_backs += 1

        return values[to_keep], times[to_keep], variables[to_keep]

    def filter_by_horizon(
        self, values, times, variables, tgt_time, horizon_minutes, needs_norm_check=True
    ):
        """Filter the provided tensors by removing observations that are too close (in time) to the target observation.

        For example, if the stay lasted 48 hours, the target happened at 47 hours, and the horizon is 12 hours,
        then all measures after the 35-hour mark will be dropped.

        Parameters
        ----------
        values : torch.Tensor
            All observed values
        times : torch.Tensor
            All observation times
        variables : torch.Tensor
            All observed variables
        tgt_time : torch.Tensor
            Time of the target observation
        horizon_minutes : int
            The length of time to leave between last input obs and tgt obs, in minutes
        needs_norm_check : bool, optional
            Whether the horizon_minutes arg needs to be normalized, by default True

        Returns
        -------
        Same types as input
            Same as input (values, times, variables) but without the filtered measures
        """
        if (not needs_norm_check) or (not self.normed_times):
            to_keep = times <= (tgt_time - horizon_minutes)
        else:
            if self.mode == "standard":
                norm_horiz = norm(horizon_minutes, self.time_mean, self.time_std)
            elif self.mode == "minmax":
                norm_horiz = scale(horizon_minutes, self.time_min, self.time_max)
            to_keep = times <= (tgt_time - norm_horiz)

        return values[to_keep], times[to_keep], variables[to_keep]

    def filter_by_n_measures(self, values, times, variables):
        """Filters the stay down to a given number of observations, dropping the oldest points first if needed"""
        current_len = len(values)
        if current_len > self.max_measures:
            return (
                values[current_len - self.max_measures :],
                times[current_len - self.max_measures :],
                variables[current_len - self.max_measures :],
            )
        return values, times, variables

    def filter_by_histories(self, values, times, variables, tgt_time):
        """Filter the provided tensors by keeping only a shorter input history at the start of the stay.

        Parameters
        ----------
        values : torch.Tensor
            All observed values
        times : torch.Tensor
            All observation times
        variables : torch.Tensor
            All observed variables
        tgt_time : torch.Tensor
            Time of the target observation

        Returns
        -------
        Same types as input
            Same as input (values, times, variables) but without the filtered measures
        """
        if self.normed_times:
            # histories (hours) -> histories (minutes + normalized)
            if self.mode == "standard":
                usable_hists = [
                    norm(h * 60, self.time_mean, self.time_std) for h in self.histories
                ]
            elif self.mode == "minmax":
                usable_hists = [
                    scale(h * 60, self.time_min, self.time_max) for h in self.histories
                ]
        else:
            usable_hists = [h * 60 for h in self.histories]

        for h in usable_hists:
            # See ValueError below
            # Assuming histories is sorted, descending
            if h < tgt_time:
                return self.filter_by_horizon(
                    values,
                    times,
                    variables,
                    tgt_time,
                    tgt_time - h,
                    needs_norm_check=False,  # already normed
                )
        raise ValueError(
            f"The least greedy history ({self.histories[-1]}h) is still too long for this stay and oversteps the target time."
        )

    def __getitem__(
        self, index: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Note: This may return an empty sequence, especially if masked_features is not None

        Parameters
        ----------
        index : int

        Returns
        -------
        Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
            demog, values, times, variables, target_value, target_time
        """

        if self.has_loaded_all:
            return self.loaded_dict[index]

        base_data = self.__get_stay_data__(index)

        try:
            tgt_line = base_data.loc[base_data["itemid"] == self.var_id].iloc[-1]
            tgt_val = tensor([tgt_line["value"]], dtype=float32)
            tgt_time = tensor([tgt_line["time"]], dtype=float32)
        except IndexError:
            raise IndexError(
                f"This stay ({self.indexes[index]}) might be empty or have no measure for the target variable and thus cannot be used for regression. Running culling_reg.py beforehand can avoid some of these cases."
            )

        # Keep only observations before the target observation
        data = base_data[base_data["time"] < tgt_time.item()]

        if self.masked_features is not None:
            # Remove masked variables observations
            data = data.loc[~data["itemid"].isin(self.masked_features)]

        values = tensor(data["value"].to_numpy(), dtype=float32)
        times = tensor(data["time"].to_numpy(), dtype=float32)
        variables = tensor(data["itemid"].to_numpy(), dtype=float32)

        if (
            self.max_measures is None
            and self.horizon is None
            and self.back is not None
            and len(values) != 0
        ):
            values, times, variables = self.filter_by_back_time(
                values, times, variables, tgt_time
            )

        if self.horizon is not None:
            values, times, variables = self.filter_by_horizon(
                values, times, variables, tgt_time, self.horizon * 60
            )

        if self.histories is not None:
            values, times, variables = self.filter_by_histories(
                values, times, variables, tgt_time
            )

        if self.max_measures is not None:
            values, times, variables = self.filter_by_n_measures(
                values, times, variables
            )

        # Add last token with target time information
        values = cat([values, tensor([0.0])])
        times = cat([times, tensor([tgt_time])])
        variables = cat([variables, tensor([float(self.time_token_var)])])

        gender_value = base_data["gender"].iloc[0]
        try:
            if gender_value[0] == "M":
                gender = tensor([-1.0])
            elif gender_value[0] == "F":
                gender = tensor([1.0])
            else:
                # value is not 'F' nor 'M' nor 'Female' nor 'Male'
                gender = tensor([0.0])
        except TypeError:
            # gender_value is NaN
            gender = tensor([0.0])
        except IndexError:
            # gender value is a scalar
            gender = tensor([0.0])

        age = tensor([base_data["age"].iloc[0]], dtype=float32)
        demog = cat([gender, age])

        return (
            demog,
            values,
            times,
            variables,
            tgt_val,
            tgt_time,
        )

    def load_all_stays(self, verbose: bool = False) -> None:
        """Prepares all items and loads them in memory for faster access later.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print a progress bar, by default False
        """
        self.has_loaded_all = False

        self.loaded_dict: dict[
            int, tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
        ] = {}

        n_indexes = len(self.indexes)
        if verbose:
            prog = tqdm(range(n_indexes), desc="Preparing all stays...")
        else:
            prog = range(n_indexes)

        for index in prog:
            self.loaded_dict[index] = self.__getitem__(index)

        self.has_loaded_all = True


def padded_collate_fn(
    batch: list[tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """collate_fn for sequences of irregular lengths, using padding

    Parameters
    ----------
    batch : list[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]]
        List of __getitem__ outputs (e.g. when fetched by a DataLoader)

    Returns
    -------
    Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
        demog, padded_values, padded_times, padded_variables, tgt_vals, tgt_times, masks
    """

    # Regroup demog, values, times... from each tuple
    grouped_inputs = list(zip(*batch))
    (demog, values, times, variables, tgt_vals, tgt_times) = grouped_inputs

    # Produce square tensors (shape [batch_size, longest_seq_len])
    # See pad_sequence docstring
    padded_values = pad_sequence(values, batch_first=True, padding_value=-1e4)
    padded_times = pad_sequence(times, batch_first=True, padding_value=-1e4)
    padded_variables = pad_sequence(variables, batch_first=True, padding_value=0)

    # unsqueeze to add the batch dimension
    # [batch_size, 2]
    demog = cat([t.unsqueeze(0) for t in demog])

    # No need to unsqueeze, [batch_size]
    tgt_vals = tensor(tgt_vals)
    tgt_times = tensor(tgt_times)

    # Mask for unobserved values (where we put the padding value)
    # shape [batch_size, longest_seq_len]
    masks = ones_like(padded_values)
    masks[padded_values == -1e4] = 0

    return (
        demog,
        padded_values,
        padded_times,
        padded_variables,
        tgt_vals,
        tgt_times,
        masks,
    )


def get_ds_shorthand(
    loaded_df: pd.DataFrame,
    preset: str,
    key_path: str = None,
) -> ICU_Reg:
    """Shorthand function to quickly instantiate an ICU_Reg Dataset
    with the experimental settings from the paper. Saves the user from having to type variable labels.

    Note that the dataset parameters (horizons, max_measures...) are left untouched by this helper.
    You can always set them later using the attributes directly:
    ```
    ds = get_ds_shorthand(df, key_file, '29-SCr')
    ds.max_measures = 4000
    ```

    Note: if key_file is None, the filename is assumed from the preset as follows:
        - '29' and '29-SCr' -> 'mimic_BU_key.csv' (BU=Bottom-Up)
        - '206' and '206-SCr' -> 'mimic_TD_key.csv' (TD=Top-Down)
        - 'eICU' and 'eICU-SCr' -> 'eICU_key.csv'

    Presets
    ----------
    - '29' -> Bottom-up MIMIC variable selection
    - '206' -> Top-down MIMIC variable selection
    - 'eICU' -> eICU variables
    - '29-SCr' -> Bottom-up MIMIC variable selection, no creatinine in input
    - '206-SCr' -> Top-down MIMIC variable selection, no creatinine in input
    - 'eICU-SCr' -> eICU variables, no creatinine in input

    Parameters
    ----------
    loaded_df : pd.DataFrame
        The already loaded preprocessed and culled csv.
    preset : str
        One of ['29', '29-SCr', '206', '206-SCr', 'eICU', 'eICU-SCr']. See above.
    key_path : str, optional
        The key file produced by the preprocessing code, by default None (see above).

    Returns
    -------
    ICU_Reg
        Properly initialized dataset object.
    """
    match preset:
        case "29":
            return get_ds_from_key(
                loaded_df,
                key_path,
                tgt_var_label="Creatinine (serum)",
                masked_var_labels=None,
            )
        case "29-SCr":
            return get_ds_from_key(
                loaded_df,
                key_path,
                tgt_var_label="Creatinine (serum)",
                masked_var_labels=["Creatinine (serum)"],
            )
        case "206":
            return get_ds_from_key(
                loaded_df,
                key_path,
                tgt_var_label="Creatinine (serum)",
                masked_var_labels=None,
            )
        case "206-SCr":
            return get_ds_from_key(
                loaded_df,
                key_path,
                tgt_var_label="Creatinine (serum)",
                masked_var_labels=["Creatinine (serum)"],
            )
        case "eICU":
            return get_ds_from_key(
                loaded_df,
                key_path,
                tgt_var_label="creatinine",
                masked_var_labels=None,
            )
        case "eICU-SCr":
            return get_ds_from_key(
                loaded_df,
                key_path,
                tgt_var_label="creatinine",
                masked_var_labels=["creatinine"],
            )
        case _:
            raise ValueError(
                f"Unknown preset. `preset` should be one of ['29', '29-SCr', '206', '206-SCr', 'eICU', 'eICU-SCr'], got '{preset}'."
            )


def get_ds_from_key(
    loaded_df: pd.DataFrame,
    key_path: str,
    tgt_var_label: str = "Creatinine (serum)",
    masked_var_labels: list[str] = None,
    files_dir: str = None,
    back_interval: int = None,
    horizon: int = None,
    histories: int = None,
    max_measures: int = None,
) -> ICU_Reg:
    """Instantiate an ICU_Reg dataset from a loaded preprocessed and culled csv and a key file.
    The key file provides the link between natural language labels and actual itemids in the data.

    Note: All labels (for `tgt_var_label` and `masked_var_labels`) are case sensitive.

    Note: Be mindful that histories that are too long can overstep on the target time and thus raise an error.

    Parameters
    ----------
    loaded_df : DataFrame
        The already loaded preprocessed and culled csv.
    key_path : str
        The key file path produced by the preprocessing code.
    tgt_var_label : str, optional
        Full label of the regression target variable, by default "Creatinine (serum)"
    masked_var_labels : list[str], optional
        If not None, labels of variables that will be masked, by default None
    files_dir : str, optional
        Directory with individual stay files, by default None
    back_interval : int, optional
        Observation window width, in minutes before target observation, by default None (use the full stay)
    horizon : int, optional
        Hours to try and leave between the last input observation and the target observation, by default None
    histories : int, optional
        List of history lengths (from start of seq) to try and keep, by default None
    max_measures : int, optional
        Maximum length (number of observations) of input sequence, by default None

    Returns
    -------
    ICU_Reg
        Properly initialized dataset object.
    """

    masked_lab = masked_var_labels or []

    df_keys = pd.read_csv(key_path)
    tgt_var_id = get_id_from_label(df_keys, tgt_var_label)
    masked_ids = [get_id_from_label(df_keys, label) for label in masked_lab]

    time_token_var = df_keys["key"].max() + 1

    return ICU_Reg(
        data_path=None,
        loaded_df=loaded_df,
        files_dir=files_dir,
        crop_back_interval=back_interval,
        var_id=tgt_var_id,
        time_token_var=time_token_var,
        masked_features=masked_ids,
        horizon=horizon,
        max_measures=max_measures,
        histories=histories,
    )
