"""Various Config classes: dataclasses containing all the adjustable parameters for a given script.

All configs are serializable to JSON if they inherit from SavableConfig.
"""

import json
from dataclasses import asdict, dataclass
from os.path import splitext

from utility_functions.utils import choose_fallback, list_to_compact_str


class SavableConfig:
    """A generic Config dataclass that can be saved to a json format"""

    def save_to_json(self, path: str) -> None:
        """Save this instance as json

        Parameters
        ----------
        path : str
            The .json filepath, e.g. 'directory/config.json'
        """
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=4)


@dataclass
class CullingConfig(SavableConfig):
    """All parameters for culling a preprocessed csv.

    TODO Check consistency of using None vs empty lists

    Parameters
    ----------
    data_path: str
        Path to the preprocessed .csv (non-culled)
    key_path: str
        Path to the key file used to interpret variable labels
    out_path: str
        Path to save the culled .csv (the config will be saved here too as a .json)
    tgt_var_label: str
        Label of the variable that will be used as the regression target
    skippable_vars_labels: list[str], optional
        Labels of variables that should be skippable without resulting in empty stays, by default None (no skippable variables)
    must_include_labels: list[str], optional
        Labels of variables that must be present at least once (any of them) in a stay for it to be kept, by default None
    must_exclude_labels: list[str], optional
        Labels of variables that must be absent from a stay for it to be kept, by default None
    kept_entry_stages: list[int], optional
        List of valid stages for the first value of the target variable to be in, by default None (=all stages ok)
    kept_exit_stages: list[int], optional
        List of valid stages for the last value of the target variable to be in, by default None (=all stages ok)
    val_to_stage: str, optional
        Name used to retrieve a function that maps target variable values into stages (see the string_to_thresholding function in utils.py), by default None
    pred_horizon: int, optional
        Time (in hours) that must be able to exist between the last input obs and the tgt obs, by default 0
    pred_history: int, optional
        Time (in hours) after which at least one tgt obs must exist, by default 0
    max_hours: int, optional
        Maximum length (in hours) of a stay to keep, by default None (no limit)
    entry_above_exit: bool, optional
        Whether the entry stage must be strictly above the exit stage, by default None (all orders are ok)
    attr_in_paths: bool, optional
        Whether to augment the out_path with attribute values, by default False (see self.augment_path function)
    split_into_files: bool, optional
        Whether to split the culled .csv into individual stay files, by default False
    save_classif_labels: bool, optional
        Whether to save a dict of {stay_id: tgt_class} alongside the culled data, useful for stratified splitting. Note that the label is based on the last tgt obs of a stay. By default False
    """

    data_path: str
    key_path: str
    out_path: str
    tgt_var_label: str
    skippable_vars_labels: list[str] | None = None
    must_include_labels: list[str] | None = None
    must_exclude_labels: list[str] | None = None
    kept_entry_stages: list[int] | None = None  # Note: None means all are kept
    kept_exit_stages: list[int] | None = None
    val_to_stage: str | None = None
    pred_horizon: int = 0
    pred_history: int = 0
    max_hours: int | None = None
    entry_above_exit: bool | None = None
    attr_in_paths: bool = False
    split_into_files: bool = False
    save_classif_labels: bool = False

    def __post_init__(self):
        """Augment out_path with attributes if self.attr_in_paths. (Runs automatically after init.)"""
        if self.attr_in_paths:
            self.out_path = self._augment_path(self.out_path)

    def save_to_json(self, path: str | None = None):
        """Saves this config at `path` if provided, else in the same place as `self.out_path`"""
        if path is None:
            path, _ = splitext(self.out_path)

        super().save_to_json(path + ".json")

    def _augment_path(self, path: str) -> str:
        """Augment a path by adding all attributes that are not default to that path.

        Example:
        With out_path="out.csv", pred_horizon=10, pred_history=0, max_hours=None, kept_entry_stages=None, kept_exit_stages=[2, 3], entry_above_exit=False
        This will return "out_10ho23outUp.csv"

        Parameters
        ----------
        path : str
            The path to augment

        Returns
        -------
        str
            Augmented path, see example.
        """
        # TODO find a way to include tgt var and skipped vars ?
        # And also included and excluded labels?

        path_root, path_ext = splitext(path)

        # If default value, empty string
        # else number + letters to indicate which attr
        ho = choose_fallback(
            str(choose_fallback(self.pred_horizon, 0, "")) + "ho", "ho", ""
        )
        hi = choose_fallback(
            str(choose_fallback(self.pred_history, 0, "")) + "hi", "hi", ""
        )
        max = choose_fallback(
            str(choose_fallback(self.max_hours, None, "")) + "max",
            "max",
            "",
        )
        entry = choose_fallback(
            list_to_compact_str(choose_fallback(self.kept_entry_stages, None, ""))
            + "in",
            "in",
            "",
        )
        exit = choose_fallback(
            list_to_compact_str(choose_fallback(self.kept_exit_stages, None, ""))
            + "out",
            "out",
            "",
        )
        if self.entry_above_exit is None:
            trajectory = ""
        else:
            trajectory = "Down" if self.entry_above_exit else "Up"

        to_add = f"{ho}{hi}{max}{entry}{exit}{trajectory}"
        if to_add == "":
            # only add the underscore if we're adding things
            return path_root + path_ext
        return f"{path_root}_{to_add}{path_ext}"


@dataclass
class TrainingConfig(SavableConfig):
    """All parameters for a training of T-MITS

    Parameters
    ----------
    exp_name : str
        A name for the experiment. Will be turned into the folder where results are stored.
    data_path : str
        A path to the preprocessed .csv used for this experiment.
    dataset_key : str, optional
        Path to the key file associated with the dataset (see `get_ds_from_key` function), by default None. If provided, will override `dataset_preset`.
    n_folds : int
        The number of cross-validation folds to perform. If 1, do an 80/20 train/test split.
    train_batch_size : int
        Training `DataLoader` batch size.
    test_batch_size : int
        Testing `DataLoader` batch size.
    lr: float
        Adam initial learning rate.
    n_epochs: int
        Number of training epochs to do for each fold.
    n_var_embs: int
        Model parameter - How many different variables the model can encode. Automatically inferred if `dataset_key` is provided.
    dim_embed : int
        Model parameter - Embedding dimension.
    n_layers : int
        Model parameter - Number of Transformer layers.
    n_heads : int
        Model parameter - Nunmber of attention heads in the Transformer.
    dropout : float
        Model parameter - probability for each neuron to drop its value when forwarding.
    n_quantiles : int
        Model parameter - Nunmber of output quantiles of T-MITS. If 1, training will use the Huber loss. If >1, training will use the QuantileLoss. By default 3
    sched_factor : float
        Factor by which to multiply the learning rate when the test loss has not decreased for `sched_patience` epochs, by default 0.5
    sched_patience : int
        How many epochs of non-improvement of the test loss to wait before decreasing the learning rate, by default 3
    sched_per_batch : bool
        Whether to call scheduler.step() after each epoch or after each batch (useful for some schedulers). By default False.
    resume : bool
        Whether to resume training from a number of completed folds and a number of completed epochs in the last started fold.
    n_completed_folds : int, optional
        If `resume`, how many folds have already been completed and thus should not be redone. Ignored if `resume==False`. By default 0.
    n_completed_epochs : int, optional
        If `resume`, how many epochs have already been completed in the incompleted fold. Ignored if `resume==False`. By default 0.
    resuming_lr : float, optional
        If `resume`, starting learning rate. Ignored if `resume==False`. By default 0.00025.
    huber_delta : float, optional
        Delta parameter of the Huber loss (higher means higher penalty of large errors). By default 1.0.
    random_state : int, optional
        Randomness seed for the KFold generator. By default 1.
    lr_threshold : float, optional
        Learning rate threshold below (strict) which to stop training early. By default 1e-5.
    dataset_preset : str, optional
        A preset to get a `ICU_Reg` object. See the docstring of `get_ds_shorthand` to learn more. WARNING: Setting this will ignore most other datasets parameters; You will have to set them manually.
    tgt_var_label : str, optional
        Text label of the target variable (unused if dataset_key is not provided), by default None
    masked_var_labels : list[str], optional
        Text label of the masked variables, if any (unused if dataset_key is not provided), by default None
    config_file_name : str, optional
        File name for the saved config object, by default 'T_MITS_config.json'
    files_dir : str, optional
        If given, the directory where individual (per-stay) files can be found. By default None.
    back_interval : int
        The width of the observation window, in minutes before target time.
    max_measures : int, optional
        Maximum length of a sequence (crops the oldest measures if needed), by default None
    horizon : int, optional
        Prediction horizon in hours (tgt_time - cutoff_time), by default None
    histories : list[int], optional
        a list of history lengths (from start of seq) to try and keep (greedy), by default None
    verbose : bool, optional
        whether to print progress bars and time estimates to stdout during training (logging is unaffected), by default False
    do_eval_after : bool, optional
        Whether to launch the main function of the eval script after training is done, by default False
    """

    exp_name: str
    data_path: str
    dataset_key: str
    n_folds: int
    train_batch_size: int
    test_batch_size: int
    lr: float
    n_epochs: int
    n_var_embs: int
    dim_embed: int = 104
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1
    n_quantiles: int = 3
    sched_factor: float = 0.5
    sched_patience: int = 3
    sched_per_batch: bool = False
    resume: bool = False
    n_completed_folds: int = 0
    n_completed_epochs: int = 0
    resuming_lr: float = 0.00025
    huber_delta: float = 1.0
    random_state: int = 1
    lr_threshold: float = 1e-5
    dataset_preset: str | None = None
    tgt_var_label: str | None = None
    masked_var_labels: list[str] | None = None
    config_file_name: str = "T_MITS_config.json"
    files_dir: str | None = None
    back_interval: int | None = None
    max_measures: int | None = None
    horizon: int | None = None
    histories: list[int] | None = None
    verbose: bool = False
    do_eval_after: bool = False


@dataclass
class EvalConfig:
    """
    All parameters needed for model evaluation

    Parameters
    ----------
    exp_name : str
        The name of the experiment, expected to be located in a `./results/` folder
    do_train : bool, optional
        Whether to generate the training metrics alongside the testing metrics. By default False
    save_confusion_matrix : bool, optional
        Whether to save the classification confusion matrix. By default False
    save_true_pred : bool, optional
        Whether to save the y_true and y_pred numpy arrays. By default False
    save_xlsx : bool, optional
        Whether to save an `.xlsx` sheet with columns stay_id, true, pred, true_class, pred_class. By default False
    config_file_name : str, optional
        The .json training config filename. Should be located in `./results/exp_name/`. By default "T_MITS_config.json"
    """

    exp_name: str
    do_train: bool = False
    save_confusion_matrix: bool = False
    save_true_pred: bool = False
    save_xlsx: bool = False
    config_file_name: str = "T_MITS_config.json"
