"""Various utility functions."""


def denorm(val, mean: float, std: float) -> float:
    """Denormalize `val` using `mean` and `std`"""
    if std != 0:
        return (val * std) + mean
    return val + mean


def denorm_list(vals, mean: float, std: float) -> list[float]:
    """Denormalize all values in vals"""
    return [denorm(val, mean, std) for val in vals]


def norm(val, mean: float, std: float) -> float:
    """Normalize `val` using `mean` and `std`"""
    if std != 0:
        return (val - mean) / std
    return val - mean


def descale(val, min: float, max: float) -> float:
    """Descale (minmax scaling) `val` using `min` and `max`"""
    if min != max:
        return (val * (max - min)) + min
    return val + min


def descale_list(vals, min: float, max: float) -> list[float]:
    """Descale (minmax scaling) all values in vals"""
    return [descale(val, min, max) for val in vals]


def scale(val, min: float, max: float) -> float:
    """Scale (minmax scaling) `val` using `min` and `max`"""
    if min != max:
        return (val - min) / (max - min)
    return val - min


def value_to_index(
    vals, cast_from_numpy=False, return_key=False, display_prog=False
) -> list[int]:
    """Turns a list of values into a list of indexes starting at 0.

    Example:
    [12, 23, 34, 12, 45, 34] -> [0, 1, 2, 0, 3, 2]

    Parameters
    ----------
    vals : Iterable[Any]
        The original iterable of values
    cast_from_numpy : bool, optional
        If the values in vals are numpy dtypes, whether to cast them to python dtypes, by default False
    return_key : bool, optional
        Whether to return the dict used as a key for the mapping, by default False
    display_prog : bool, optional
        Whether to display a progress bar (needs tqdm) over vals, by default False

    Returns
    -------
    list[int]
        The indexes after mapping.
    Tuple[list[int], dict[Any, int]]
        (indexes, key) if `return_key==True`
    """

    if display_prog:
        from tqdm import tqdm

        vals = tqdm(vals)

    d = {}
    indexes = []
    free_index = 0  # the lowest unused index
    for id in vals:
        if cast_from_numpy:
            id = id.item()
        if id not in d.keys():
            # if the id is new
            # allocate to the id the free index
            d[id] = free_index
            free_index += 1
        indexes.append(d[id])
    if return_key:
        return indexes, d
    return indexes


def string_to_thresholding(name: str):
    """Maps a name to a thresholding function.

    Current mapping:
    'KDIGO' -> `creat_to_4_stages`
    """
    if name == "KDIGO":
        return creat_to_4_stages
    else:
        raise ValueError(f"The implemented names are ['KDIGO'], got '{name}'.")


def creat_to_4_stages(value: float) -> int:
    """Converts creatinine values to renal risk/injury/failure stages
    according to the KDIGO criteria (https://doi.org/10.1186/2047-0525-1-6)

    Parameters
    ----------
    value : float
        Creatinine (serum) value, in mg/dL

    Returns
    -------
    int
        0: Normal; 1: Risk; 2: Injury; 3: Failure
    """
    if value < 1.35:
        return 0
    elif value < 2.68:
        return 1
    elif value < 4.16:
        return 2
    return 3


def reg_to_classif(reg_vals, custom_thresholds=None) -> list[int]:
    """Converts a list of regression values to classification values.

    If `custom_thresholds` is not None, use it as the function that maps real values to classes.

    Otherwise, we use creat_to_4_stages as a fallback.
    """
    if custom_thresholds is not None:
        return [custom_thresholds(val) for val in reg_vals]
    else:
        return [creat_to_4_stages(val) for val in reg_vals]


def init_logger(logger_name: str, log_file: str):
    """Creates a logger, configures the logging system and returns the logger

    Note : the "%d-%m %H:%M" date format results in `DAY-MONTH HOUR:MINUTES` timestamps
    (with hours in a 24 hours-per-day format)

    Parameters
    ----------
    logger_name : str
        The logger instance name.
    log_file : str
        A file path (usually ending in .log) where to output the logs.

    Returns
    -------
    logging.Logger
        A configured logger object.
    """
    import logging

    logger = logging.getLogger(logger_name)
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        style="{",
        format="{asctime} {levelname}:{message}",
        datefmt="%d-%m %H:%M",
        force=True,
    )
    return logger


def choose_fallback(value, special_case, fallback):
    """Returns `fallback` if `value` is equal to `special_case`.
    Simply returns `value` in all other cases.
    """
    if value is None and special_case is None:
        return fallback
    return value if value != special_case else fallback


def list_to_compact_str(input_list: list) -> str:
    """Turns a list into a non-spaced string of all its elements.

    Ex: [1, 2, 3] -> '123'
    """
    return (
        str(input_list)
        .replace(" ", "")
        .replace(",", "")
        .replace("[", "")
        .replace("]", "")
    )


def get_id_from_label(df_key, label: str) -> int:
    """Convert a given text label to the reindexed id created during preprocessing

    Assumes df_key is a pandas dataframe, with at least columns 'label' and 'key'."""
    try:
        return df_key[df_key["label"] == label]["key"].item()
    except ValueError as e:
        raise ValueError(
            f"{e}\n\nHint: Key '{label}' might have had either 0 or >1 matches in the key dataframe. Note that variable labels are case-sensitive."
        )
