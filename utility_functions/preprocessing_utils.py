"""Several utility variables and functions for preprocessing data."""

import polars as pl


def get_item_lazy(data_dir: str, itemid: int) -> pl.DataFrame:
    """Get the events for a particular itemid of icu/chartevents.csv

    Uses less RAM than loading the whole dataframe at once.

    Leverages the speed of polars LazyFrames + batching, but will still be slower for most items.
    """
    return (
        pl.scan_csv(data_dir + "icu/chartevents.csv")
        .select(
            [
                "hadm_id",
                "subject_id",
                "charttime",
                "itemid",
                "valuenum",
            ]
        )
        .filter(pl.col("itemid").is_in([itemid]))
        .with_columns(
            pl.col("charttime").str.strptime(pl.Datetime, format="%Y-%m-%d %T")
        )
        .rename({"valuenum": "value"})
        .collect(streaming=True)
    )


def get_item_from_full(df: pl.DataFrame, itemid: int) -> pl.DataFrame:
    """Filters the 'itemid' column of `df` to keep only the specified `itemid`"""
    return df.filter(pl.col("itemid").is_in([itemid]))


def remove_outside_range(df: pl.DataFrame, low: float, high: float) -> pl.DataFrame:
    """Remove from `df` the values strictly below `low` and strictly above `high`"""
    return df.filter((pl.col("value") >= low) & (pl.col("value") <= high))


def remove_outliers(df: pl.DataFrame) -> pl.DataFrame:
    """Remove outliers from `df` using 3*IQR as the acceptable range (wide by design).

    Expects 'value' to be the column containing the measures.
    """
    q1, q3 = (
        df.select(pl.quantile("value", 0.25)).item(),
        df.select(pl.quantile("value", 0.75)).item(),
    )

    iqr = q3 - q1

    return remove_outside_range(df, low=q1 - 3 * iqr, high=q3 + 3 * iqr)


def add_index_from_col(
    df: pl.DataFrame,
    col_name: str,
    index_name: str,
    return_key=False,
    cast_to_python: bool = False,
    verbose: bool = False,
) -> pl.DataFrame:
    """Add an index column to a dataframe using values from another column.

    The index is based on the unique values in the `col_name` column

    For example, if the `col_name` column contains the values ["A", "B", "C", 'A", "D", "B", "E"],
    the new `index_name` column will contain [0, 1, 2, 0, 3, 1, 4],
    and the key (if returned) will be {"A":0, "B":1, "C":2, "D":3, "E":4}


    Parameters
    ----------
    df : pl.DataFrame
        The dataframe to be modified
    col_name : str
        Name of the values column
    index_name : str
        Name of the index column
    return_key : bool, optional
        Whether to return the key used for the value-to-index replacement, by default False
    cast_from_numpy : bool, optional
        If the values in vals are package-specific dtypes, whether to cast them to python dtypes, by default False
    verbose : bool, optional
        Whether to print each replacement, by default False

    Returns
    -------
    pl.DataFrame(, dict[int, int])
        The dataframe with the new indexing column (and the replacement key if `return_key`)
    """
    unique_vals = (
        df.select(pl.col(col_name)).unique(maintain_order=True).to_numpy().flatten()
    )

    val_to_ind = {}
    free_index = 0  # the lowest unused index
    for val in unique_vals:
        if cast_to_python:
            native_val = val.item()  # cast to native python type
        else:
            native_val = val
        if native_val not in val_to_ind.keys():
            if verbose:
                print(
                    f"Encountered new val {native_val}; will get replaced with {free_index}."
                )
            # if the id is new
            # allocate the free index to the id
            val_to_ind[native_val] = free_index
            free_index += 1

    if return_key:
        return (
            df.with_columns(
                pl.col(col_name)
                .replace_strict(val_to_ind, return_dtype=pl.Int32)
                .alias(index_name)
            ),
            val_to_ind,
        )

    return df.with_columns(
        pl.col(col_name)
        .replace_strict(val_to_ind, return_dtype=pl.Int32)
        .alias(index_name)
    )


def load_chartevents(data_dir: str) -> pl.DataFrame:
    """Loads chartevents.csv in a DataFrame, keeping only the useful columns.

    Also parses the dates for the `charttime` column, expected in the format `YYYY-MM-DD HH-mm-SS`
    (example: 2157-11-20 19:54:00)

    Parameters
    ----------
    data_dir : str
        The path to the mimic-iv folder, which should contain `icu/chartevents.csv` (unzipped).

    Returns
    -------
    pl.DataFrame
        Chartevents dataframe containing the following columns: [hadm_id, subject_id, charttime, itemid, value]
    """
    return (
        pl.scan_csv(
            data_dir + "icu/chartevents.csv",
        )
        .select(
            [
                "hadm_id",
                "stay_id",
                "subject_id",
                "charttime",
                "itemid",
                "valuenum",
            ]
        )
        .drop_nulls()
        .with_columns(
            pl.col("charttime").str.strptime(pl.Datetime, format="%Y-%m-%d %T"),
            pl.col("hadm_id").cast(pl.Int64),
        )
        .rename({"valuenum": "value"})
        .collect()  # materialize the LazyFrame into a DataFrame
    )


def load_datetimeevents(data_dir: str) -> pl.DataFrame:
    """Like `load_chartevents` but for datetimeevents.csv."""
    # Note: the 'value' column of the datetimeevents.csv is a date.
    # We convert it to the number of minutes since the earliest charttime of that stay.

    df = (
        pl.read_csv(
            data_dir + "icu/datetimeevents.csv",
            columns=[
                "hadm_id",
                "stay_id",
                "subject_id",
                "charttime",
                "itemid",
                "value",
            ],
        )
        .drop_nulls()
        .with_columns(
            pl.col("charttime").str.strptime(pl.Datetime, format="%Y-%m-%d %T"),
            pl.col("value").str.strptime(pl.Datetime, format="%Y-%m-%d %T"),
            pl.col("hadm_id").cast(pl.Int64),
        )
    )

    # Calculate the earliest charttime for each admission
    min_charttimes = df.group_by("hadm_id").agg(
        pl.col("charttime").min().alias("min_charttime")
    )

    # Join back to original DataFrame
    df_with_min = df.join(min_charttimes, on="hadm_id")

    # Calculate the difference in minutes between event time and min_charttime
    # and drop the temporary min_charttime column
    return df_with_min.with_columns(
        (pl.col("value") - pl.col("min_charttime"))
        .cast(pl.Duration)
        .dt.total_minutes()
        .alias("value")
    ).drop("min_charttime")


def load_inputevents(data_dir: str) -> pl.DataFrame:
    """Like `load_chartevents` but for inputevents.csv."""
    # Rename starttime to charttime and amount to value for consistency
    # Only use the total amount and not the input duration for now, as it would require a specific embedding
    # (still coded it, see commented lines)
    return (
        pl.read_csv(
            data_dir + "icu/inputevents.csv",
            columns=[
                "hadm_id",
                "stay_id",
                "subject_id",
                "starttime",
                "itemid",
                "amount",
                # "endtime",
            ],
        )
        .drop_nulls()
        .with_columns(
            pl.col("starttime")
            .str.strptime(pl.Datetime, format="%Y-%m-%d %T")
            .alias("charttime"),
            pl.col("hadm_id").cast(pl.Int64),
            # pl.col("endtime").str.strptime(pl.Datetime, format="%Y-%m-%d %T"),
        )
        # .with_columns(
        #     (pl.col("endtime") - pl.col("charttime"))
        #     .cast(pl.Duration)
        #     .dt.total_minutes()
        #     .alias("duration")
        # )
        # .drop("endtime")
        .rename({"amount": "value"})
        .drop("starttime")
    )


def load_outputevents(data_dir: str) -> pl.DataFrame:
    """Like `load_chartevents` but for outputevents.csv."""
    return (
        pl.read_csv(
            data_dir + "icu/outputevents.csv",
            columns=[
                "hadm_id",
                "stay_id",
                "subject_id",
                "charttime",
                "itemid",
                "value",
            ],
            # Override needed as the value column only presents its first float late in the table
            schema_overrides={"value": pl.Float64},
        )
        .drop_nulls()
        .with_columns(
            pl.col("charttime").str.strptime(pl.Datetime, format="%Y-%m-%d %T"),
            pl.col("hadm_id").cast(pl.Int64),
        )
    )


def load_procedureevents(data_dir: str) -> pl.DataFrame:
    """Like `load_chartevents` but for procedureevents.csv."""
    return (
        pl.read_csv(
            data_dir + "icu/procedureevents.csv",
            columns=[
                "hadm_id",
                "stay_id",
                "subject_id",
                "starttime",
                "itemid",
                "value",
            ],
        )
        .drop_nulls()
        .with_columns(
            pl.col("starttime")
            .str.strptime(pl.Datetime, format="%Y-%m-%d %T")
            .alias("charttime"),
            pl.col("hadm_id").cast(pl.Int64),
        )
        .drop("starttime")
    )


def load_labevents(data_dir: str) -> pl.DataFrame:
    return (
        pl.scan_csv(
            data_dir + "hosp/labevents.csv",
        )
        .select(
            [
                "subject_id",
                "hadm_id",
                "charttime",
                "itemid",
                "valuenum",
            ]
        )
        .drop_nulls()
        .with_columns(
            pl.col("charttime").str.strptime(pl.Datetime, format="%Y-%m-%d %T"),
            pl.col("hadm_id").cast(pl.Int64),
        )
        .rename({"valuenum": "value"})
        .collect()
    )


# The 29 items manually selected, with their names and itemids
# See d_items.csv in MIMIC-IV for more details
bottom_up_items = {
    "Creatinine (serum)": 220615,
    "Heart rate": 220045,
    "Arterial Blood Pressure systolic": 220050,
    "Arterial Blood Pressure diastolic": 220051,
    "Arterial Blood Pressure mean": 220052,
    "Temperature Fahrenheit": 223761,
    "Daily Weight": 224639,
    "Admission Weight (Kg)": 226512,
    "WBC": 220546,
    "Sodium (serum)": 220645,
    "Potassium (serum)": 227442,
    "PH (Arterial)": 223830,
    "Respiratory Rate": 220210,
    "Apnea Interval": 223876,
    "Minute Volume": 224687,
    "Central Venous Pressure": 220074,
    "Inspired O2 Fraction": 223835,
    "Blood Flow (ml/min)": 224144,  # Dialysis
    "BUN": 225624,  # Blood Urea Nitrogen
    "Platelet Count": 227457,
    "Lactic Acid": 225668,
    "O2 saturation pulseoxymetry": 220277,
    "Hemoglobin": 220228,
    "Albumin": 227456,
    "Anion gap": 227073,
    "Prothrombin time": 227465,
    "Arterial O2 pressure": 220224,
    "Height (cm)": 226730,
    "Glucose (serum)": 220621,
}

# Items selected as the top 206 most frequently measured items in MIMIC-IV (excluding text variables)
# 206 features means min 50_000 points per feature
# Note: this was computed automatically
# See d_items.csv in MIMIC-IV for more details
top_down_items = {
    "Heart Rate": 220045,
    "Respiratory Rate": 220210,
    "O2 saturation pulseoxymetry": 220277,
    "Non Invasive Blood Pressure systolic": 220179,
    "Non Invasive Blood Pressure diastolic": 220180,
    "Non Invasive Blood Pressure mean": 220181,
    "Arterial Blood Pressure mean": 220052,
    "Arterial Blood Pressure systolic": 220050,
    "Arterial Blood Pressure diastolic": 220051,
    "Temperature Fahrenheit": 223761,
    "Alarms On": 224641,
    "Parameters Checked": 224168,
    "ST Segment Monitoring On": 228305,
    "Inspired O2 Fraction": 223835,
    "Central Venous Pressure": 220074,
    "Glucose finger stick (range 70-100)": 225664,
    "PEEP set": 220339,
    "Heart Rate Alarm - Low": 220047,
    "Heart rate Alarm - High": 220046,
    "O2 Saturation Pulseoxymetry Alarm - Low": 223770,
    "O2 Saturation Pulseoxymetry Alarm - High": 223769,
    "Resp Alarm - High": 224161,
    "Resp Alarm - Low": 224162,
    "Tidal Volume (observed)": 224685,
    "Minute Volume": 224687,
    "SpO2 Desat Limit": 226253,
    "Mean Airway Pressure": 224697,
    "Respiratory Rate (spontaneous)": 224689,
    "Peak Insp. Pressure": 224695,
    "O2 Flow": 223834,
    "Minute Volume Alarm - Low": 220292,
    "Minute Volume Alarm - High": 220293,
    "Apnea Interval": 223876,
    "Paw High": 223873,
    "Vti High": 223874,
    "Respiratory Rate (Total)": 224690,
    "Fspn High": 223875,
    "20 Gauge Dressing Occlusive": 227368,
    "20 Gauge placed in outside facility": 226138,
    "Non-Invasive Blood Pressure Alarm - Low": 223752,
    "Non-Invasive Blood Pressure Alarm - High": 223751,
    "Potassium (serum)": 227442,
    "Sodium (serum)": 220645,
    "Chloride (serum)": 220602,
    "20 Gauge placed in the field": 228100,
    "Hematocrit (serum)": 220545,
    "Hemoglobin": 220228,
    "Creatinine (serum)": 220615,
    "HCO3 (serum)": 227443,
    "BUN": 225624,
    "Anion gap": 227073,
    "Arterial Line placed in outside facility": 226107,
    "Glucose (serum)": 220621,
    "Magnesium": 220635,
    "Inspired Gas Temp.": 223872,
    "Phosphorous": 225677,
    "Platelet Count": 227457,
    "Calcium non-ionized": 225625,
    "Multi Lumen placed in outside facility": 226113,
    "Subglottal Suctioning": 226169,
    "WBC": 220546,
    "18 Gauge Dressing Occlusive": 227367,
    "18 Gauge placed in outside facility": 226137,
    "Respiratory Rate (Set)": 224688,
    "Tidal Volume (set)": 224684,
    "Arterial Line Dressing Occlusive": 227292,
    "Inspiratory Time": 224738,
    "Tidal Volume (spontaneous)": 224686,
    "PSV Level": 224701,
    "PH (Arterial)": 223830,
    "Pulmonary Artery Pressure diastolic": 220060,
    "Pulmonary Artery Pressure systolic": 220059,
    "Arterial O2 pressure": 220224,
    "Arterial CO2 Pressure": 220235,
    "Arterial Base Excess": 224828,
    "TCO2 (calc) Arterial": 225698,
    "Pulmonary Artery Pressure mean": 220061,
    "18 Gauge placed in the field": 228099,
    "PTT": 227466,
    "Multi Lumen Dressing Occlusive": 227293,
    "INR": 227467,
    "Prothrombin time": 227465,
    "Temperature Celsius": 223762,
    "Eye Care": 225184,
    "Back Care": 225187,
    "Skin Care": 225185,
    "Impaired Skin Odor #1": 224564,
    "PICC Line placed in outside facility": 226115,
    "Lactic Acid": 225668,
    "ART BP Mean": 225312,
    "ART BP Systolic": 225309,
    "ART BP Diastolic": 225310,
    "High risk (>51) interventions": 227349,
    "Ultrafiltrate Output": 226457,
    "Plateau Pressure": 224696,
    "Ionized Calcium": 225667,
    "Expiratory Ratio": 226871,
    "Inspiratory Ratio": 226873,
    "Hourly Patient Fluid Removal": 224191,
    "Cough/Deep Breath": 225188,
    "Access Pressure": 224149,
    "Filter Pressure": 224150,
    "Current Goal": 225183,
    "Effluent Pressure": 224151,
    "Return Pressure": 224152,
    "Daily Weight": 224639,
    "Dialysate Rate": 224154,
    "Arterial Blood Pressure Alarm - Low": 220056,
    "Arterial Blood Pressure Alarm - High": 220058,
    "Replacement Rate": 224153,
    "Blood Flow (ml/min)": 224144,
    "Ventilator Tank #1": 227565,
    "Ventilator Tank #2": 227566,
    "Flow Rate (L/min)": 224691,
    "Total PEEP Level": 224700,
    "PBP (Prefilter) Replacement Rate": 228005,
    "Post Filter Replacement Rate": 228006,
    "Arterial Line Zero/Calibrate": 225210,
    "Citrate (ACD-A)": 228004,
    "Intra Cranial Pressure": 220765,
    "PICC Line Dressing Occlusive": 227358,
    "SvO2": 223772,
    "Blood Temperature CCO (C)": 226329,
    "Glucose (whole blood)": 226537,
    "Cardiac Output (CCO)": 224842,
    "22 Gauge Dressing Occlusive": 227369,
    "22 Gauge placed in outside facility": 226139,
    "Insulin pump": 228236,
    "Trans Membrane Pressure": 229247,
    "Pressure Drop": 229248,
    "Intravenous  / IV access prior to admission": 225103,
    "Impaired Skin Odor #2": 224923,
    "Potassium (whole blood)": 227464,
    "Cerebral Perfusion Pressure": 227066,
    "22 Gauge placed in the field": 228101,
    "Admission Weight (lbs.)": 226531,
    "Cuff Pressure": 224417,
    "Total Bilirubin": 225690,
    "ALT": 220644,
    "AST": 220587,
    "Incentive Spirometry": 225189,
    "Alkaline Phosphate": 225612,
    "Heparin Dose (per hour)": 224145,
    "Self ADL": 225092,
    "Home TF": 228648,
    "Pressure Ulcer Present": 228649,
    "History of slips / falls": 225094,
    "GI #1 Tube Mark (CM)": 229300,
    "Impaired Skin Length #1": 224562,
    "ETOH": 225106,
    "Impaired Skin Width #1": 224846,
    "16 Gauge Dressing Occlusive": 227366,
    "16 Gauge placed in outside facility": 226136,
    "Dialysis Catheter placed in outside facility": 226118,
    "Visual / hearing deficit": 225087,
    "Difficulty swallowing": 225118,
    "EtCO2": 228640,
    "Arterial O2 Saturation": 220227,
    "Dialysis patient": 225126,
    "Currently experiencing pain": 225113,
    "Recreational drug use": 225110,
    "Is the spokesperson the Health Care Proxy": 225067,
    "MDI #1 Puff": 224169,
    "Special diet": 225122,
    "16 Gauge placed in the field": 228098,
    "Any fear in relationships": 225074,
    "Social work consult": 225078,
    "Unintentional weight loss >10 lbs.": 225124,
    "PICC Line Power PICC": 229476,
    "Impaired Skin Odor #3": 224924,
    "Temporary Pacemaker Rate": 224751,
    "Temporary Pacemaker Wires Venticular": 223962,
    "Central Venous Pressure Alarm - High": 220072,
    "Central Venous Pressure  Alarm - Low": 220073,
    "Sexuality / reproductive problems": 226180,
    "Admission Weight (Kg)": 226512,
    "Temporary Pacemaker Wires Atrial": 224839,
    "Emotional / physical / sexual harm by partner or close relation": 225076,
    "Pminimum": 229662,
    "LDH": 220632,
    "PH (Venous)": 220274,
    "Multi Lumen Zero/Calibrate": 225206,
    "Dialysis Catheter Dressing Occlusive": 227357,
    "Bed Bath": 225313,
    "Pressure ulcer #1- Length": 228723,
    "Pinsp (Hamilton)": 229663,
    "VEN Lumen Volume": 224406,
    "Pressure Ulcer #1- Width": 228620,
    "ART Lumen Volume": 224404,
    "RCexp (Measured Time Constant)": 229660,
    "Compliance": 229661,
    "Resistance Exp": 229664,
    "Resistance Insp": 229665,
    "Cordis/Introducer Dressing Occlusive": 227350,
    "Impaired Skin Length #2": 224916,
    "Albumin": 227456,
    "Hematocrit (whole blood - calc)": 226540,
    "Impaired Skin Width #2": 224951,
    "Troponin-T": 227429,
    "Cordis/Introducer placed in outside facility": 226109,
    "Unable to assess psychological": 225070,
    "Differential-Lymphs": 225641,
    "Differential-Monos": 225642,
    "Differential-Basos": 225639,
    "Differential-Eos": 225640,
    "Differential-Neuts": 225643,
}

# Labs items included in the eICU preprocessing
eicu_included_labs = [
    "creatinine",
    "WBC x 1000",
    "sodium",
    "potassium",
    "pH",
    "FiO2",  # Inspired O2 fraction
    "BUN",
    "platelets x 1000",
    "lactate",
    "Hgb",
    "albumin",
    "anion gap",
    "PT - INR",  # Prothrombin time - International normalized ratio
    "glucose",
    "bedside glucose",
    "chloride",
    "HCO3",
    "magnesium",
    "phosphate",
    "calcium",
    "paCO2",  # Arterial CO2 pressure
    "PTT",  # Partial thromboplastin time
    "ionized calcium",
    "PEEP",  # Positive End-Expiratory Pressure
    "total bilirubin",
    "ALT (SGPT)",  # Alanine transaminase (used to be called serum glutamic-pyruvic transaminase)
    "AST (SGOT)",  # aspartate aminotransferase (serum glutamic-oxaloacetic transaminase)
    "alkaline phos.",
    "LDH",  # Lactate Dehydrogenase
    # Differentials
    "-lymphs",
    "-monos",
    "-basos",
    "-eos",
]

eicu_included_periodic = [
    "temperature",
    "heartrate",
    "respiration",
    "cvp",
    "sao2",
    "systemicsystolic",
    "systemicdiastolic",
    "systemicmean",
    "pamean",
]
