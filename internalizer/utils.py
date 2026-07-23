import json
from scipy import sparse
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import List
import csv

from .filesystem_constants import DATA_DIR

LCIA_METHODS =  DATA_DIR / "lcia_ei310.json"
FILEPATH_NCVS = DATA_DIR / "NCVs.csv"
FILEPATH_CPI = DATA_DIR / "CPI_US.csv"
FILEPATH_PPP = DATA_DIR / "PPPdata.csv"
FILEPATH_HICP = DATA_DIR / "HICPdata.csv"
FILEPATH_COKEPROD_CORRECTION = DATA_DIR / "coke_production_ei310_correction.csv"
HICP = pd.read_csv(FILEPATH_HICP).set_index("year")
HICP = HICP["average HICP"]
PPP = pd.read_csv(FILEPATH_PPP).set_index("year")["PPP"]
CPI = pd.read_csv(FILEPATH_CPI).set_index("year")["CPI"]

METHOD2IC_MAPPING = DATA_DIR / "method2ic_mapping.csv"
COMPARTMENTS_CHANGE = DATA_DIR / "PM_compartments_change.csv"

BMATRIX_DTYPES = {
    "index of activity": np.int32,
    "index of biosphere flow": np.int32,
    "value": np.float64,
    "uncertainty type": np.int8,
    "loc": np.float64,
    "scale": np.float64,
    "shape": np.float64,
    "minimum": np.float64,
    "maximum": np.float64,
    "negative": np.int8,
    "flip": np.int8
}

AMATRIX_DTYPES = {
    "index of activity": np.int32,
    "index of product": np.int32,
    "value": np.float64,
    "uncertainty type": np.int8,
    "loc": np.float64,
    "scale": np.float64,
    "shape": np.float64,
    "minimum": np.float64,
    "maximum": np.float64,
    "negative": np.int8,
    "flip": np.int8
}

def get_lcia_method_names():
    """Get a list of available LCIA methods."""
    with open(LCIA_METHODS, "r") as f:
        data = json.load(f)

    return [" - ".join(x["name"]) for x in data]


def get_ncv_dict():
    ncvs = pd.read_csv(FILEPATH_NCVS)

    return ncvs.set_index(["dataset reference product", "dataset unit"])["NCV in MJ/product"].to_dict()


def convert_euros_to_dollar(year, target_year):
    return (HICP[target_year] / HICP[year]) / PPP[target_year]


def convert_dollars_to_euro(year, target_year):
    return (CPI[target_year] / CPI[year]) * PPP[target_year]


def format_lcia_method_exchanges(method):
    """
        Format LCIA method data to fit such structure:
        (name, unit, type, category, subcategory, amount, uncertainty type, uncertainty amount)
    -
        :param method: LCIA method
        :return: list of tuples
    """

    return {
        (
            x["name"],
            x["categories"][0],
            x["categories"][1] if len(x["categories"]) > 1 else "unspecified",
        ): x["amount"]
        for x in method["exchanges"]
    }


def get_lcia_methods(methods: list = None):
    """Get a list of available LCIA methods."""
    with open(LCIA_METHODS, "r") as f:
        data = json.load(f)

    if methods:
        data = [x for x in data if " - ".join(x["name"]) in methods]

    return {" - ".join(x["name"]): format_lcia_method_exchanges(x) for x in data}


def fill_characterization_factors_matrices(
    methods: list, biosphere_matrix_dict: dict, biosphere_dict: dict
) -> csr_matrix:
    """
    Create one CSR matrix for all LCIA methods, with the last dimension being the index of the method
    :param methods: contains names of the LCIA methods to use (e.g., ["IPCC 2021, Global wArming Potential"]).
    :param biosphere_matrix_dict: dictionary with biosphere flows and their indices in bw2calc's matrix
    :param biosphere_dict: dictionary with biosphere flows and their indices in the biosphere matrix (not bw2calc's matrix)
    :param debug: if True, log debug information
    :return: a sparse matrix with the characterization factors
    """

    lcia_data = get_lcia_methods(methods=methods)

    # Prepare data for efficient creation of the sparse matrix
    data = []
    rows = []
    cols = []
    cfs = []

    for m, method in enumerate(methods):
        method_data = lcia_data[method]

        for flow_name in method_data:
            if flow_name in biosphere_dict:
                idx = biosphere_dict[flow_name]
                if idx in biosphere_matrix_dict:
                    data.append(method_data[flow_name])
                    rows.append(biosphere_matrix_dict[idx])
                    cols.append(m)
                    cfs.append((method, flow_name, idx, method_data[flow_name]))

    # Efficiently create the sparse matrix
    matrix = sparse.csr_matrix(
        (data, (cols, rows)),
        shape=(len(methods), len(biosphere_matrix_dict)),
        dtype=np.float64,
    )

    return matrix


def check_monetization_factors(
        monetization: dict
) -> None:
    available_methods = get_lcia_method_names()
    for k in monetization.keys():
        if k not in available_methods:
            raise ValueError(f"Method {k} not available!")


def read_indices_csv(file_path: Path) -> dict[tuple[str, str, str, str], int]:
    """Parse a semicolon-separated index CSV into a lookup dictionary.

    :param file_path: Path to the CSV file containing activity metadata.
    :type file_path: pathlib.Path
    :returns: Mapping from ``(name, product, location, unit)`` tuples to indices.
    :rtype: dict[tuple[str, str, str, str], int]
    """
    indices = dict()
    with open(file_path, encoding="utf-8") as read_obj:
        csv_reader = csv.reader(read_obj, delimiter=";")
        for row in csv_reader:
            if row[4] == "index":
                continue
            try:
                indices[(row[0], row[1], row[2], row[3])] = int(row[4])
            except IndexError as err:
                print(
                    f"Error reading row {row} from {file_path}: {err}. "
                    f"Could it be that the file uses commas instead of semicolons?"
                )
    # remove any unicode characters
    indices = {tuple([str(x) for x in k]): v for k, v in indices.items()}
    return indices


def interpolate_and_weight_xr(
    data: dict,
    interpolation_years: list,
    weighting_factors: xr.DataArray | float
) -> xr.DataArray:
    # add padding    
    last_array = data[max(data.keys())]
    first_array = data[min(data.keys())]
    data[interpolation_years[0]] = first_array
    data[interpolation_years[-1]] = last_array

    # transform to DataArray
    x = xr.concat(
                list(data.values()),
                pd.Index(list(data.keys()), name="year")
            )
    
    # interpolate
    x = x.interp(year=interpolation_years)

    # weight
    return x * weighting_factors


def ramp(
    x: list | np.ndarray,
    a: float,
    b: float
) -> np.ndarray:
    if isinstance(x, list):
        x = np.array(x)

    return np.where(x <  a, 0, np.where(x > b, 1, (x-a)/(b-a)))


def get_linear_ramp_up(
    interpolation_years: list,
    start: int,
    end: int
) -> xr.DataArray:
    
    return xr.DataArray(
        ramp(interpolation_years, start, end),
        coords={
            "year": interpolation_years
        }
    )


def split_remind_index(
    df: pd.DataFrame,
    domains: List[str]
) -> pd.DataFrame:

    for i, domain in enumerate(domains):
        df[domain] = df["REMIND index"].apply(lambda x: x.split(" - ")[i])

    return df


def apply_filter_to_dataframe(
    df: pd.DataFrame,
    fltr: dict,
    msk: dict
) -> pd.DataFrame:
    if len(fltr) == 0 and len(msk) == 0:
        return None
    
    mask1 = False
    for col, slist in fltr.items():
        for s in slist:
            mask1 = mask1 | df[col].str.contains(s)

    mask2 = True
    for col, slist in msk.items():
        for s in slist:
            mask2 = mask2 & ~df[col].str.contains(s) 

    return df[mask1 & mask2]


def correct_coke_production_flows(matrixfolder: Path | str):
    # load data
    data = pd.read_csv(FILEPATH_COKEPROD_CORRECTION)

    # load matrix data
    Bmatrix = pd.read_csv(
        matrixfolder + "/B_matrix.csv",
        sep=";",
        dtype=BMATRIX_DTYPES
    )
    Bidx = pd.read_csv(matrixfolder + "/B_matrix_index.csv", sep=";", header=None,
                    names=["exchange name", "compartment", "subcompartment", "exchange unitName", "index of biosphere flow"])
    Aidx = pd.read_csv(matrixfolder + "/A_matrix_index.csv", sep=";", header=None,
                    names=["activityName", "reference product", "unit", "geography", "index of activity"])
    
    # set matrix indices
    flow_cols = ["exchange name", "compartment", "subcompartment"]
    act_cols = ["activityName", "reference product", "geography"]
    Bidx = Bidx.set_index(flow_cols)["index of biosphere flow"]
    Aidx = Aidx.set_index(act_cols)["index of activity"]
    Bmatrix = Bmatrix.set_index(["index of activity", "index of biosphere flow"])

    # create index for data selection and assignment
    flow_indices = Bidx.loc[pd.Index(data[flow_cols])].values
    act_indices = Aidx.loc[pd.Index(data[act_cols])].values
    idx = pd.MultiIndex.from_arrays([act_indices, flow_indices],
                                    names=["index of activity", "index of biosphere flow"])
    
    Bmatrix.loc[idx, "value"] = data["exchange amount - 3.10.1"].values
    Bmatrix.reset_index().to_csv(matrixfolder + "/B_matrix.csv", sep=";", index=False)


def change_compartments_PM(filepaths) -> None:
    """Change compartments in biosphere matrices specified by COMPARTMENTS_CHANGE.

    :returns: ``None``
    :rtype: None
    """
    changes = pd.read_csv(COMPARTMENTS_CHANGE).dropna(subset="new compartment")

    # select filepaths
    def select_filepath(keyword: str, fps):
        matches = [fp for fp in fps if keyword in fp.name]
        if not matches:
            raise FileNotFoundError(f"Expected file containing '{keyword}' not found.")
        return matches[0]

    # load indices and biosphere matrix
    Aidx = pd.read_csv(
        select_filepath(("A_matrix_index"), filepaths),
        sep=";",
        header=None,
        names=["name", "reference product", "unit", "location", "index"]
    )
    Bidx = pd.read_csv(
        select_filepath(("B_matrix_index"), filepaths),
        sep=";",
        header=None,
        names=["name", "compartment", "subcompartment", "unit", "index"]
    ).set_index(["name", "compartment", "subcompartment"])["index"]
    fp_biosphere = select_filepath(
        "B_matrix", [fp for fp in filepaths if "index" not in fp.name]
    )
    Bdata = pd.read_csv(fp_biosphere, sep=";", dtype=BMATRIX_DTYPES)

    # get needed biosphere indices
    pm_pollutants = [
        "Ammonia",
        "Nitrogen oxides",
        "Particulate Matter, > 2.5 um and < 10um",
        "Particulate Matter, < 2.5 um",
        "Sulfur dioxide",
        "Nitrate",
    ]

    # change compartments
    # one pollutant and dataset name at a time
    for pollutant in pm_pollutants:
        Bidx_all = Bidx.loc[pollutant, "air", :]
        for idx, row in changes.iterrows():
            act_idx = Aidx[Aidx["name"] == row["dataset name"]]["index"]
            b_idx_new = Bidx.loc[pollutant, "air", row["new compartment"]]
            Amask = Bdata["index of activity"].isin(act_idx)
            Bmask = Bdata["index of biosphere flow"].isin(Bidx_all)
            mask = Amask & Bmask
            Bdata.loc[mask, "index of biosphere flow"] = b_idx_new

    # remove redundancy in matrix
    aggfuncs = {
        "value": "sum",
        "uncertainty type": "first",
        "loc": "sum",
        "scale": "first",
        "shape": "first",
        "minimum": "first",
        "maximum": "first",
        "negative": "first",
        "flip": "first",
    }
    Bdata_new = (
        Bdata.groupby(["index of activity", "index of biosphere flow"])
        .agg(aggfuncs)
        .reset_index()
    )

    # save new matrix
    Bdata_new.to_csv(fp_biosphere, sep=";", index=False)


def get_dict_mapping_from_df(df, col1, col2):
    temp = df[[col1, col2]].copy().drop_duplicates()
    return dict(zip(temp[col1], temp[col2]))


def map_lcia_methods_to_impact_categories(
    df: pd.DataFrame,
    value_col: str = "cost"
) -> pd.DataFrame:
    """
    Map LCIA methods to impact categories based on keywords.
    :param df: DataFrame to be mapped
    """
    mapping = pd.read_csv(METHOD2IC_MAPPING).set_index("LCIA method")["impact_category"].to_dict()
    df["impact_category"] = df["impact_category"].map(lambda x: mapping.get(x, x))
    df = df.groupby([col for col in df.columns if col != value_col], as_index=False)[value_col].sum()


def get_automatic_exclude_list(
    methods: list,
    ics_exclude = ["climate change", "fossil resources"],
) -> list:
    """
    Get a list of impact categories to automatically exclude based on the LCIA methods used.
    :param methods: List of LCIA methods
    :return: List of impact categories to exclude
    """
    mapping = pd.read_csv(METHOD2IC_MAPPING).set_index("LCIA method")["impact_category"].to_dict()
    return [m for m in methods if mapping.get(m, m).lower() in ics_exclude]

