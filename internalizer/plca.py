from premise import NewDatabase
import bw2data as bd
import bw_processing as bwp
from bw_processing import Datapackage
from bw2calc import MultiLCA
import xarray as xr
import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import List, Optional, Tuple

from .regionalization import regionalize_costs, regionalize_impacts
from .utils import (
    fill_characterization_factors_matrices,
    get_ncv_dict,
    get_lcia_method_names,
    csr_matrix,
    correct_coke_production_flows,
    change_compartments_PM,
    read_indices_csv,
    get_shares_adjustments,
)
from .filesystem_constants import DATA_DIR

FILEPATH_MONETIZATION_FACTORS = DATA_DIR / "mfs_MCsample_EUR2022.nc"
FILEPATH_MONETIZATION_FACTORS_PERSPECTIVES = DATA_DIR / "mfs_perspectives_EUR2022.nc"
NCV_DICT = get_ncv_dict()

GAINS_MASKS = [
    "burned in container ship",
    "smelting of copper concentrate, sulfide ore",
    "heat production, hardwood chips from forest, at furnace 50kW",
    "heat production, mixed logs, at wood heater"
]

ALL_INTERVENTIONS = [
    "tailings",
    "slags",
    "copper",
    # "brake_wear",
    "smelting",
    "woodstoves",
    "shipping",
]

DEFAULT_PREMISE_KWARGS = {
    "keep_imports_uncertainty": True,
    "fleet_regionalization": "global"
}

SECTOR_UPDATES = [
    "biomass",
    "electricity",
    "cement",
    "steel" ,
    "fuels",
    "renewable",
    "metals",
    # "mining",
    "interventions",
    "heat",
    "cdr",
    "battery",
    "emissions",
    "cars",
    "two_wheelers",
    "trucks",
    "ships",
    "buses",
    "trains",
    "final energy",
    # "capacity",
#    "external",
]


def load_matrix_and_index(
    file_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a sparse matrix representation and uncertainties from a CSV export.

    :param file_path: CSV file containing row, column, value, and distribution columns.
    :type file_path: pathlib.Path
    :returns: Tuple of data values, index pairs, sign flags, and distribution metadata.
    :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """
    # Load the data from the CSV file
    array = np.genfromtxt(file_path, delimiter=";", skip_header=1)

    # give `indices_array` a list of tuples of indices
    indices_array = np.array(
        list(zip(array[:, 1].astype(int), array[:, 0].astype(int))),
        dtype=bwp.INDICES_DTYPE,
    )

    data_array = array[:, 2]

    # make a boolean scalar array to store the sign of the data
    flip_array = array[:, -1].astype(bool)

    distributions_array = np.array(
        list(
            zip(
                array[:, 3].astype(int),  # uncertainty type
                array[:, 4].astype(float),  # loc
                array[:, 5].astype(float),  # scale
                array[:, 6].astype(float),  # shape
                array[:, 7].astype(float),  # minimum
                array[:, 8].astype(float),  # maximum
                array[:, 9].astype(bool),  # negative
            )
        ),
        dtype=bwp.UNCERTAINTY_DTYPE,
    )

    return data_array, indices_array, flip_array, distributions_array


def get_lca_matrices(
    filepaths: list,
    model: str,
    scenario: str,
    year: int,
    remove_uncertainty: bool = True,
    change_pm_compartments: bool = False,
) -> tuple[
    Datapackage,
    dict[tuple[str, str, str, str], int],
    dict[tuple, int],
    list[tuple] | None,
    dict | None,
]:
    """Retrieve the technosphere and biosphere matrices plus indices for a scenario.

    :param filepaths: Candidate CSV file paths bundled in the datapackage.
    :type filepaths: list[str]
    :param model: Name of the IAM model to filter for.
    :type model: str
    :param scenario: Pathway identifier to match in filenames.
    :type scenario: str
    :param remove_uncertainty: When ``True``, zero out distribution parameters.
    :type remove_uncertainty: bool
    :param change_pm_compartments: When ``True``, adjust PM compartments in the matrices.
    :type change_pm_compartments: bool
    :returns: Datapackage with LCI matrices, technosphere/biosphere indices
    :rtype: tuple[bw_processing.Datapackage, dict, dict]
    :raises FileNotFoundError: If expected matrix files cannot be located.
    :raises ValueError: When the set of candidate files does not match expectations.
    """

    # find the correct filepaths in filepaths
    # the correct filepath are the strings that contains
    # the model, scenario and year
    def filter_filepaths(suffix: str, contains: List[str]):
        return [
            Path(fp)
            for fp in filepaths
            if all(kw in fp.replace(" ", "") for kw in contains)
            and Path(fp).suffix == suffix
            and Path(fp).exists()
        ]

    def select_filepath(keyword: str, fps):
        matches = [fp for fp in fps if keyword in fp.name]
        if not matches:
            raise FileNotFoundError(f"Expected file containing '{keyword}' not found.")
        return matches[0]

    fps = filter_filepaths(
        suffix=".csv",
        contains=[model, f"/{str(year)}/"] + scenario.replace(" ", "").split("-"),
    )
    if len(fps) != 4:
        raise ValueError(
            f"Expected 4 filepaths, got {len(fps)} when looking at {filepaths} for terms: {model}, {scenario}, {year}"
        )

    # if change_pm_compartments:
    #     change_compartments_PM(fps)

    fp_technosphere_inds = select_filepath("A_matrix_index", fps)
    fp_biosphere_inds = select_filepath("B_matrix_index", fps)
    technosphere_inds = read_indices_csv(fp_technosphere_inds)
    biosphere_inds = read_indices_csv(fp_biosphere_inds)
    # remove the last element of the tuple, which is the index
    biosphere_inds = {k[:-1]: v for k, v in biosphere_inds.items()}

    dp = bwp.create_datapackage()

    fp_A = select_filepath("A_matrix", [fp for fp in fps if "index" not in fp.name])
    fp_B = select_filepath("B_matrix", [fp for fp in fps if "index" not in fp.name])

    # Load matrices and add them to the datapackage
    uncertain_parameters = None
    for matrix_name, fp in [("technosphere_matrix", fp_A), ("biosphere_matrix", fp_B)]:
        data, indices, sign, distributions = load_matrix_and_index(fp)

        # remove uncertainty data
        if remove_uncertainty is True:
            distributions = np.array(
                [
                    (0, None, None, None, None, None, False)
                    for _ in range(len(distributions))
                ],
                dtype=bwp.UNCERTAINTY_DTYPE,
            )

        dp.add_persistent_vector(
            matrix=matrix_name,
            indices_array=indices,
            data_array=data,
            flip_array=sign if matrix_name == "technosphere_matrix" else None,
            distributions_array=distributions,
        )

    return dp, technosphere_inds, biosphere_inds


def get_cfs_and_mfs(
    monetization: float | str | dict,
    lca: MultiLCA,
    biosphere_inds: dict
) -> Tuple[csr_matrix, xr.DataArray, list]:
    methods = get_lcia_method_names()
    if isinstance(monetization, float):
        mfs = xr.load_dataarray(FILEPATH_MONETIZATION_FACTORS)
        methods = list(mfs.coords["LCIA method"].values)
        cfs = fill_characterization_factors_matrices(
            methods=methods,
            biosphere_matrix_dict=lca.dicts.biosphere,
            biosphere_dict=biosphere_inds
        )
        return cfs, mfs, methods 
    elif isinstance(monetization, str):
        mfs = xr.load_dataarray(FILEPATH_MONETIZATION_FACTORS_PERSPECTIVES).sel(
            {"perspective": monetization}
        )
        methods = list(mfs.coords["LCIA method"].values)
        cfs = fill_characterization_factors_matrices(
            methods=methods,
            biosphere_matrix_dict=lca.dicts.biosphere,
            biosphere_dict=biosphere_inds
        )
        return cfs, mfs, methods
    else:
        methods = list(monetization.keys())
        cfs = fill_characterization_factors_matrices(
            methods=methods,
            biosphere_matrix_dict=lca.dicts.biosphere,
            biosphere_dict=biosphere_inds
        )
        mfs = xr.DataArray(
            np.diag(list(monetization.values())),
            {
                "LCIA method": methods,
                "impact category": methods
            }
        )
        return cfs, mfs, methods
    

def get_monetized_results(
    lca: MultiLCA,
    selected_inds: dict,
    biosphere_inds: dict,
    monetization: float | str | dict
) -> pd.DataFrame:
    # get characterization and monetization factors
    cfs, mfs, methods = get_cfs_and_mfs(monetization, lca, biosphere_inds)
    quantile = None
    if isinstance(monetization, float):
        quantile = monetization
    
    dflist = []
    dflist_impacts = []
    for k, value in lca.inventories.items():
        data = np.squeeze(np.array((cfs @ value).sum(axis=-1)))
        if len(methods) == 1:
            impacts = xr.DataArray(
                [data],
                {
                    "LCIA method": methods
                }
            )
        else:
            impacts = xr.DataArray(
                data,
                {
                    "LCIA method": methods
                }
            )
        
        costs = (mfs * impacts).sum(dim="LCIA method")
        if quantile is not None:
            a = costs.to_numpy()
            costs = xr.DataArray(
                np.where(np.all(a < 0, axis=1),
                            np.quantile(a, 1-quantile, axis=1),
                            np.quantile(a, quantile, axis=1)),
                {
                    "impact category": list(costs.coords["impact category"].values),
                }
            )
        df = costs.to_dataframe(name="cost").reset_index()
        name, refprod, unit, location = list(selected_inds)[int(k)]
        df["dataset name"] = name
        df["dataset reference product"] = refprod
        df["dataset unit"] = unit
        df["dataset location"] = location
        dflist.append(df)

        df_impacts = impacts.to_dataframe(name="impact").reset_index()
        df_impacts["dataset name"] = name
        df_impacts["dataset reference product"] = refprod
        df_impacts["dataset unit"] = unit
        df_impacts["dataset location"] = location
        dflist_impacts.append(df_impacts)

    all_costs = pd.concat(dflist, ignore_index=True)[
        ["dataset name", "dataset reference product", "dataset unit", 
        "dataset location", "impact category", "cost"]
    ]
    all_impacts = pd.concat(dflist_impacts, ignore_index=True)[
        ["dataset name", "dataset reference product", "dataset unit", 
        "dataset location", "LCIA method", "impact"]
    ]

    return all_costs, all_impacts
   

def _run_premise_year(
    project: str,
    scen: dict,
    ei_version: str,
    source_version: str,
    outdir: str,
    quiet: bool,
    include_interventions: bool,
    change_pm_compartments: bool,
) -> None:
    bd.projects.set_current(project)

    ei_label = "ecoinvent-{}-cutoff".format(ei_version)

    kwargs = {
        "scenarios": [scen],
        "source_db": ei_label,
        "source_version": source_version,
        "biosphere_name": "ecoinvent-{}-biosphere".format(ei_version),
        "quiet": quiet
    }
    kwargs.update(DEFAULT_PREMISE_KWARGS)
    if include_interventions:
        kwargs["metals_scenario"] = "intervention"
        kwargs["intervention_scenarios"] = {iv: "intervention" for iv in ALL_INTERVENTIONS}
        kwargs["gains_masks"] = GAINS_MASKS
        kwargs["shares_adjustments"] = get_shares_adjustments("all:intervention")
    
    ndb = NewDatabase(**kwargs)
    ndb.update(sectors=SECTOR_UPDATES)
    ndb.write_db_to_matrices(outdir)

    if ei_version == "3.10":
        model = scen["model"]
        scenario = scen["pathway"]
        year = scen["year"]
        mfolder = outdir + f"/{model}/{scenario}/{str(year)}/"
        correct_coke_production_flows(mfolder)

    if change_pm_compartments:
        model = scen["model"]
        scenario = scen["pathway"]
        year = scen["year"]
        matrix_folder = outdir + f"/{model}/{scenario}/{str(year)}/"
        fps = [Path(matrix_folder + fn) for fn in os.listdir(matrix_folder) if "matrix" in fn]
        if len(fps) != 4:
            raise ValueError(
                f"Expected 4 filepaths, got {len(fps)} when looking at filepaths for terms: {model}, {scenario}, {year}"
            )
    
        change_compartments_PM(fps)

def _calculate_costs_year(
    mapping: pd.DataFrame,
    monetization: float | str | dict,
    remove_activities: Optional[pd.DataFrame],
    scenario: str,
    year: int,
    level: str,
    outdir: str,
    model: str,
    save_intermediate_results: bool,
) -> xr.DataArray:

    # load matrices
    matrix_folder = outdir + f"/{model}/{scenario}/{str(year)}/"
    dp, technosphere_inds, biosphere_inds = get_lca_matrices(
        [matrix_folder + fn for fn in os.listdir(matrix_folder) if "matrix" in fn],
        model,
        scenario,
        year,
    )
    print(f"{level}, {year}: Matrices loaded")

    # select functional units
    idx_list = list(mapping.set_index(
        ["dataset name", "dataset reference product", "dataset unit"]
    ).index.unique())
    selected_inds = {k: v for k, v in technosphere_inds.items() if (k[0], k[1], k[2]) in idx_list}
    fus = {str(i): {selected_inds[k]: 1/NCV_DICT[(k[1], k[2])]} for i, k in enumerate(selected_inds.keys())}
    print(f"{level}, {year}: Functional units selected")

    # select activities to remove
    to_remove = []
    if remove_activities is not None:
        remove_idx_list = list(remove_activities.set_index(
            ["dataset name", "dataset reference product", "dataset unit"]
        ).index.unique())
        to_remove = [v for k, v in technosphere_inds.items() if (k[0], k[1], k[2]) in remove_idx_list]

    print(f"{level}, {year}: {len(to_remove)} activities in the removal list")
    
    # set up LCA, LCI calculation
    lca = MultiLCA(
        demands=fus,
        method_config={"impact_categories": []},
        data_objs=[dp,]
    )
    if len(to_remove) > 0:
        lca.load_lci_data()
        new_technosphere_matrix = remove_from_technosphere(lca.technosphere_matrix, to_remove)
        lca.technosphere_matrix = new_technosphere_matrix
        lca.build_demand_array()
        lca.lci_calculation()
    else:
        lca.lci()
    print(f"{level}, {year}: LCI done.")

    # impact and cost calculation
    costs, impacts = get_monetized_results(lca, selected_inds, biosphere_inds, monetization)
    if save_intermediate_results:
        costs.to_csv(Path(matrix_folder) / f"costs_{level}.csv", index=False)
        impacts.to_csv(Path(matrix_folder) / f"impacts_{level}.csv", index=False)

    print(f"{level}, {year}: Cost calculation done.")

    # regionalize costs and combine with shares
    regionalized_costs = regionalize_costs(costs)
    regionalized_costs.to_csv(Path(matrix_folder) / f"regionalized_costs_{level}.csv", index=False)
    regionalized_impacts = regionalize_impacts(impacts)
    regionalized_impacts.to_csv(Path(matrix_folder) / f"regionalized_impacts_{level}.csv", index=False)

    print(f"{level}, {year}: Regionalization done.")


def remove_from_technosphere(
    technosphere_matrix: np.array, activities_to_zero: List[int]
):
    """
    Remove double counting from a technosphere matrix by zeroing out the demanded row values
    in all columns, except for those on the diagonal.
    :param technosphere_matrix: bw2calc.LCA object
    :return: Technosphere matrix with double counting removed
    """

    # Copy and convert the technosphere matrix
    # to COO format for easy manipulation
    technosphere_matrix = technosphere_matrix.tocoo()

    # Create a mask for elements to zero out
    mask = np.isin(technosphere_matrix.row, activities_to_zero) & (
        technosphere_matrix.row != technosphere_matrix.col
    )

    # Apply the mask to set the relevant elements to zero
    technosphere_matrix.data[mask] = 0
    # technosphere_matrix.eliminate_zeros()

    return technosphere_matrix.tocsr()
    