from premise import NewDatabase
import bw2data as bd
import bw2calc as bc
from pathways.lca import get_lca_matrices
import xarray as xr
import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import List

from .regionalization import regionalize_costs, combine_shares_and_costs
from .utils import *
from .filesystem_constants import DATA_DIR

FILEPATH_MONETIZATION_FACTORS = DATA_DIR / "mfs_monte_carlo_sample_euro2022.nc"
FILEPATH_MONETIZATION_FACTORS_PERSPECTIVES = DATA_DIR / "mfs_perspectives_euro2022.nc"
NCV_DICT = get_ncv_dict()

def _run_premise_year(
    project: str,
    scen: dict,
    ei_version: str,
    outdir: str
) -> None:
    bd.projects.set_current(project)

    ei_label = "ecoinvent-{}-cutoff".format(ei_version)
    ndb = NewDatabase(
        scenarios=[scen],
        source_db=ei_label,
        source_version=ei_version,
        biosphere_name="ecoinvent-{}-biosphere".format(ei_version)
    )

    ndb.update()

    ndb.write_db_to_matrices(outdir)

def _calculate_costs_year_new(
    mapping: pd.DataFrame,
    cost_perspective: str | float,
    remove_double_counting: bool,
    extra_activities: List,
    scenario: str,
    year: int,
    outdir: str,
    model: str
) -> xr.DataArray:

    # load matrices
    matrix_folder = outdir + f"/{model}/{scenario}/{str(year)}/"
    dp, technosphere_inds, biosphere_inds, _, _ = get_lca_matrices(
        [matrix_folder + fn for fn in os.listdir(matrix_folder) if "matrix" in fn],
        model,
        scenario,
        year
    )

    # select indices
    idx_list = list(mapping.set_index(
        ["dataset name", "dataset reference product", "dataset unit"]
    ).index.unique())
    selected_inds = {k: v for k, v in technosphere_inds.items() if (k[0], k[1], k[2]) in idx_list}
    fus = {str(i): {selected_inds[k]: 1/NCV_DICT[(k[1], k[2])]} for i, k in enumerate(selected_inds.keys())}

    # select activities to remove

    # set up LCA, LCI calculation
    lca = bc.MultiLCA(
        demands=fus,
        method_config={"impact_categories": []},
        data_objs=[dp,]
    )
    if remove_double_counting:
        to_remove = list(selected_inds.values()) + extra_activities
        lca.load_lci_data()
        new_technosphere_matrix = remove_from_technosphere(lca.technosphere_matrix, to_remove)
        lca.technosphere_matrix = new_technosphere_matrix
        lca.build_demand_array()
        lca.lci_calculation()
    else:
        lca.lci()

    # get characterization matrix
    methods = get_lcia_method_names()
    characterization_matrix = fill_characterization_factors_matrices(
            methods=methods,
            biosphere_matrix_dict=lca.dicts.biosphere,
            biosphere_dict=biosphere_inds
        )

    # impact and cost calculation
    if isinstance(cost_perspective, float):
        q = cost_perspective
        mfs = xr.load_dataarray(FILEPATH_MONETIZATION_FACTORS)
        dflist = []
        for k, value in lca.inventories.items():
            impacts = xr.DataArray(
                np.squeeze(
                    np.array((characterization_matrix @ value).sum(axis=-1))
                ),
                {
                    "LCIA method": [m.replace(" - ", ", ") for m in methods]
                }
            )
            costs = (mfs * impacts).sum(dim="LCIA method")
            a = costs.to_numpy()
            qcosts = xr.DataArray(
                np.where(np.all(a < 0, axis=1),
                            np.quantile(a, 1-q, axis=1),
                            np.quantile(a, q, axis=1)),
                {
                    "impact category": list(costs.coords["impact category"].values),
                }
            )
            df = qcosts.to_dataframe(name="cost").reset_index()
            name, refprod, unit, location = list(selected_inds)[int(k)]
            df["dataset name"] = name
            df["dataset reference product"] = refprod
            df["dataset unit"] = unit
            df["dataset location"] = location
            dflist.append(df)
    else:
        mfs = xr.load_dataarray(FILEPATH_MONETIZATION_FACTORS_PERSPECTIVES).sel(
            {"perspective": cost_perspective}
        )
        dflist = []
        for k, value in lca.inventories.items():
            impacts = xr.DataArray(
                np.squeeze(
                    np.array((characterization_matrix @ value).sum(axis=-1))
                ),
                {
                    "LCIA method": [m.replace(" - ", ", ") for m in methods]
                }
            )
            costs = (mfs * impacts).sum(dim="LCIA method")
            df = costs.to_dataframe(name="cost").reset_index()
            name, refprod, unit, location = list(selected_inds)[int(k)]
            df["dataset name"] = name
            df["dataset reference product"] = refprod
            df["dataset unit"] = unit
            df["dataset location"] = location
            dflist.append(df)

    costs = pd.concat(dflist, ignore_index=True)[
        ["dataset name", "dataset reference product", "dataset unit", 
        "dataset location", "impact category", "cost"]
    ]
    costs.to_csv(Path(matrix_folder) / "costs.csv", index=False)

    # regionalize costs and combine with shares
    regionalized_costs = regionalize_costs(costs)
    regionalized_costs.to_csv(Path(matrix_folder) / "regionalized_costs.csv", index=False)
    return combine_shares_and_costs(mapping, regionalized_costs).melt(
        var_name="impact category", value_name="cost", ignore_index=False).reset_index().set_index(
    ["REMIND tech", "region", "quantile", "impact category"])["cost"].to_xarray()

def _calculate_costs_year(
    mapping: pd.DataFrame,
    scenario: str,
    year: int,
    outdir: str,
    model: str,
    quantiles: np.ndarray,
) -> xr.DataArray:

    # load matrices
    matrix_folder = outdir + f"/{model}/{scenario}/{str(year)}/"
    dp, technosphere_inds, biosphere_inds, _, _ = get_lca_matrices(
        [matrix_folder + fn for fn in os.listdir(matrix_folder) if "matrix" in fn],
        model,
        scenario,
        year
    )

    # select indices
    idx_list = list(mapping.set_index(
        ["dataset name", "dataset reference product", "dataset unit"]
    ).index.unique())
    selected_inds = {k: v for k, v in technosphere_inds.items() if (k[0], k[1], k[2]) in idx_list}
    fus = {str(i): {selected_inds[k]: 1/NCV_DICT[(k[1], k[2])]} for i, k in enumerate(selected_inds.keys())}

    # set up LCA, LCI calculation
    lca = bc.MultiLCA(
        demands=fus,
        method_config={"impact_categories": []},
        data_objs=[dp,]
    )
    lca.lci()

    # get characterization matrix
    methods = get_lcia_method_names()
    characterization_matrix = fill_characterization_factors_matrices(
            methods=methods,
            biosphere_matrix_dict=lca.dicts.biosphere,
            biosphere_dict=biosphere_inds
        )

    # impact and cost calculation
    mfs = xr.load_dataarray(FILEPATH_MONETIZATION_FACTORS)
    dflist = []
    for k, value in lca.inventories.items():
        impacts = xr.DataArray(
            np.squeeze(
                np.array((characterization_matrix @ value).sum(axis=-1))
            ),
            {
                "LCIA method": [m.replace(" - ", ", ") for m in methods]
            }
        )
        costs = (mfs * impacts).sum(dim="LCIA method")
        a = costs.to_numpy()
        qcosts = xr.DataArray(
            np.where(np.all(a < 0, axis=1),
                        np.quantile(a, 1-quantiles, axis=1),
                        np.quantile(a, quantiles, axis=1)),
            {
                "quantile": quantiles,
                "impact category": list(costs.coords["impact category"].values),
            }
        )
        df = qcosts.to_dataframe(name="cost").reset_index()
        name, refprod, unit, location = list(selected_inds)[int(k)]
        df["dataset name"] = name
        df["dataset reference product"] = refprod
        df["dataset unit"] = unit
        df["dataset location"] = location
        dflist.append(df)

    costs = pd.concat(dflist, ignore_index=True)[
        ["dataset name", "dataset reference product", "dataset unit", 
        "dataset location", "impact category", "quantile", "cost"]
    ]
    costs.to_csv(Path(matrix_folder) / "costs.csv", index=False)

    # regionalize costs and combine with shares
    regionalized_costs = regionalize_costs(costs)
    regionalized_costs.to_csv(Path(matrix_folder) / "regionalized_costs.csv", index=False)
    return combine_shares_and_costs(mapping, regionalized_costs).melt(
        var_name="impact category", value_name="cost", ignore_index=False).reset_index().set_index(
    ["REMIND tech", "region", "quantile", "impact category"])["cost"].to_xarray()

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
    