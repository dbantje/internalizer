from premise import NewDatabase
import bw2data as bd
from bw2calc import MultiLCA
from pathways.lca import get_lca_matrices
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
    correct_coke_production_flows
)
from .filesystem_constants import DATA_DIR

FILEPATH_MONETIZATION_FACTORS = DATA_DIR / "mfs_MCsample_EUR2022.nc"
FILEPATH_MONETIZATION_FACTORS_PERSPECTIVES = DATA_DIR / "mfs_perspectives_EUR2022.nc"
NCV_DICT = get_ncv_dict()

def get_cfs_and_mfs(
    monetization: float | str | dict,
    lca: MultiLCA,
    biosphere_inds: dict
) -> Tuple[csr_matrix, xr.DataArray, list]:
    methods = get_lcia_method_names()
    if isinstance(monetization, float):
        cfs = fill_characterization_factors_matrices(
            methods=methods,
            biosphere_matrix_dict=lca.dicts.biosphere,
            biosphere_dict=biosphere_inds
        )
        mfs = xr.load_dataarray(FILEPATH_MONETIZATION_FACTORS)
        return cfs, mfs, methods 
    elif isinstance(monetization, str):
        cfs = fill_characterization_factors_matrices(
            methods=methods,
            biosphere_matrix_dict=lca.dicts.biosphere,
            biosphere_dict=biosphere_inds
        )
        mfs = xr.load_dataarray(FILEPATH_MONETIZATION_FACTORS_PERSPECTIVES).sel(
            {"perspective": monetization}
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
    outdir: str,
    quiet: bool
) -> None:
    bd.projects.set_current(project)

    ei_label = "ecoinvent-{}-cutoff".format(ei_version)
    ndb = NewDatabase(
        scenarios=[scen],
        source_db=ei_label,
        source_version=ei_version,
        biosphere_name="ecoinvent-{}-biosphere".format(ei_version),
        quiet=quiet
    )

    ndb.update()

    ndb.write_db_to_matrices(outdir)

    if ei_version == "3.10":
        model = scen["model"]
        scenario = scen["pathway"]
        year = scen["year"]
        mfolder = outdir + f"/{model}/{scenario}/{str(year)}/"
        correct_coke_production_flows(mfolder)

def _calculate_costs_year(
    mapping: pd.DataFrame,
    monetization: float | str | dict,
    remove_activities: Optional[pd.DataFrame],
    scenario: str,
    year: int,
    level: str,
    outdir: str,
    model: str,
    save_intermediate_results: bool
) -> xr.DataArray:

    # load matrices
    matrix_folder = outdir + f"/{model}/{scenario}/{str(year)}/"
    dp, technosphere_inds, biosphere_inds, _, _ = get_lca_matrices(
        [matrix_folder + fn for fn in os.listdir(matrix_folder) if "matrix" in fn],
        model,
        scenario,
        year
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
    