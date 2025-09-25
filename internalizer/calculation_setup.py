import pandas as pd
from typing import List, Optional
from pathlib import Path
from .regionalization import (
    get_regionalized_mapping,
    regionalize_SE_mapping,
    fill_mapping_from_mif,
    get_prodSE,
    get_demFE
)
from .filesystem_constants import DATA_DIR
from .utils import apply_filter_to_dataframe, get_dict_mapping_from_df
import yaml
import xarray as xr

EI_INDEX = ["dataset name", "dataset reference product", "dataset unit"]
TWa2EJ = 31.536

class CalculationSetup:
    def __init__(
        self,
        yaml_path: Path | str
    ):
        setup = yaml.safe_load(open(yaml_path, "r"))
        data = {}
        # read in mapping files
        for lvl, ddict in setup["mappings"].items():
            data[lvl] = {}
            mapping = pd.read_csv(ddict["path"], sep=";")
            data[lvl]["base mapping"] = mapping

            if "domains" in setup.keys():
                data[lvl]["domains"] = setup["domains"].get(lvl, ["index"])
            else:
                data[lvl]["domains"] = ["index"]
                
        self.levels = setup["levels"]
        self.aggregate_levels = setup["aggregate_levels"]
        self.setup = setup
        self.data = data

    def build_removal_lists(
        self,
        all_activities: pd.DataFrame
    ) -> None:
        for lvl in self.levels:
            mapping = self.data[lvl]["base mapping"]
            if lvl not in self.setup["removal_lists"].keys():
                self.data[lvl]["removal list"] = None
            else:
                ddict = self.setup["removal_lists"][lvl]
                dflist = []
                # exclude activities from specified other levels
                for other_lvl in ddict["remove_layers"]:
                    other_mapping = self.data[other_lvl]["base mapping"]
                    dflist.append(other_mapping[EI_INDEX])

                # get extra activities to remove
                dflist.append(apply_filter_to_dataframe(all_activities, ddict["fltr"], ddict["mask"]))

                df = pd.concat(dflist).drop_duplicates()

                # remove activities that are in this level's mapping
                merged = df.merge(mapping, how="outer", indicator=True)
                self.data[lvl]["removal list"] = merged[merged['_merge'] == 'left_only'][EI_INDEX]

    def regionalize_constant_mappings(
        self,
    ) -> None:
        for lvl in self.levels:
            if not self.setup["mappings"][lvl]["depends_on_year"]:
                self.data[lvl]["regionalized mapping"] = get_regionalized_mapping(self.data[lvl]["base mapping"], None)

    def get_production_volume(
        self,
        lvl: str
    ) -> float:
        return 1.0


class RemindInternalizationSetup(CalculationSetup):
    def __init__(
        self,
        yaml_file: Path | str,
        mifpath: Path | str,
        gdxpath: Path | str,
        technosphere_inds: pd.DataFrame
    ):
        super(RemindInternalizationSetup, self).__init__(
            yaml_file
        )
        self.mifpath = mifpath
        self.gdxpath = gdxpath
        self.build_removal_lists(technosphere_inds)
        self.regionalize_constant_mappings()

    def regionalize_constant_mappings(self) -> None:
        self.data["pe2se"]["regionalized mapping"] = regionalize_SE_mapping(self.data["pe2se"]["base mapping"], self.gdxpath)
        self.data["se2h2"]["regionalized mapping"] = regionalize_SE_mapping(self.data["se2h2"]["base mapping"], self.gdxpath)
        self.data["h22se"]["regionalized mapping"] = regionalize_SE_mapping(self.data["h22se"]["base mapping"], self.gdxpath)

    def regionalize_dynamic_mapping(
        self,
        lvl: str,
        year: int,
    ) -> pd.DataFrame:
        if lvl == "fe":
            return fill_mapping_from_mif(self.data[lvl]["base mapping"], self.mifpath, year)
        else:
            return get_regionalized_mapping(self.data[lvl]["base mapping"], None)
        
    def get_production_volumes(
        self,
        lvl: str,
        years: List[int]
    ) -> xr.DataArray | float:
        if lvl == "fe":
            return get_demFE(self.gdxpath, years) * TWa2EJ * 1e12
        elif lvl in ["pe2se", "se2h2", "h22se"]: # xarray automatically uses smallest index
            return get_prodSE(self.gdxpath, years) * TWa2EJ * 1e12
        else:
            return 1.0
        
