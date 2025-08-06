import pandas as pd
from typing import List, Optional
from pathlib import Path
from .regionalization import get_regionalized_mapping, regionalize_SE_mapping, fill_mapping_from_mif
from .filesystem_constants import DATA_DIR
from .utils import apply_filter_to_dataframe
import yaml

prodSE_MAPPING = DATA_DIR / "mappings" / "prodSE.csv"
demFE_mapping = DATA_DIR / "mappings" / "demFE.csv"

DEFAULT_CONFIG = DATA_DIR / "mappings" / "remind_internalization_setup.yaml"
EI_INDEX = ["dataset name", "dataset reference product", "dataset unit"]

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


class RemindInternalizationSetup(CalculationSetup):
    def __init__(
        self,
        mifpath: Path | str,
        gdxpath: Path | str,
        technosphere_inds: pd.DataFrame
    ):
        super(RemindInternalizationSetup, self).__init__(
            DEFAULT_CONFIG
        )
        self.mifpath = mifpath
        self.gdxpath = gdxpath
        self.build_removal_lists(technosphere_inds)
        self.regionalize_constant_mappings()

    def regionalize_constant_mappings(self) -> None:
        self.data["SE"]["regionalized mapping"] = regionalize_SE_mapping(self.data["SE"]["base mapping"], self.gdxpath)

    def regionalize_dynamic_mapping(
        self,
        lvl: str,
        year: int,
    ) -> pd.DataFrame:
        if lvl == "FE":
            return fill_mapping_from_mif(self.data[lvl]["base mapping"], self.mifpath, year)
        else:
            return get_regionalized_mapping(self.data[lvl]["base mapping"], None)
        

def prepare_setup(
    activities_mappings: List[str | Path],
    gdxpath: Optional[str] = None,
    level_names: Optional[List[str]] = None,
    remove_layers: str = "all"
) -> dict:
    setup = {}
    # read in mapping files
    for i, fp in enumerate(activities_mappings):
        mapping = pd.read_csv(fp, sep=";")
        k = fp.split("/")[-1].split(".")[0] if level_names is None else level_names[i]
        setup[k] = {}
        setup[k]["mapping"] = get_regionalized_mapping(mapping, gdxpath)

    # build removal lists
    for k in setup.keys():
        dflist = []
        for j in setup.keys():
            if k != j:
                dflist.append(setup[j]["mapping"])
            elif remove_layers == "all":
                dflist.append(setup[j]["mapping"])

        if len(dflist) > 0:
            setup[k]["removal list"] = pd.concat(dflist).drop(
                columns=["region", "share"]).drop_duplicates()
        else:
            setup[k]["removal list"] = None
        
    return setup

def default_setup(gdxpath, remove_layers=None):
    mappings = [
        prodSE_MAPPING,
    ]
    levels = [
        "SE",
    ]

    return prepare_setup(
        mappings, gdxpath, level_names=levels, remove_layers=remove_layers
    )

def default_setup_new(gdxpath, remove_layers=None):
    mappings = [
        prodSE_MAPPING,
        demFE_mapping,
    ]
    levels = [
        "SE",
        "FE"
    ]

    return prepare_setup(
        mappings, gdxpath, level_names=levels, remove_layers=remove_layers
    )
