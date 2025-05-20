import pandas as pd
from typing import List, Optional
from pathlib import Path
from .regionalization import get_regionalized_mapping

def prepare_setup(
    activities_mappings: List[str | Path],
    gdxpath: str,
    level_names: Optional[List[str]] = None,
    remove_layers: str = "all"
) -> dict:
    setup = {}
    # read in mapping files
    for i, fp in enumerate(activities_mappings):
        mapping = pd.read_csv(fp, sep=";")
        k = fp.split(".")[0] if level_names is None else level_names[i]
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

        setup[k]["removal list"] = pd.concat(dflist).drop(
            columns=["region", "share"]).drop_duplicates()
        
    return setup