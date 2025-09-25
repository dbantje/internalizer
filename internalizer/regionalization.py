import pandas as pd
import subprocess
from pathlib import Path
import os
import numpy as np
import uuid
from typing import Optional, List
import xarray as xr

from premise.geomap import Geomap

from .filesystem_constants import DATA_DIR

REMIND_REGIONS = [
    "CAZ",
    "CHA",
    "EUR",
    "IND",
    "JPN",
    "LAM",
    "MEA",
    "NEU",
    "OAS",
    "REF",
    "SSA",
    "USA"
]

FILEPATH_COALTYPE_SHARES = DATA_DIR / "shares_coal.csv"

def get_coupled_production_parameters(gdxpath: str, techs: list) -> pd.DataFrame:
    """
    Get the coupled production parameters from a REMIND `.gdx` file.
    :param rundir: Filepath to a REMIND .gdx
    :params tech: List of technologies to filter out
    :return: dataframe containing coupled production parameters.
    """
    csvname = f"pm_prodCouple_{uuid.uuid4()}.csv"
    subprocess.run(["gdxdump", gdxpath, "symb=pm_prodCouple", "format=csv", f"output={csvname}"])

    df = pd.read_csv(csvname)
    os.remove(csvname)

    df.rename(columns={"all_regi": "region", "all_te": "REMIND index"},
              inplace=True)

    return df[df["REMIND index"].isin(techs)][["region", "REMIND index", "Val"]]

def get_prodSE(gdxpath: str, years: List[int]) -> xr.DataArray:
    csvname = f"vm_prodSE_{uuid.uuid4()}.csv"
    subprocess.run(["gdxdump", gdxpath, "symb=vm_prodSE", "format=csv", f"output={csvname}"])

    df = pd.read_csv(csvname)
    os.remove(csvname)
    mapper = {"all_te": "REMIND index", "Val": "prodSE", "tall": "year", "all_regi": "region"}
    df = df.rename(columns=mapper)
    df = df.loc[df["year"].isin(years)]

    return df.set_index(["REMIND index", "year", "region"])["prodSE"].to_xarray()

def get_demFE(gdxpath: str, years: List[int]) -> xr.DataArray:
    csvname = f"vm_demFeSector_{uuid.uuid4()}.csv"
    subprocess.run(["gdxdump", gdxpath, "symb=vm_demFeSector", "format=csv", f"output={csvname}"])

    df = pd.read_csv(csvname)
    os.remove(csvname)

    df["emi_sectors"] = df["emi_sectors"].str.lower()
    df["REMIND index"] = [" - ".join([sec, fuel]) for sec, fuel in zip(df["emi_sectors"], df["all_enty.1"])]
    mapper = {"Val": "demFE", "ttot": "year", "all_regi": "region"}
    df = df.rename(columns=mapper)
    df = df.loc[df["year"].isin(years)]

    return df.groupby(["REMIND index", "year", "region"])["demFE"].sum().to_xarray()

def apply_regional_shares_to_dataframe(df: pd.DataFrame, shares: pd.Series | float, factors: pd.Series | float = 1.0) -> pd.DataFrame:
    """
    :param df: base dataframe
    :param shares: Series of shares, indexed by region
    :param factors: constant additional factor or series of factors to apply
    :return: dataframe including regional shares
    """
    # combine shares and factors
    if isinstance(factors, float):
        factors = pd.Series(factors * np.ones(len(REMIND_REGIONS)),
                            index=pd.Index(REMIND_REGIONS, name="region"))
    combined_shares = shares * factors

    # prepare dataframe, add regionalized shares
    df = df.loc[df.index.repeat(len(REMIND_REGIONS))]
    df["region"] = REMIND_REGIONS
    df.set_index("region", inplace=True)
    df["share"] = combined_shares

    return df.reset_index()

def get_chp_regional_shares(mapping: pd.DataFrame, gdxpath: str | Path) -> pd.DataFrame:
    """
    Get regional shares for CHP technologies.
    :param mapping: dataframe containing the mapping from REMIND to LCA datasets
    :param gdxpath: path to the REMIND gdx
    :return: dataframe with regional shares applied
    """
    dflist = []

    # bio CHP
    shares = get_coupled_production_parameters(gdxpath, ["biochp"]).set_index("region")["Val"]
    sel = mapping[mapping["REMIND index"].str.endswith("biochp")]
    dflist.append(apply_regional_shares_to_dataframe(
        sel[sel["dataset reference product"].str.contains("electricity")], 1.0
    ))
    dflist.append(apply_regional_shares_to_dataframe(
        sel[sel["dataset reference product"].str.contains("heat")], shares
    ))

    # coal CHP
    shares = get_coupled_production_parameters(gdxpath, ["coalchp"]).set_index("region")["Val"]
    sel = mapping[mapping["REMIND index"].str.endswith("coalchp")]
    coal_type_shares = pd.read_csv(FILEPATH_COALTYPE_SHARES).set_index("region")
    sel2 = sel[sel["dataset name"].str.contains("lignite")]
    dflist.append(apply_regional_shares_to_dataframe(
        sel2[sel2["dataset reference product"].str.contains("electricity")], 1.0, factors=coal_type_shares["lignite"]
    ))
    dflist.append(apply_regional_shares_to_dataframe(
        sel2[sel2["dataset reference product"].str.contains("heat")], shares, factors=coal_type_shares["lignite"]
    ))
    sel2 = sel[sel["dataset name"].str.contains("hard coal")]
    dflist.append(apply_regional_shares_to_dataframe(
        sel2[sel2["dataset reference product"].str.contains("electricity")], 1.0, factors=coal_type_shares["hard coal"]
    ))
    dflist.append(apply_regional_shares_to_dataframe(
        sel2[sel2["dataset reference product"].str.contains("heat")], shares, factors=coal_type_shares["hard coal"]
    ))


    # gas CHP
    shares = get_coupled_production_parameters(gdxpath, ["gaschp"]).set_index("region")["Val"]
    sel = mapping[mapping["REMIND index"].str.endswith("gaschp")]
    sel2 = sel[sel["dataset name"].str.contains("combined cycle")]
    dflist.append(apply_regional_shares_to_dataframe(
        sel2[sel2["dataset reference product"].str.contains("electricity")], 1.0, factors=0.1
    ))
    dflist.append(apply_regional_shares_to_dataframe(
        sel2[sel2["dataset reference product"].str.contains("heat")], shares, factors=0.1
    ))
    sel2 = sel[sel["dataset name"].str.contains("conventional")]
    dflist.append(apply_regional_shares_to_dataframe(
        sel2[sel2["dataset reference product"].str.contains("electricity")], 1.0, factors=0.9
    ))
    dflist.append(apply_regional_shares_to_dataframe(
        sel2[sel2["dataset reference product"].str.contains("heat")], shares, factors=0.9
    ))

    return pd.concat(dflist, axis=0, ignore_index=True)

def get_biofuels_regional_shares(mapping: pd.DataFrame) -> pd.DataFrame:
    """
    Apply regional shares from 
    :param mapping: dataframe containing the mapping from REMIND to LCA datasets
    :return: dataframe with regional shares applied
    """
    dflist = []

    for fueltype in ["bioeths", "biodiesel", "bioethl"]:
        sel = mapping[mapping["REMIND index"].str.endswith(fueltype)]
        shares = pd.read_csv(DATA_DIR / f"shares_{fueltype}.csv").set_index("region")
        for feedstock in shares.columns:
            sel2 = sel[sel["dataset name"].str.contains(feedstock)]
            dflist.append(apply_regional_shares_to_dataframe(
                sel2, shares[feedstock]
            ))

    return pd.concat(dflist, axis=0, ignore_index=True).dropna(subset="share")

def fill_mapping_from_mif(
    mapping: pd.DataFrame,
    mifpath: str | Path,
    year: int
) -> pd.DataFrame:
    # extend mapping
    dflist = []
    for region in REMIND_REGIONS:
        df = mapping.copy()
        df["region"] = region
        dflist.append(df)

    regionalized_mapping = pd.concat(dflist).set_index(["scenario variable", "region"])

    # load .mif
    mif = pd.read_csv(mifpath, sep=";").rename(columns={"Region": "region", "Variable": "scenario variable"})
    mifdata = mif.set_index(["scenario variable", "region"])[str(year)]

    # select shared index (some regions don't have certain FE variables)
    combined_idx = regionalized_mapping.index.intersection(mifdata.index)
    sel = regionalized_mapping.loc[combined_idx]

    # calculate shares
    sel["weight"] = mifdata.loc[combined_idx]
    sel = sel.reset_index()
    sel["total"] = sel.reset_index().groupby(["REMIND index", "region"])["weight"].transform("sum")
    sel["share"] = sel["weight"] / sel["total"]

    return sel[["REMIND index", "dataset name", 
                "dataset reference product", "dataset unit", "share", "region"]]

def get_mif_variables(
    mapping: pd.DataFrame,
    mifpath: str | Path,
    years: List[int]
) -> pd.DataFrame:
    # load .mif and select years
    mif = pd.read_csv(mifpath, sep=";").rename(columns={"Region": "region", "Variable": "scenario variable"})
    years = [str(y) for y in years]
    cols = [c for c in mif.columns if c in years]
    mifdata = mif.set_index(["scenario variable", "region"])[cols]

    # select needed variables
    all_vars = list(mapping["scenario variable"].unique())
    sel = mifdata.loc[pd.IndexSlice[all_vars, REMIND_REGIONS]].copy().reset_index()

    return sel.melt(
        id_vars=["scenario variable", "region"],
        var_name="year",
    )

def regionalize_SE_mapping(
    mapping: pd.DataFrame,
    gdxpath: Path | str
) -> pd.DataFrame:
    # for globally defined shares, simply copy from mapping
    dflist = []
    for region in REMIND_REGIONS:
        df = mapping[mapping["share"] != "regional"].copy()
        df["region"] = region
        dflist.append(df)

    # get all regional shares that are available
    all_regional_shares = pd.concat(
        [
            get_chp_regional_shares(mapping, gdxpath),
            get_biofuels_regional_shares(mapping)
        ],
        axis=0,
        ignore_index=True
    )

    # select only those that are in the mapping
    needed_techs = list(mapping[mapping["share"] == "regional"]["REMIND index"].unique())
    needed_regional_shares = all_regional_shares[all_regional_shares["REMIND index"].isin(needed_techs)]

    all_shares = pd.concat(
        [
            pd.concat(dflist, axis=0, ignore_index=True),
            needed_regional_shares
        ],
        axis=0,
        ignore_index=True
    )

    all_shares["share"] = all_shares["share"].astype(float)

    return all_shares


def get_regionalized_mapping(
    mapping: pd.DataFrame,
    gdxpath: Optional[str],
) -> pd.DataFrame:
    """
    Regionalize mapping.
    :param mapping: the unregionalized mapping
    :param rundir: Filepath of the REMIND .gdx file
    :return: the regionalized mapping
    """
    # for globally defined shares, simply copy from mapping
    dflist = []
    for region in REMIND_REGIONS:
        df = mapping[mapping["share"] != "regional"].copy()
        df["region"] = region
        dflist.append(df)
    
    if len(mapping[mapping["share"] == "regional"] == 0) or gdxpath is None:
        all_shares = pd.concat(dflist, axis=0, ignore_index=True)
    else:
        # get all regional shares that are available
        all_regional_shares = pd.concat(
            [
                get_chp_regional_shares(mapping, gdxpath),
                get_biofuels_regional_shares(mapping)
            ],
            axis=0,
            ignore_index=True
        )

        # select only those that are in the mapping
        needed_techs = list(mapping[mapping["share"] == "regional"]["REMIND index"].unique())
        needed_regional_shares = all_regional_shares[all_regional_shares["REMIND index"].isin(needed_techs)]

        all_shares = pd.concat(
            [
                pd.concat(dflist, axis=0, ignore_index=True),
                needed_regional_shares
            ],
            axis=0,
            ignore_index=True
        )
        
    all_shares["share"] = all_shares["share"].astype(float)

    return all_shares

def test_share_summation(mapping):
    sums = mapping.groupby(["REMIND index", "region"]).agg({"share": sum})["share"]
    if not all(sums == 1):
        print("Some shares don't sum to 1!")
        return sums[sums != 1]
    else:
        print("All shares sum to 1!")
        return sums
    
def get_fallback_location(locations: list) -> str:
    """
    Get a fallback location from an array of locations.
    """
    if "World" in locations:
        return "World"
    elif "RoW" in locations:
        return "RoW"
    elif "GLO" in locations:
        return "GLO"
    else:
        return None

def select_regional_mixes(df: pd.DataFrame, geo: Geomap) -> pd.DataFrame:
    """
    Select regional mixes from dataframe. If present as dataset location, the IAM region
    is chosen. Else, all locations within the IAM region are chosen. Additionally, a 'World'
    region mix is created.
    :param df: dataframe, containing column 'dataset location'
    :param geo: a Geomap instance
    :return: dataframe with regional mixes for all REMIND regions.
    """
    locations = list(df["dataset location"].unique())

    dflist = []
    for region in REMIND_REGIONS:
        if region in locations:
            rdf = df.loc[df["dataset location"] == region].copy()
            rdf["region"] = region
            dflist.append(rdf)
        else:
            locations_in_region = [loc for loc in locations if geo.ecoinvent_to_iam_location(loc) == region]
            if len(locations_in_region) > 0:
                rdf = df.loc[df["dataset location"].isin(locations_in_region)].copy()
                rdf["region"] = region
                dflist.append(rdf)  
        

    # create World mix
    global_location = get_fallback_location(locations)
    if global_location is None:
        rdf = df.copy()
        rdf["region"] = "World"
        dflist.append(rdf)
    else:
        rdf = df.loc[df["dataset location"] == global_location].copy()
        rdf["region"] = "World"
        dflist.append(rdf)

    return pd.concat(dflist, axis=0, ignore_index=False)


def regionalize_costs(df: pd.DataFrame) -> pd.DataFrame:
    """
    :param df: dataframe with costs
    :return: regionalized dataframe
    """
    df = df.set_index(["dataset name", "dataset reference product"]).sort_index()
    geo = Geomap(model="remind")

    dflist = []
    for i in df.index.unique():
        sel = df.loc[i].copy()
        dflist.append(select_regional_mixes(sel, geo).reset_index())

    return pd.concat(
        dflist, axis=0, ignore_index=True).groupby(
            ["dataset name", "dataset reference product", 
            "dataset unit", "region", "impact category"]
            ).agg({"cost": "mean"}).reset_index()

def regionalize_impacts(df: pd.DataFrame) -> pd.DataFrame:
    """
    :param df: dataframe with impacts
    :return: regionalized dataframe
    """
    df = df.set_index(["dataset name", "dataset reference product"]).sort_index()
    geo = Geomap(model="remind")

    dflist = []
    for i in df.index.unique():
        sel = df.loc[i].copy()
        dflist.append(select_regional_mixes(sel, geo).reset_index())

    return pd.concat(
        dflist, axis=0, ignore_index=True).groupby(
            ["dataset name", "dataset reference product", 
            "dataset unit", "region", "LCIA method"]
            ).agg({"impact": "mean"}).reset_index()

def aggregate_with_mapping(
    mapping: pd.DataFrame,
    df: pd.DataFrame,
    var_col: str = "impact category",
    value_col: str = "cost"
) -> pd.DataFrame:
    """
    Aggregate dataframe with mapping.
    :param mapping: dataframe of regionalized mapping containing shares
    :param costs: regionalized dataframe
    :return: regionalized aggregated data
    """
    mapping = mapping.set_index(["dataset name", "dataset reference product", "dataset unit", "region"])
    df = df.pivot(
        index=["dataset name", "dataset reference product", "dataset unit", "region"],
        columns=var_col,
        values=value_col
        )
    ics = df.columns

    # build index with fallback to 'World' region
    old_idx = mapping.index
    new_idx = []
    for idx in old_idx:
        if idx in df.index:
            new_idx.append(idx)
        else:
            new_idx.append((idx[0], idx[1], idx[2], "World"))

    # combine and multiply by shares
    combined = pd.DataFrame(
        df.loc[new_idx].to_numpy(),
        index=old_idx,
        columns=ics
        ).mul(mapping["share"], axis=0)
    combined["REMIND index"] = mapping["REMIND index"]
    combined = combined.groupby(["REMIND index", "region"])[ics].sum()

    return combined.melt(
        var_name=var_col, value_name=value_col, ignore_index=False
    ).reset_index()