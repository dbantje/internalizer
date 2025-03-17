import pandas as pd
import subprocess
from pathlib import Path
import os
import numpy as np
import xarray as xr
import uuid

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

def get_coupled_production_parameters(rundir: str | Path, techs: list) -> pd.DataFrame:
    """
    Get the coupled production parameters from a REMIND `.gdx` file.
    :param rundir: Directory of the REMIND run
    :params tech: List of technologies to filter out
    :return: dataframe containing coupled production parameters.
    """
    if isinstance(rundir, str):
        rundir = Path(rundir)
    gdxpath = Path(rundir) / "fulldata.gdx"
    csvname = f"pm_prodCouple_{uuid.uuid4()}.csv"
    subprocess.run(["gdxdump", gdxpath, "symb=pm_prodCouple", "format=csv", f"output={csvname}"])

    df = pd.read_csv(csvname)
    os.remove(csvname)

    df.rename(columns={"all_regi": "region", "all_te": "REMIND tech"},
              inplace=True)

    return df[df["REMIND tech"].isin(techs)][["region", "REMIND tech", "Val"]]

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

def get_chp_regional_shares(mapping: pd.DataFrame, rundir: str | Path) -> pd.DataFrame:
    """
    Get regional shares for CHP technologies.
    :param mapping: dataframe containing the mapping from REMIND to LCA datasets
    :param rundir: Directory of the REMIND run
    :return: dataframe with regional shares applied
    """
    dflist = []

    # bio CHP
    shares = get_coupled_production_parameters(rundir, ["biochp"]).set_index("region")["Val"]
    sel = mapping[mapping["REMIND tech"].str.endswith("biochp")]
    dflist.append(apply_regional_shares_to_dataframe(
        sel[sel["dataset reference product"].str.contains("electricity")], 1.0
    ))
    dflist.append(apply_regional_shares_to_dataframe(
        sel[sel["dataset reference product"].str.contains("heat")], shares
    ))

    # coal CHP
    shares = get_coupled_production_parameters(rundir, ["coalchp"]).set_index("region")["Val"]
    sel = mapping[mapping["REMIND tech"].str.endswith("coalchp")]
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
    shares = get_coupled_production_parameters(rundir, ["gaschp"]).set_index("region")["Val"]
    sel = mapping[mapping["REMIND tech"].str.endswith("gaschp")]
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
        sel = mapping[mapping["REMIND tech"].str.endswith(fueltype)]
        shares = pd.read_csv(DATA_DIR / f"shares_{fueltype}.csv").set_index("region")
        for feedstock in shares.columns:
            sel2 = sel[sel["dataset name"].str.contains(feedstock)]
            dflist.append(apply_regional_shares_to_dataframe(
                sel2, shares[feedstock]
            ))

    return pd.concat(dflist, axis=0, ignore_index=True).dropna(subset="share")


def get_regionalized_mapping(mapping: pd.DataFrame, rundir: Path | str) -> pd.DataFrame:
    """
    Regionalize mapping.
    :param mapping: the unregionalized mapping
    :param rundir: Directory of the REMIND run
    :return: the regionalized mapping
    """
    # for globally defined shares, simply copy from mapping
    dflist = []
    for region in REMIND_REGIONS:
        df = mapping[mapping["share"] != "regional"].copy()
        df["region"] = region
        dflist.append(df)

    all_shares = pd.concat(
        [
            pd.concat(dflist, axis=0, ignore_index=True),
            get_chp_regional_shares(mapping, rundir),
            get_biofuels_regional_shares(mapping)
        ],
        axis=0,
        ignore_index=True
    )
    all_shares["share"] = all_shares["share"].astype(float)

    return all_shares

def test_share_summation(mapping):
    sums = mapping.groupby(["REMIND tech", "region"]).agg({"share": sum})["share"]
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
            rdf = df[df["dataset location"] == region]
            rdf["region"] = region
            dflist.append(rdf)
        else:
            locations_in_region = [loc for loc in locations if geo.ecoinvent_to_iam_location(loc) == region]
            if len(locations_in_region) > 0:
                rdf = df[df["dataset location"].isin(locations_in_region)]
                rdf["region"] = region
                dflist.append(rdf)  
        

    # create World mix
    global_location = get_fallback_location(locations)
    if global_location is None:
        rdf = df.copy()
        rdf["region"] = "World"
        dflist.append(rdf)
    else:
        rdf = df[df["dataset location"] == global_location]
        rdf["region"] = "World"
        dflist.append(rdf)

    return pd.concat(dflist, axis=0, ignore_index=False)


def regionalize_costs(df: pd.DataFrame) -> pd.DataFrame:
    """
    :param df: dataframe with costs
    :return: regionalized dataframe
    """
    df = df.set_index(["dataset name", "dataset reference product"])
    geo = Geomap(model="remind")

    dflist = []
    for i in df.index.unique():
        sel = df.loc[i]
        dflist.append(select_regional_mixes(sel, geo).reset_index())

    return pd.concat(
        dflist, axis=0, ignore_index=True).groupby(
            ["dataset name", "dataset reference product", 
            "dataset unit", "region", "quantile", "impact category"]
            ).agg({"cost": np.mean}).reset_index()

def combine_shares_and_costs(shares: pd.DataFrame, costs: pd.DataFrame) -> pd.DataFrame:
    """
    Weight costs per shares.
    :param shares: dataframe of regionalized shares
    :param costs: dataframe of regionalized costs
    :return: regionalized aggregated costs per REMIND technology
    """
    shares = shares.set_index(["dataset name", "dataset reference product", "dataset unit", "region"])
    costs = costs.set_index(["dataset name", "dataset reference product", "dataset unit", "region"])

    dflist = []
    costs_index = costs.index
    for idx, row in shares.iterrows():
        tech = row["REMIND tech"]
        factor = row["share"]
        j = idx
        if idx not in costs_index:
            j = (idx[0], idx[1], idx[2], "World")
        try:
            sel = costs.loc[j]
        except KeyError:
            break
        sel["cost"] = factor * sel["cost"]
        sel = sel.pivot(index="quantile", columns="impact category", values="cost").reset_index()
        sel["REMIND tech"] = tech
        sel["region"] = idx[-1]
        dflist.append(sel)
        
    return pd.concat(dflist, axis=0, ignore_index=True).groupby(["REMIND tech", "region", "quantile"]).sum()