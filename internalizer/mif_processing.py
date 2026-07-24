import pandas as pd
from .allocate_bio_liquids import process_mif_file
import xarray as xr

def add_ES_subcategories(mifpath, newpath):
    """
    Add subcategories for ES transport variables that are needed for premise.
    """
    df = pd.read_csv(mifpath, sep=";")
    if len(df.columns) == 25:
        df = df.iloc[:, :-1]
    if len(df.columns) != 24:
        raise ValueError("Unexpected number of columns in the MIF file.")

    all_variables = list(df["Variable"].unique())
    ES_variables = [v for v in all_variables if v.startswith("ES|Transport")]
    parent_variables = [v for v in ES_variables if v.split("|")[-1] in ["Liquids", "Gases"]]
    dflist = []
    for parent in parent_variables:
        prefix = parent.replace("ES|", "FE|")
        children = [v for v in all_variables if v.startswith(prefix + "|")]
        if len(children) == 3:
            subcats = [v.split("|")[-1] for v in children]
            subcats2add = [cat for cat in subcats if "|".join([parent, cat]) not in all_variables]

            if len(subcats2add) > 0:
                years = list(df.columns)[5:]

                total = df[df["Variable"] == prefix].set_index("Region")[years].astype(float)
                parent_df = df[df["Variable"] == parent]
                new_unit = list(parent_df["Unit"].unique())[0]
                parent_df = parent_df.set_index("Region")[years].astype(float)

                for cat in subcats2add:
                    s = prefix + "|" + cat
                    sel = df[df["Variable"] == s].copy().set_index("Region")[years].astype(float)
                    share = sel.div(total, axis=0)
                    new_df = share.mul(parent_df, axis=0).reset_index()
                    new_df["Model"] = list(df["Model"].unique())[0]
                    new_df["Scenario"] = list(df["Scenario"].unique())[0]
                    new_df["Unit"] = new_unit
                    new_df["Variable"] = "|".join([parent, cat])
                    dflist.append(new_df)

    if len(dflist) > 0:
        additions = pd.concat(dflist, axis=0)[["Model", "Scenario", "Region", "Variable", "Unit"]+years]
        pd.concat((df, additions), axis=0).to_csv(newpath, sep=";", index=False)


def add_woNonEnergy_vars(mifpath, newpath):
    data = pd.read_csv(mifpath, sep=";")
    years = list(data.columns)[5:]
    candidates = [
        v for v in data["Variable"].unique()
        if v.startswith("FE|Non-energy Use|Industry|")
        and v.split("|")[3] in ["Liquids", "Gases", "Solids"]
    ]

    dflist = []
    for v in candidates:
        feedstock = "|".join(v.split("|")[3:])
        chemvar = "FE|Industry|Chemicals|" + feedstock
        newvar = "FE|w/o Non-energy Use|Industry|Chemicals|" + feedstock

        A = data.set_index(["Variable", "Region"]).loc[chemvar]
        B = data.set_index(["Variable", "Region"]).loc[v]
        df = (A[years] - B[years]).reset_index()
        df["Model"] = A["Model"].unique()[0]
        df["Unit"] = "EJ/yr"
        df["Scenario"] = A["Scenario"].unique()[0]
        df["Variable"] = newvar
        dflist.append(df[["Model", "Scenario", "Region", "Variable", "Unit"]+years])

    return pd.concat([data]+dflist).to_csv(newpath, sep=";", index=False)


def add_lignosplit(mifpath, newpath):
    parents = [
        "SE|Liquids|Biomass|Lignocellulosic Ethanol",
        "SE|Liquids|Biomass|BioFTR|w/o CC",
        "SE|Liquids|Biomass|BioFTR|w/ CC",
        "SE|Liquids|Biomass|BioFTR|w/ pyrolysis",
    ]

    feedstocks = [
        "PE|Biomass|+++|Residues",
        "PE|Biomass|+++|Energy Crops",
    ]

    data = pd.read_csv(mifpath, sep=";")
    years = list(data.columns)[5:]

    unit = "EJ/yr"
    model = "REMIND"
    scenario = data["Scenario"].unique()[0]

    dflist = []
    total = data[data["Variable"].isin(feedstocks)].groupby("Region")[years].sum()
    for fs in feedstocks:
        # calculate feedstock shares
        fs_shares = data[data["Variable"] == fs].groupby("Region")[years].sum() / total
        for v in parents:
            sel = data[data["Variable"] == v].copy()
            parentdata = sel.set_index("Region")[years]
            allocated = parentdata * fs_shares
            allocated["Variable"] = v + "|+|" + fs.split("|")[-1]
            allocated["Model"] = model
            allocated["Scenario"] = scenario
            allocated["Unit"] = unit
            allocated = allocated.reset_index()
            dflist.append(allocated[["Model", "Scenario", "Variable", "Region", "Unit"] + years])

    df2add = pd.concat(dflist, ignore_index=True)

    pd.concat([data, df2add]).to_csv(newpath, sep=";", index=False)

    
def get_ratios(mif, var1, var2, years, s1=1.0, s2=1.0):
    sel = mif[["Variable", "Region"]+[str(y) for y in years]]

    df1 = sel[sel["Variable"] == var1].set_index(["Region"]).drop(columns="Variable").astype(float)
    df2 = sel[sel["Variable"] == var2].set_index(["Region"]).drop(columns="Variable").astype(float)

    return (s1 * df1) / (s2 * df2)

def select_var(mif, var, years):
    sel = mif[["Variable", "Region"]+[str(y) for y in years]]
    df1 = sel[sel["Variable"] == var].set_index(["Region"]).drop(columns="Variable").astype(float)

    return df1


def get_capacity_ratios(mif, years):
    newcap_storage = [
        "New Cap|Electricity|Storage|Battery|For PV",
        "New Cap|Electricity|Storage|Battery|For Wind",
    ]
    newcap_vre = [
        "New Cap|Electricity|Solar|+|PV",
        "New Cap|Electricity|+|Wind",
    ]
    techs = ["PV", "Wind"]

    dflist = []
    for tech, v1, v2 in zip(techs, newcap_storage, newcap_vre):
        df = get_ratios(mif, v1, v2, years).melt(ignore_index=False, var_name="Year")
        df["technology"] = tech
        dflist.append(df.reset_index())

    capratiodata = pd.concat(dflist)
    return capratiodata.set_index(
        ["technology", "Region", "Year"])["value"].to_xarray()


def get_capacity_factors(mif, years):
    production_vars = [
        "SE|Electricity|Solar|+|PV",
        "SE|Electricity|+|Wind",
    ]

    cap_vars = [
        "Cap|Electricity|Solar|+|PV",
        "Cap|Electricity|+|Wind",
    ]

    # scaling factors
    scap = 1e-03 # from giga to tera-watt
    sprod = 31.71e-03 # EJ to TWa
    techs = ["PV", "Wind"]

    dflist = []
    for tech, v1, v2 in zip(techs, production_vars, cap_vars):
        df = get_ratios(mif, v1, v2, years, s1=sprod, s2=scap).melt(ignore_index=False, var_name="Year")
        df["technology"] = tech
        dflist.append(df.reset_index())

    capfacdata = pd.concat(dflist)
    return capfacdata.set_index(
        ["technology", "Region", "Year"])["value"].to_xarray()


def get_SEprod_data(mif, years):
    varnames = [
        "SE|Electricity|Solar|+|PV",
        "SE|Electricity|+|Wind",
    ]

    techs = ["PV", "Wind"]

    dflist = []
    for tech, var in zip(techs, varnames):
        df = select_var(mif, var, years).melt(ignore_index=False, var_name="Year")
        df["technology"] = tech
        dflist.append(df.reset_index())

    SEproddata = pd.concat(dflist)
    return SEproddata.set_index(
        ["technology", "Region", "Year"])["value"].to_xarray()


def add_stationary_battery_requirements(mifpath, newpath, energy2power=5,
                                        lifetime_pv=30, lifetime_wind=25):
    data = pd.read_csv(mifpath, sep=";")
    years = list(data.columns)[5:]

    techs = ["PV", "Wind"]
    lifetimes_xr = xr.DataArray(
        [lifetime_pv, lifetime_wind],
        coords={
            "technology": techs
        }
    )
    capfacs_xr = get_capacity_factors(data, years)
    capratios_xr = get_capacity_ratios(data, years)
    SEprod_xr = get_SEprod_data(data, years)

    # calculations
    installation_fraction_xr = 1 / (capfacs_xr * 8760 * lifetimes_xr)

    battery_requirements = (SEprod_xr * capratios_xr
                            * energy2power * installation_fraction_xr)
    df2add = battery_requirements.to_dataframe(name="value").reset_index()
    df2add["Model"] = data["Model"].unique()[0]
    df2add["Scenario"] = data["Scenario"].unique()[0]
    df2add["Unit"] = "EJ/yr"
    df2add["Variable"] = df2add["technology"].apply(lambda x: "SE|Battery Power Capacity|For " + x)
    df2add = df2add.pivot(
        index=["Model", "Scenario", "Region", "Variable", "Unit"],
        columns="Year", values="value"
    ).reset_index()
    
    pd.concat((data, df2add)).to_csv(newpath, sep=";", index=False)


def remove_regions(mifpath, newpath, regions_to_remove):
    data = pd.read_csv(mifpath, sep=";")
    data[~data["Region"].isin(regions_to_remove)].to_csv(newpath, sep=";", index=False)


def process_mif(mifpath, model="remind"):
    add_ES_subcategories(mifpath, mifpath)
    print("\tAdded ES subcategories.")
    add_woNonEnergy_vars(mifpath, mifpath)
    print("\tAdded w/o Non-energy Use variables.")
    add_stationary_battery_requirements(mifpath, mifpath)
    print("\tAdded stationary battery requirements.")
    if model == "remind-eu":
        remove_regions(mifpath, mifpath, ["EU27", "EUR", "NEU"])
        print("\tRemoved EU27, EUR and NEU regions.")
    process_mif_file(mifpath, suffix="")
    print("\tAllocated bio feedstocks.")

