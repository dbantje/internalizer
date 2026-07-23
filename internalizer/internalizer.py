from typing import List, Optional
from pathlib import Path
import os

from multiprocessing import Pool, cpu_count
import shutil
import pandas as pd
import bw2data as bd

from premise import NewDatabase, clear_cache
from .plca import _run_premise_year, _calculate_costs_year, DEFAULT_PREMISE_KWARGS
from .utils import (
    convert_euros_to_dollar,
    check_monetization_factors,
    get_linear_ramp_up,
    interpolate_and_weight_xr,
    split_remind_index,
    get_source_version,
)
from .calculation_setup import (
    CalculationSetup,
    RemindInternalizationSetup,
    EI_INDEX
)
from .regionalization import aggregate_with_mapping, regionalize_impacts
from .filesystem_constants import DATA_DIR

EURO_REF_YEAR = 2022
REMIND_USD_REF_YEAR = 2017
COST_PERSPECTIVES = [
    "damage costs",
    "prevention costs",
    "budget constraint",
    "taxation costs"
]

DEFAULT_CONFIG = DATA_DIR / "mappings" / "remind_internalization_setup_v2.yaml"
CONFIG_NO_REMOVAL = DATA_DIR / "mappings" / "remind_internalization_setup_noRemoval.yaml"
CONFIG_PE2SE_ONLY = DATA_DIR / "mappings" / "remind_internalization_setup_pe2seonly.yaml"

MODEL_YEARS = {
    "remind": [2005, 2010, 2015, 2020, 2025, 2030, 2035, 2040, 2045,
               2050, 2055, 2060, 2070, 2080, 2090, 2100, 2110, 2130,
               2150]
}

def extract_folder_and_filename(fp):
    fname = (fp.split("/")[-1]).split(".")[0]
    folder = "/".join(fp.split("/")[:-1])

    return folder, fname

def extract_output_folder(fp):
    return fp.split("/")[-2]

class Internalizer:
    """
    The Internalizer class
    """

    def __init__(
        self,
        mifpath: str,
        model: str,
        pathway: str,
        ei_version: str,
        bw_project: str,
        gdxpath: str,
        outputfolder: str = "output",
        single_run: bool = True,
        max_mp_tasks: int | None = 4
    ):
        # get directory of data file and scenario name
        self.model = model
        rundir, filename = extract_folder_and_filename(mifpath)
        self.rundir = rundir
        outputdir = f"./{outputfolder}"
        if not single_run:
            outputdir += "/" + extract_output_folder(mifpath)
        self.outdir = outputdir
        namecheck = "_".join((model.lower(), pathway))
        if filename != namecheck:
            shutil.copy(mifpath, rundir+f"/{namecheck}.mif")
        self.scenario = pathway

        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)

        self.ei_version = ei_version
        self.source_version = get_source_version(ei_version) 
        self.bw_project = bw_project
        self.gdxpath = gdxpath
        self.mifpath = mifpath
        if max_mp_tasks is not None:
            self.max_mp_tasks = max_mp_tasks
        else:
            self.max_mp_tasks = cpu_count()

    def recreate_premise_cache(
        self,
    ) -> None:
        # set brightway project
        bd.projects.set_current(self.bw_project)
        
        # clear premise cache
        clear_cache()

        # newdatabase initialization creates cache
        ei_label = "ecoinvent-{}-cutoff".format(self.ei_version)
        scen = {"model": self.model, "pathway": self.scenario, "year": 2020, "filepath": self.rundir}
        kwargs = {
            "scenarios": [scen],
            "source_db": ei_label,
            "source_version": self.source_version,
            "biosphere_name": "ecoinvent-{}-biosphere".format(self.ei_version),
        }
        kwargs.update(DEFAULT_PREMISE_KWARGS)
        ndb = NewDatabase(**kwargs)

    def run_premise(
        self,
        years: List[int],
        multiprocessing: bool = True,
        quiet: bool = True,
        include_interventions: bool = True,
    ) -> None:
        self.years = years
        
        args = [
            (
                self.bw_project,
                {"model": self.model, "pathway": self.scenario, "year": year, "filepath": self.rundir},
                self.ei_version,
                self.self.source_version,
                self.outdir,
                quiet,
                include_interventions
            )
            for year in self.years
        ]

        if multiprocessing:
            with Pool(self.max_mp_tasks) as p:
            #with Pool(cpu_count(), maxtasksperchild=1000) as p:
                p.starmap(_run_premise_year, args)
        else:
            for arg in args:
                _run_premise_year(*arg)

    def get_technosphere_df(self):
        dflist = []
        for year in self.years:
            matrix_folder = self.outdir + f"/{self.model}/{self.scenario}/{str(year)}/"
            df = pd.read_csv(
                matrix_folder + "/A_matrix_index.csv", sep=";",
                usecols=[0, 1, 2],
                names=EI_INDEX
            )
            dflist.append(df)
        
        return pd.concat(dflist, ignore_index=True).drop_duplicates()

    def set_calculation_setup(
        self,
        setup: str = "default",
        yaml_file: Path | str = DEFAULT_CONFIG,
        levels = "SE,FE",
    ) -> None:
        if setup == "default":
            self.cs = RemindInternalizationSetup(yaml_file, levels, self.mifpath, self.gdxpath, self.get_technosphere_df())
        elif setup == "pe2se":
            self.cs = RemindInternalizationSetup(CONFIG_PE2SE_ONLY, levels, self.mifpath, self.gdxpath, self.get_technosphere_df())
        else:
            cs = CalculationSetup(yaml_file)
            cs.build_removal_lists(self.get_technosphere_df())
            cs.regionalize_constant_mappings()
            self.cs = cs   

    def calculate_costs(
        self,
        monetization: float | str | dict,
        multiprocessing: bool = True,
        save_intermediate_results: bool = False,
        change_pm_compartments: bool = False,
    ) -> None:
        """
        Calculate all costs.
        """           
        # check monetization
        if isinstance(monetization, float):
            if monetization >= 1 or monetization <= 0:
                raise ValueError("Given number for cost perspective is not a "
                "valid quantile (not between 0 and 1)!")
        elif isinstance(monetization, str):
            if monetization not in COST_PERSPECTIVES:
                raise ValueError(f"Cost perspective must be one of {COST_PERSPECTIVES}.")
        elif isinstance(monetization, dict):
            check_monetization_factors(monetization)
        else:
            raise ValueError(f"Argument 'monetization' must be a float, string, or dictionary!")
        
        if not hasattr(self, "cs"):
            raise AttributeError("No calculation setup is set. " \
            "Run 'Internalizer.set_calculation_setup(...)` first. Exiting.")

        args = []
        for lvl in self.cs.levels:
            rlist = self.cs.data[lvl]["removal list"]
            for year in self.years:
                mapping = self.cs.data[lvl].get(
                    "regionalized mapping", self.cs.regionalize_dynamic_mapping(lvl, year)
                )
                    
                args.append(
                    (
                        mapping,
                        monetization,
                        rlist,
                        self.scenario,
                        year,
                        lvl,
                        self.outdir,
                        self.model,
                        save_intermediate_results,
                        change_pm_compartments,
                    )   
                )

        if multiprocessing:
            with Pool(self.max_mp_tasks) as p:
            # with Pool(cpu_count(), maxtasksperchild=1000) as p:
                p.starmap(_calculate_costs_year, args)

                # for lvl, y, r in zip(lvllist, yearlist, results):
                #     self.cost_results[lvl][y] = r
        else:
            for arg in args:
                _calculate_costs_year(*arg)

    def load_costs(
        self
    ) -> None:
        
        if not hasattr(self, "cs"):
            raise AttributeError("No calculation setup is set. Run 'Internalizer.set_calculation_setup(...)`\n",
                   "and `Internalizer.calculate_costs(...)`` first. Exiting.")
        
        # load raw costs and aggregate with mappings
        self.cost_results = {}
        for lvl in self.cs.levels:
            self.cost_results[lvl] = {}
            for year in self.years:
                # recalculate mapping
                mapping = self.cs.data[lvl].get(
                    "regionalized mapping", self.cs.regionalize_dynamic_mapping(lvl, year)
                )
                fp = self.outdir + f"/{self.model}/{self.scenario}/{str(year)}/regionalized_costs_{lvl}.csv"
                regionalized_costs = pd.read_csv(fp)
                costs_agg = aggregate_with_mapping(mapping, regionalized_costs)
                self.cost_results[lvl][year] = costs_agg.set_index(
                    ["REMIND index", "region", "impact category"]
                )["cost"].to_xarray()
        
    def calculate_aggregated_impacts(self) -> None:
        if not hasattr(self, "cs"):
            raise AttributeError("No calculation setup is set. Run 'Internalizer.set_calculation_setup(...)`\n",
                   "and `Internalizer.calculate_costs(...)`` first. Exiting.")
        
        dflist = []
        for agg_lvl, lvllist in self.cs.aggregate_levels.items():
            for lvl in lvllist:
                results = {}
                for year in self.years:
                    # recalculate mapping
                    mapping = self.cs.data[lvl].get(
                        "regionalized mapping", self.cs.regionalize_dynamic_mapping(lvl, year)
                    )
                    fp = self.outdir + f"/{self.model}/{self.scenario}/{str(year)}/regionalized_impacts_{lvl}.csv"
                    regionalized_impacts = pd.read_csv(fp)

                    impacts_agg = aggregate_with_mapping(
                        mapping, regionalized_impacts,
                        var_col="LCIA method",
                        value_col="impact"
                    )

                    results[year] = impacts_agg.set_index(
                        ["REMIND index", "region", "LCIA method"]
                    )["impact"].to_xarray()

                # interpolate
                x = interpolate_and_weight_xr(results, MODEL_YEARS[self.model], 1.0)

                # multiply with production volumes
                prodVol = self.cs.get_production_volumes(lvl, MODEL_YEARS[self.model])
                df = (prodVol * x).to_dataframe(name="impact").reset_index()
                df["level"] = agg_lvl
                dflist.append(df)

        pd.concat(dflist, ignore_index=True)[
            ["level", "REMIND index", "region", "year", "LCIA method", "impact"]].to_csv(
                os.path.join(self.outdir, "aggregated_impacts.csv"), index=False
            )
    
    def write_remind_input_files(
        self,
        ramp_up_startyear: int,
        ramp_up_endyear: int,
        impact_categories: List[str]
    ) -> None:
        if not hasattr(self, "cost_results"):
            raise AttributeError("No cost results available. " \
            "Run `.calculate_costs()` and `.load_costs()` first. Exiting.")
        
        ramp_up = get_linear_ramp_up(MODEL_YEARS[self.model], ramp_up_startyear, ramp_up_endyear)

        for agg_lvl, lvllist in self.cs.aggregate_levels.items():
            dflist = []
            for lvl in lvllist:
                domains = self.cs.data[lvl]["domains"]
                x = interpolate_and_weight_xr(
                        self.cost_results[lvl],
                        MODEL_YEARS[self.model],
                        ramp_up
                    )
                
                # convert to USD / GJ
                x =  x * convert_euros_to_dollar(EURO_REF_YEAR, REMIND_USD_REF_YEAR) * 1000
                    
                total = x.sel({"impact category": impact_categories}).sum(dim="impact category")
                df = total.to_dataframe(name="cost").reset_index()
                df = split_remind_index(df, domains).rename(columns={"year": "ttot", "region": "all_regi"})
                dflist.append(df[["ttot", "all_regi"] + domains + ["cost"]])
            pd.concat(dflist).to_csv(
                os.path.join(self.outdir, f"lca_costs_{agg_lvl}.csv"), index=False, header=True
            )
            


        

        