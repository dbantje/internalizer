from typing import List, Optional
from pathlib import Path
import os

from multiprocessing import Pool, cpu_count
import shutil

from .plca import _run_premise_year, _calculate_costs_year
from .utils import convert_euros_to_dollar, check_monetization_factors, get_linear_ramp_up, interpolate_and_weight_costs
from .calculation_setup import prepare_setup

EURO_REF_YEAR = 2022
REMIND_USD_REF_YEAR = 2017
COST_PERSPECTIVES = [
    "damage costs",
    "prevention costs",
    "budget constraint",
    "taxation costs"
]

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
        filepath: str,
        model: str,
        pathway: str,
        ei_version: str,
        bw_project: str,
        gdxpath = str,
        outputfolder: str = "output"
    ):
        # get directory of data file and scenario name
        self.model = model
        rundir, filename = extract_folder_and_filename(filepath)
        self.rundir = rundir
        self.outdir = f"./{outputfolder}/" + extract_output_folder(filepath)
        namecheck = "_".join((model.lower(), pathway))
        if filename != namecheck:
            shutil.copy(filepath, rundir+f"/{namecheck}.mif")
        self.scenario = pathway

        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)

        self.ei_version = ei_version 
        self.bw_project = bw_project
        self.gdxpath = gdxpath

    def run_premise(
        self,
        years: List[int],
        multiprocessing: bool = True
    ) -> None:
        self.years = years
        
        args = [
            (
                self.bw_project,
                {"model": self.model, "pathway": self.scenario, "year": year, "filepath": self.rundir},
                self.ei_version,
                self.outdir
            )
            for year in self.years
        ]

        if multiprocessing:
            with Pool(cpu_count(), maxtasksperchild=1000) as p:
                p.starmap(_run_premise_year, args)
        else:
            for arg in args:
                _run_premise_year(*arg)

    def calculate_costs(
        self,
        monetization: float | str | dict,
        activities_mapping: List[str | Path],
        level_names: Optional[List[str]] = None,
        multiprocessing: bool = True
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

        # obtain mappings and removal lists
        setup = prepare_setup(activities_mapping, self.gdxpath, level_names)
        self.levels = setup.keys()
        args = []
        yearlist = []
        lvllist = []
        for lvl in self.levels:
            mapping = setup[lvl]["mapping"]
            rlist = setup[lvl]["removal list"]
            for year in self.years:
                args.append(
                    (
                        mapping,
                        monetization,
                        rlist,
                        self.scenario,
                        year,
                        self.outdir,
                        self.model,
                    )   
                )
                yearlist.append(year)
                lvllist.append(lvl)

        self.cost_results = {}
        for lvl in self.levels:
            self.cost_results[lvl] = {}
        if multiprocessing:
            with Pool(cpu_count(), maxtasksperchild=1000) as p:
                results = p.starmap(_calculate_costs_year, args)

                for lvl, y, r in zip(lvllist, yearlist, results):
                    self.cost_results[lvl][y] = r
        else:
            for lvl, y, arg in zip(lvllist, yearlist, args):
                self.cost_results[lvl][y] = _calculate_costs_year(*arg)
    
    def write_remind_input_files(
        self,
        ramp_up_startyear: int,
        ramp_up_endyear: int,
        impact_categories: List[str]
    ) -> None:
        ramp_up = get_linear_ramp_up(MODEL_YEARS["remind"], ramp_up_startyear, ramp_up_endyear)

        for lvl in self.levels:
            x = interpolate_and_weight_costs(
                    self.cost_results[lvl],
                    MODEL_YEARS["remind"],
                    ramp_up
                )
            
            # convert to USD / GJ
            x =  x * convert_euros_to_dollar(EURO_REF_YEAR, REMIND_USD_REF_YEAR) * 1000
                
            total = x.sel({"impact category": impact_categories}).sum(dim="impact category")
            df = total.to_dataframe().reset_index()
            df[["year", "region", "REMIND index", "cost"]].to_csv(
                self.outdir + f"/lca_costs_{lvl}.csv", index=False, header=False)

        

        