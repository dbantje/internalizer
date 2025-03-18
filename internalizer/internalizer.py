from typing import List, Optional, Tuple
from pathlib import Path
import os

from multiprocessing import Pool, cpu_count
import bw2data as bd
import pandas as pd
import numpy as np
import xarray as xr
import shutil

from .plca import _run_premise_year, _calculate_costs_year, _calculate_costs_year_new
from .regionalization import get_regionalized_mapping
from .utils import convert_euros_to_dollar

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
        bw_project: str
    ):
        # get directory of data file and scenario name
        self.model = model
        rundir, filename = extract_folder_and_filename(filepath)
        self.rundir = rundir
        self.outdir = "./output/" + extract_output_folder(filepath)
        namecheck = "_".join((model.lower(), pathway))
        if filename != namecheck:
            shutil.copy(filepath, rundir+f"/{namecheck}.mif")
        self.scenario = pathway

        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)

        self.ei_version = ei_version 
        self.bw_project = bw_project

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

    def calculate_costs_new(
            self,
            regionalized_mapping: pd.DataFrame,
            cost_perspective: str | float,
            remove_double_counting: bool,
            extra_activities: List[Tuple] = [],
            multiprocessing: bool = True
    ) -> None:
        # check cost perspectives
        if isinstance(cost_perspective, float):
            if cost_perspective >= 1 or cost_perspective <= 0:
                raise ValueError("Given number for cost perspective is not a "
                "valid quantile (not between 0 and 1)!")
        else:
            if cost_perspective not in COST_PERSPECTIVES:
                raise ValueError(f"Cost perspective must be one of {COST_PERSPECTIVES}.")
            
        args = [
            (
                regionalized_mapping,
                cost_perspective,
                remove_double_counting,
                extra_activities,
                self.scenario,
                year,
                self.outdir,
                self.model,
            )
            for year in self.years
        ]

        self.cost_results = {}
        if multiprocessing:
            with Pool(cpu_count(), maxtasksperchild=1000) as p:
                results = p.starmap(_calculate_costs_year_new, args)

                for y, r in zip(self.years, results):
                    self.cost_results[y] = r
        else:
            for year, arg in zip(self.years, args):
                self.cost_results[year] = _calculate_costs_year_new(*arg)

    def calculate_costs(
        self,
        mapping: pd.DataFrame,
        quantiles: np.ndarray,
        multiprocessing: bool = True
    ) -> None:
        # regionalize mapping
        all_shares = get_regionalized_mapping(mapping, self.rundir)

        args = [
            (
                all_shares,
                self.scenario,
                year,
                self.outdir,
                self.model,
                quantiles
            )
            for year in self.years
        ]

        self.cost_results = {}
        if multiprocessing:
            with Pool(cpu_count(), maxtasksperchild=1000) as p:
                results = p.starmap(_calculate_costs_year, args)

                for y, r in zip(self.years, results):
                    self.cost_results[y] = r
        else:
            for year, arg in zip(self.years, args):
                self.cost_results[year] = _calculate_costs_year(*arg)

    def export_costs(
            self,
            ramp_up_startyear: Optional[int] = None,
    ) -> xr.DataArray:
        years_extended = MODEL_YEARS[self.model]

        data = self.cost_results.copy()

        years_in_data = np.array(list(data.keys()))
        first_year = years_extended[0]
        years_before = [y for y in years_extended if y < years_in_data[0]]
        years_after = [y for y in years_extended if y > years_in_data[-1]]
        last_year = years_extended[-1]

        last_array = data[max(years_in_data)]
        zero_array = xr.zeros_like(last_array)
        
        if len(years_before) > 0:
            # add some years before
            data[first_year] = zero_array
            closest_year_before = max(years_before)
            if ramp_up_startyear is None:
                if first_year < closest_year_before:
                    data[closest_year_before] = zero_array
            else:
                ramp_start = min(closest_year_before, ramp_up_startyear)
                if first_year < ramp_start: 
                    data[ramp_start] = zero_array

        if len(years_after) > 0:
            # add last year
            data[last_year] = last_array

        data = dict(sorted(data.items()))

        return xr.concat(
            list(data.values()),
            pd.Index(list(data.keys()), name="year")
        ).interp(year=years_extended)
    
    def write_remind_input_file(
            self,
            ramp_up_startyear,
            excluded_impacts: List[str] = ["fossil resources", "climate change"],
            q: float | List[float] = 0.5,
    ) -> None:
        x = self.export_costs(ramp_up_startyear=ramp_up_startyear)
        x = x.sel({"quantile": q})

        # convert to USD / GJ
        x =  x * convert_euros_to_dollar(EURO_REF_YEAR, REMIND_USD_REF_YEAR) * 1000

        total = x.drop_sel({"impact category": excluded_impacts}).sum(dim="impact category")
        df = total.to_dataframe().reset_index()
        df["all_te"] = df["REMIND tech"].apply(lambda x: x.split(".")[-1])
        df["quantile"] = df["quantile"].apply(lambda x: str(int(100 * x)))
        df[["year", "region", "all_te", "quantile", "cost"]].to_csv(self.outdir + "/inputfile.cs4r", index=False, header=False)