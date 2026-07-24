from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# ============================
# Configuration
# ============================

SE = {
    "ethanol_conv": "SE|Liquids|Biomass|Conventional Ethanol",
    "ethanol_ligno": "SE|Liquids|Biomass|Lignocellulosic Ethanol",
    "biodiesel": "SE|Liquids|Biomass|Biodiesel",
    "ft_wo_cc": "SE|Liquids|Biomass|BioFTR|w/o CC",
    "ft_w_cc": "SE|Liquids|Biomass|BioFTR|w/ CC",
    "ft_pyro": "SE|Liquids|Biomass|BioFTR|w/ pyrolysis",

    "cell_resid": "SE|Liquids|Biomass|Cellulosic|++|Residues",
    "cell_crops": "SE|Liquids|Biomass|Cellulosic|++|Energy Crops",

    "noncell_sugar": "SE|Liquids|Biomass|Non-Cellulosic|+|Sugar and Starch",
    "noncell_oil": "SE|Liquids|Biomass|Non-Cellulosic|+|Oil-based",
}

# --- Fallback: map pools (G/D) to feedstock -> SE variables to sum ---
# This is ONLY used when the main proxy shares are all-zero.
# It respects your constraint: Sugar&Starch never goes to diesel-like, Oil-based never goes to gasoline-like.
FALLBACK_POOL_VARS = {
    "G": {  # gasoline-like / ethanol-like
        "Non-Cellulosic|Sugar and Starch": [
            SE["ethanol_conv"],
            SE["noncell_sugar"],
        ],
        "Cellulosic|Residues": [
            SE["cell_resid"],
        ],
        "Cellulosic|Energy Crops": [
            SE["cell_crops"],
        ],
    },
    "D": {  # diesel-like
        "Non-Cellulosic|Oil-based": [
            SE["biodiesel"],
            SE["noncell_oil"],
        ],
        "Cellulosic|Residues": [
            SE["cell_resid"],
        ],
        "Cellulosic|Energy Crops": [
            SE["cell_crops"],
        ],
    },
}


# Example structure: pool name -> feedstock -> list of SE variables that represent that feedstock in that pool
SE_POOL_VARS = {
  "ethanol_like": {
    "first_gen": [
      "SE|Liquids|Biomass|Conventional Ethanol",
      "SE|Liquids|Biomass|Non-Cellulosic|+|Sugar and Starch",
    ],
    # Cellulosic ethanol is reported, but not split by crop/residue.
    # If you want crop/residue inside the ethanol pool, proxy it using the cellulosic split:
    "energy_crops": ["SE|Liquids|Biomass|Lignocellulosic Ethanol|+|Energy Crops"],
    "residues":     ["SE|Liquids|Biomass|Lignocellulosic Ethanol|+|Residues"],
    # Optional validation-only (do not include in sums to avoid double counting):
    # "validation": ["SE|Liquids|Biomass|Lignocellulosic Ethanol"]
  },
  "diesel_like": {
    "first_gen": [
      "SE|Liquids|Biomass|Non-Cellulosic|+|Oil-based",
    ],
    "energy_crops": [
        "SE|Liquids|Biomass|BioFTR|w/o CC|+|Energy Crops",
        "SE|Liquids|Biomass|BioFTR|w/ CC|+|Energy Crops",
        "SE|Liquids|Biomass|BioFTR|w/ pyrolysis|+|Energy Crops",
    ],
    "residues": [
        "SE|Liquids|Biomass|BioFTR|w/o CC|+|Residues",
        "SE|Liquids|Biomass|BioFTR|w/ CC|+|Residues",
        "SE|Liquids|Biomass|BioFTR|w/ pyrolysis|+|Residues",
    ],
    # Same issue: cellulosic product split is not explicit. You can either:
    # (A) include the cellulosic crop/residue proxies here too (but then ethanol_like also includes them → double count),
    # or (B) keep cellulosic only once globally and split later.
    #
    # I recommend (B): do NOT include cellulosic here if you also include it in ethanol_like.
  },
}


FEEDSTOCK_KEYS = [
    "Non-Cellulosic|Sugar and Starch",
    "Non-Cellulosic|Oil-based",
    "Cellulosic|Residues",
    "Cellulosic|Energy Crops",
]

POOL_ALLOWED_FEEDSTOCKS = {
    "G": {
        "Non-Cellulosic|Sugar and Starch",
        "Cellulosic|Residues",
        "Cellulosic|Energy Crops",
    },
    "D": {
        "Non-Cellulosic|Oil-based",
        "Cellulosic|Residues",
        "Cellulosic|Energy Crops",
    },
}

TRANSPORT_PREFIX = "FE|Transport|"

GASOLINE_TOKENS = [
    r"\|Pass\|Road\|LDV\|",
    r"\|Two Wheelers\|",
    r"\|Motorcycle",
    r"\|Moped\|",
    r"\|Three Wheelers\|",
    r"\|Rickshaw\|",
    r"\|Pass\|Road\|Liquids\|",
    r"\|Pass\|Short-Medium distance\|Petrol Liquids",
]
DIESEL_TOKENS = [
    r"\|Freight\|",
    r"\|Bus\|",
    r"\|Truck",
    r"\|Rail\|",
    r"\|Aviation\|",
    r"\|Shipping\|",
    r"\|Bunkers\|",
    r"\|Diesel Liquids",
]


# ============================
# Helpers
# ============================

def year_columns(df: pd.DataFrame) -> List[int]:
    years: List[int] = []
    for c in df.columns:
        if isinstance(c, (int, np.integer)):
            years.append(int(c))
        elif isinstance(c, str) and re.fullmatch(r"\d{4}", c.strip()):
            years.append(int(c.strip()))
    return sorted(set(years))


def normalize_year_columns_to_int(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ren = {}
    for c in df.columns:
        if isinstance(c, str) and re.fullmatch(r"\d{4}", c.strip()):
            ren[c] = int(c.strip())
    if ren:
        df.rename(columns=ren, inplace=True)
    return df


def safe_div(num: float, den: float, default: float) -> float:
    if den == 0 or np.isclose(den, 0.0):
        return default
    return num / den


def classify_pool(fe_variable: str) -> str:
    fe_variable = str(fe_variable)
    if not fe_variable.startswith(TRANSPORT_PREFIX):
        return "D"
    for pat in DIESEL_TOKENS:
        if re.search(pat, fe_variable):
            return "D"
    for pat in GASOLINE_TOKENS:
        if re.search(pat, fe_variable):
            return "G"
    if re.search(r"\|LDV\|", fe_variable):
        return "G"
    return "D"


# ============================
# Share model
# ============================

@dataclass(frozen=True)
class FeedstockShares:
    poolG: Dict[str, float]
    poolD: Dict[str, float]


def compute_feedstock_shares_for_group(group: pd.DataFrame, years: List[int]) -> Dict[int, FeedstockShares]:
    """
    Compute per-year feedstock shares for gasoline-like (G) and diesel-like (D) pools
    for ONE (Model, Scenario, Region) group.

    Performance: use one groupby wide table instead of scanning per variable.
    """

    # Aggregate duplicates once (huge speedup vs repeated boolean filtering)
    sub = group[["Variable", *years]].copy()
    wide = sub.groupby("Variable", sort=False)[years].sum()

    def val(var: str, y: int) -> float:
        if var in wide.index and y in wide.columns:
            v = wide.at[var, y]
            return 0.0 if pd.isna(v) else float(v)
        return 0.0

    out: Dict[int, FeedstockShares] = {}

    for y in years:
        # Cellulosic split (residues vs energy crops)
        cell_res = val(SE["cell_resid"], y)
        cell_crp = val(SE["cell_crops"], y)
        cell_total = cell_res + cell_crp
        if np.isclose(cell_total, 0.0):
            s_res, s_crp = 0.5, 0.5
        else:
            s_res = safe_div(cell_res, cell_total, default=0.5)
            s_crp = 1.0 - s_res

        # Pool G proxy: conventional vs lignocellulosic ethanol
        conv = val(SE["ethanol_conv"], y)
        ligno = val(SE["ethanol_ligno"], y)
        denomG = conv + ligno
        if np.isclose(denomG, 0.0):
            g_sugar = 0.0
            g_cell = 0.0
        else:
            g_sugar = safe_div(conv, denomG, default=0.0)
            g_cell = 1.0 - g_sugar

        poolG = {
            "Non-Cellulosic|Sugar and Starch": g_sugar,
            "Non-Cellulosic|Oil-based": 0.0,
            "Cellulosic|Residues": g_cell * s_res,
            "Cellulosic|Energy Crops": g_cell * s_crp,
        }

        # Pool D proxy: biodiesel vs FT liquids
        biod = val(SE["biodiesel"], y)
        ft = val(SE["ft_wo_cc"], y) + val(SE["ft_w_cc"], y) + val(SE["ft_pyro"], y)
        denomD = biod + ft
        if np.isclose(denomD, 0.0):
            d_oil = 0.0
            d_cell = 0.0
        else:
            d_oil = safe_div(biod, denomD, default=0.0)
            d_cell = 1.0 - d_oil

        poolD = {
            "Non-Cellulosic|Sugar and Starch": 0.0,
            "Non-Cellulosic|Oil-based": d_oil,
            "Cellulosic|Residues": d_cell * s_res,
            "Cellulosic|Energy Crops": d_cell * s_crp,
        }

        out[y] = FeedstockShares(poolG=poolG, poolD=poolD)

    return out



def fallback_feedstock_shares_from_SE(group: pd.DataFrame, years: list[int], pool_vars: dict[str, list[str]]) -> pd.DataFrame:
    """
    Compute fallback feedstock shares from SE pools for ONE (Model, Scenario, Region) group.

    pool_vars: e.g. {"ethanol": [...vars...], "diesel": [...vars...]}
    Returns a DataFrame indexed by feedstock with columns=years (shares sum to 1 where pool>0).
    """

    # Only variables we actually need
    needed = set()
    for vlist in pool_vars.values():
        needed.update(vlist)

    # Filter early (huge speedup)
    sub = group[group["Variable"].isin(needed)][["Variable", *years]].copy()
    if sub.empty:
        # caller can decide what to do if no SE pool data
        return pd.DataFrame(index=[], columns=years, dtype=float)

    # In case some variables appear multiple times, aggregate
    wide = sub.groupby("Variable", sort=False)[years].sum()

    # Helper to safely fetch a row (returns zeros if missing)
    def get_row(var: str) -> "pd.Series":
        if var in wide.index:
            return wide.loc[var]
        return pd.Series(0.0, index=years)

    # Example logic: build pool totals and then compute shares
    # (Adapt these names to your exact pool_vars structure)
    # Suppose your pool_vars defines mapping from feedstock -> contributing SE vars:
    # pool_vars = {
    #   "energy_crops": ["SE|...|Ethanol|Energy Crops", "SE|...|FT|Energy Crops", ...],
    #   "residues":     ["SE|...|Ethanol|Residues",     "SE|...|FT|Residues", ...],
    #   ...
    # }

    feedstocks = list(pool_vars.keys())
    totals = {}
    for fs in feedstocks:
        s = pd.Series(0.0, index=years)
        for var in pool_vars[fs]:
            s = s.add(get_row(var), fill_value=0.0)
        totals[fs] = s

    pool_total = pd.Series(0.0, index=years)
    for fs in feedstocks:
        pool_total = pool_total.add(totals[fs], fill_value=0.0)

    # Shares: totals[fs] / pool_total, but avoid division by 0
    shares = {}
    for fs in feedstocks:
        denom = pool_total.copy()
        denom[denom == 0.0] = float("nan")
        shares[fs] = (totals[fs] / denom).fillna(0.0)

    return pd.DataFrame(shares).T[years]




# ============================
# Allocation for one .mif DataFrame
# ============================

def allocate_one_mif(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    df = normalize_year_columns_to_int(df)
    years = year_columns(df)
    if not years:
        raise ValueError("No year columns found (expected 4-digit year columns).")

    required = {"Model", "Scenario", "Region", "Variable", "Unit"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    group_keys = ["Model", "Scenario", "Region"]

    fe_mask = (
        df["Variable"].astype(str).str.startswith("FE|", na=False)
        & df["Variable"].astype(str).str.contains(r"\|Liquids\|", regex=True, na=False)
        & df["Variable"].astype(str).str.contains("Biomass", na=False)
    )
    fe_df = df.loc[fe_mask].copy()

    counters = {
        "rows_allocated": 0,
        "classified_G": 0,
        "classified_D": 0,
        "fallback_proxy_zero": 0,
        "fallback_allzero": 0,
    }

    pool_rows: List[dict] = []
    alloc_long_rows: List[dict] = []

    def _as_dict(p) -> dict:
        """Normalize proxy container to a plain dict[str,float]."""
        if isinstance(p, dict):
            return p
        if isinstance(p, pd.Series):
            return {k: float(v) for k, v in p.to_dict().items()}
        raise TypeError(f"Unsupported proxy type: {type(p)}")

    for (m, s, r), grp in df.groupby(group_keys, dropna=False):
        shares_by_year = compute_feedstock_shares_for_group(grp, years)

        # Compute fallback shares ONCE per group, for both pools (big speedup)
        fb_shares = {
            "G": fallback_feedstock_shares_from_SE(grp, years, FALLBACK_POOL_VARS["G"]),
            "D": fallback_feedstock_shares_from_SE(grp, years, FALLBACK_POOL_VARS["D"]),
        }

        fe_grp = fe_df[(fe_df["Model"] == m) & (fe_df["Scenario"] == s) & (fe_df["Region"] == r)]
        if fe_grp.empty:
            continue

        for _, fe_row in fe_grp.iterrows():
            fe_var = str(fe_row["Variable"])
            # print("Allocating FE variable:", fe_var, "in group:", (m, s, r))
            pool = classify_pool(fe_var)  # "G" or "D"
            # print(f"\tClassified as pool {pool}")
            fb_by_year = fb_shares[pool]  # DataFrame (feedstock x years)

            counters["rows_allocated"] += 1
            counters["classified_G"] += 1 if pool == "G" else 0
            counters["classified_D"] += 1 if pool == "D" else 0

            pr = {k: fe_row[k] for k in ["Model", "Scenario", "Region", "Variable", "Unit"]}
            pr["Pool"] = pool
            for y in years:
                pr[y] = float(fe_row[y])
            pool_rows.append(pr)

            meta = {k: fe_row[k] for k in ["Model", "Scenario", "Region", "Unit"]}

            for y in years:
                fe_val = float(fe_row[y])
                if np.isclose(fe_val, 0.0):
                    continue

                sh = shares_by_year[y]
                proxy = sh.poolG if pool == "G" else sh.poolD
                proxy = _as_dict(proxy)

                # If main proxy is zero, try fallback from SE_POOL_VARS (per pool)
                if np.isclose(sum(proxy.values()), 0.0):
                    counters["fallback_proxy_zero"] += 1

                    if (not fb_by_year.empty) and (y in fb_by_year.columns):
                        fb_series = fb_by_year[y]  # Series indexed by feedstock keys
                        proxy = _as_dict(fb_series)
                    else:
                        proxy = {}

                    # If still zero, uniform over allowed feedstocks for that pool
                    if np.isclose(sum(proxy.values()), 0.0):
                        counters["fallback_allzero"] += 1
                        allowed = POOL_ALLOWED_FEEDSTOCKS[pool]
                        proxy = {k: 1.0 / len(allowed) for k in allowed}

                # Normalize to sum=1
                s_proxy = float(sum(proxy.values()))
                if not np.isclose(s_proxy, 1.0) and not np.isclose(s_proxy, 0.0):
                    proxy = {k: safe_div(v, s_proxy, 0.0) for k, v in proxy.items()}

                # print(f"\tFinal proxy for {fe_var}: {proxy}")

                for fs in FEEDSTOCK_KEYS:
                    alloc_long_rows.append({
                        **meta,
                        "Variable": f"{fe_var}|Allocated Feedstock|{fs}",
                        "Year": y,
                        "Value": fe_val * float(proxy.get(fs, 0.0)),
                    })


    # Build allocated in wide format
    alloc_long = pd.DataFrame(alloc_long_rows)
    if alloc_long.empty:
        alloc_wide = pd.DataFrame(columns=["Model", "Scenario", "Region", "Variable", "Unit"] + years)
    else:
        alloc_wide = (
            alloc_long.pivot_table(
                index=["Model", "Scenario", "Region", "Variable", "Unit"],
                columns="Year",
                values="Value",
                aggfunc="sum",
                fill_value=0.0,
            )
            .reset_index()
        )
        for y in years:
            if y not in alloc_wide.columns:
                alloc_wide[y] = 0.0
        alloc_wide = alloc_wide[["Model", "Scenario", "Region", "Variable", "Unit"] + years]

    pool_df = pd.DataFrame(pool_rows)
    if not pool_df.empty:
        pool_df = pool_df[["Model", "Scenario", "Region", "Variable", "Unit", "Pool"] + years]

    # Validation + top offenders (by abs error)
    validation, offenders = compute_validation(df, alloc_wide, pool_df, years)

    # Output scenario table = original + appended allocated
    out_df = pd.concat([df, alloc_wide], ignore_index=True)

    return out_df, alloc_wide, pool_df, validation, offenders, counters


def compute_validation(
    original: pd.DataFrame,
    allocated_feedstocks: pd.DataFrame,
    fe_pool_table: pd.DataFrame,
    years: List[int],
    eps_rel: float = 1e-6,
    eps_den: float = 1e-12,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validations:
      - Hard FE closure (abs + rel where |FE|>eps_rel) per-row and aggregate
      - Soft diagnostics (composition proxy only): FE_pool / SE_proxy ratios
        computed ONLY where both FE and SE proxies exist for the same (Model,Scenario,Region).
    """
    keys = ["Model", "Scenario", "Region"]

    # --- Hard closure: FE rows to be allocated ---
    fe_rows = original[
        original["Variable"].astype(str).str.startswith("FE|", na=False)
        & original["Variable"].astype(str).str.contains(r"\|Liquids\|", regex=True, na=False)
        & original["Variable"].astype(str).str.contains("Biomass", na=False)
    ][keys + ["Variable"] + years].copy()

    alloc = allocated_feedstocks.copy()
    alloc["BaseVariable"] = alloc["Variable"].str.split(r"\|Allocated Feedstock\|", n=1, expand=True)[0]
    alloc_sum = (
        alloc.groupby(keys + ["BaseVariable"], as_index=False)[years]
        .sum()
        .rename(columns={"BaseVariable": "Variable"})
    )

    merged = fe_rows.merge(alloc_sum, on=keys + ["Variable"], how="left", suffixes=("_orig", "_alloc"))
    for y in years:
        col = f"{y}_alloc"
        if col not in merged.columns:
            merged[col] = 0.0
        merged[col] = merged[col].fillna(0.0)

    # Row-wise max errors
    if merged.empty:
        max_abs = 0.0
        max_rel = 0.0
        offenders = pd.DataFrame(columns=keys + ["Variable", "max_abs_err", "max_rel_err"])
        agg_max_abs = 0.0
    else:
        abs_err_row = []
        rel_err_row = []
        for _, r in merged.iterrows():
            o = r[[f"{y}_orig" for y in years]].to_numpy(dtype=float)
            a = r[[f"{y}_alloc" for y in years]].to_numpy(dtype=float)
            e = np.abs(o - a)
            abs_err_row.append(float(np.nanmax(e)))

            mask = np.abs(o) > eps_rel
            if np.any(mask):
                rel = e[mask] / np.maximum(np.abs(o[mask]), eps_den)
                rel_err_row.append(float(np.nanmax(rel)))
            else:
                rel_err_row.append(np.nan)

        merged["max_abs_err"] = abs_err_row
        merged["max_rel_err"] = rel_err_row

        max_abs = float(np.nanmax(merged["max_abs_err"].to_numpy(dtype=float)))
        max_rel = float(np.nanmax(merged["max_rel_err"].to_numpy(dtype=float)))

        offenders = (
            merged.sort_values("max_abs_err", ascending=False)
            .loc[:, keys + ["Variable", "max_abs_err", "max_rel_err"]]
            .head(20)
            .reset_index(drop=True)
        )

        agg_max_abs = float(
            np.max(
                np.abs(
                    (fe_rows[years].sum() - alloc[years].sum()).to_numpy(dtype=float)
                )
            )
        )

    # --- Soft diagnostics: FE pools vs SE proxies ---
    g_stats = (np.nan, np.nan, np.nan)
    d_stats = (np.nan, np.nan, np.nan)

    if (not fe_pool_table.empty):
        # FE pools aggregated to keys
        fe_g = fe_pool_table[fe_pool_table["Pool"] == "G"].groupby(keys, as_index=False)[years].sum()
        fe_d = fe_pool_table[fe_pool_table["Pool"] == "D"].groupby(keys, as_index=False)[years].sum()

        # SE proxy rows present in file (may be only World etc.)
        se_subset = original[
            original["Variable"].isin([
                SE["ethanol_conv"], SE["ethanol_ligno"],
                SE["biodiesel"], SE["ft_wo_cc"], SE["ft_w_cc"], SE["ft_pyro"],
            ])
        ][keys + ["Variable"] + years].copy()

        if not se_subset.empty:
            # Build SE pools by grouping keys, summing selected variables
            se_eth = se_subset[se_subset["Variable"].isin([SE["ethanol_conv"], SE["ethanol_ligno"]])].groupby(keys, as_index=False)[years].sum()
            se_dsl = se_subset[se_subset["Variable"].isin([SE["biodiesel"], SE["ft_wo_cc"], SE["ft_w_cc"], SE["ft_pyro"]])].groupby(keys, as_index=False)[years].sum()

            # Merge so we compute ratios ONLY where both exist
            g_merge = fe_g.merge(se_eth, on=keys, how="inner", suffixes=("_fe", "_se"))
            d_merge = fe_d.merge(se_dsl, on=keys, how="inner", suffixes=("_fe", "_se"))

            def ratio_stats(merged_df: pd.DataFrame) -> Tuple[float, float, float]:
                if merged_df.empty:
                    return (np.nan, np.nan, np.nan)

                # Extract FE and SE matrices with aligned year columns
                fe_mat = merged_df[[f"{y}_fe" for y in years]].to_numpy(dtype=float)
                se_mat = merged_df[[f"{y}_se" for y in years]].to_numpy(dtype=float)

                eps_soft = 1e-6
                mask = (fe_mat > eps_soft) | (se_mat > eps_soft)
                ratio = fe_mat / (se_mat + eps_den)
                flat = ratio[mask]
                flat = flat[np.isfinite(flat)]
                if flat.size == 0:
                    return (np.nan, np.nan, np.nan)
                return (float(np.nanmin(flat)), float(np.nanmedian(flat)), float(np.nanmax(flat)))

            g_stats = ratio_stats(g_merge)
            d_stats = ratio_stats(d_merge)

    summary = pd.DataFrame([
        {"Check": "FE closure (hard): max_abs_err over all FE|...|Liquids|Biomass rows and years [PJ]", "Value": f"{max_abs:.6g}"},
        {"Check": f"FE closure (hard): max_rel_err where |FE|>{eps_rel} PJ (NaN if no points)", "Value": f"{max_rel:.6g}"},
        {"Check": "FE closure (hard): aggregate max abs( sum(FE) - sum(allocated feedstocks) ) over years [PJ]", "Value": f"{agg_max_abs:.6g}"},
        {"Check": "Soft diagnostic (not a balance): FE gasoline-like / SE ethanol proxy ratio (min, median, max)",
         "Value": f"min={g_stats[0]:.6g}, median={g_stats[1]:.6g}, max={g_stats[2]:.6g}"},
        {"Check": "Soft diagnostic (not a balance): FE diesel-like / SE (biodiesel+FT) proxy ratio (min, median, max)",
         "Value": f"min={d_stats[0]:.6g}, median={d_stats[1]:.6g}, max={d_stats[2]:.6g}"},
        {"Check": "Interpretation note", "Value": "SE proxies are used for composition only; ratios ≠ 1 are expected. Ratios are computed only where FE and SE exist for the same (Model,Scenario,Region)."},
    ])

    return summary, offenders



# ============================
# .mif I/O + batch processing
# ============================

def read_mif(path: Path) -> pd.DataFrame:
    # Your example shows a clean header line with ';' separators and no comment lines.
    return pd.read_csv(path, sep=";", engine="python")


def write_mif(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, sep=";", index=False)


def process_mif_file(mifpath: str, output_folder: Optional[str] = None, suffix: str = "_feedstock") -> None:
    mif_path = Path(mifpath).expanduser().resolve()
    if not mif_path.exists():
        raise FileNotFoundError(f"Input .mif file does not exist: {mif_path}")

    out_dir = Path(output_folder).expanduser().resolve() if output_folder else mif_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_mif(mif_path)
    out_df, alloc_wide, pool_df, validation, offenders, counters = allocate_one_mif(df)

    out_path = out_dir / f"{mif_path.stem}{suffix}{mif_path.suffix}"
    write_mif(out_df, out_path)

    print(
        f"[OK] {mif_path.name} -> {out_path.name} | "
        f"allocated_rows={counters['rows_allocated']} | "
        f"G={counters['classified_G']} D={counters['classified_D']} | "
        f"fallback_proxy_zero={counters['fallback_proxy_zero']}"
    )


def process_folder(input_folder: str, output_folder: Optional[str] = None, suffix: str = "_feedstock", mask=None) -> None:
    in_dir = Path(input_folder).expanduser().resolve()
    if not in_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {in_dir}")

    out_dir = Path(output_folder).expanduser().resolve() if output_folder else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mif_files = sorted(in_dir.rglob("*.mif"))
    if not mif_files:
        print(f"No .mif files found under: {in_dir}")
        return
    
    if mask is not None:
        mif_files = [p for p in mif_files if not any([m in p.name for m in mask])]
        if not mif_files:
            print(f"No .mif files found under: {in_dir} that do not contain '{mask}'")
            return

    for p in mif_files:
        try:
            df = read_mif(p)

            out_df, alloc_wide, pool_df, validation, offenders, counters = allocate_one_mif(df)

            out_path = out_dir / f"{p.stem}{suffix}{p.suffix}"
            write_mif(out_df, out_path)

            # Write per-file diagnostics
            validation_path = out_dir / f"{p.stem}{suffix}_validation.csv"
            offenders_path = out_dir / f"{p.stem}{suffix}_closure_offenders.csv"
            counters_path = out_dir / f"{p.stem}{suffix}_counters.csv"

            validation.to_csv(validation_path, index=False)
            offenders.to_csv(offenders_path, index=False)
            pd.DataFrame([counters]).to_csv(counters_path, index=False)

            print(
                f"[OK] {p.name} -> {out_path.name} | "
                f"allocated_rows={counters['rows_allocated']} | "
                f"G={counters['classified_G']} D={counters['classified_D']} | "
                f"fallback_proxy_zero={counters['fallback_proxy_zero']}"
            )

        except Exception as e:
            print(f"[FAIL] {p} -> {type(e).__name__}: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Allocate FE bio-liquids to biomass feedstocks for all .mif files in a folder.")
    parser.add_argument("input_folder", type=str, help="Folder to scan recursively for .mif files")
    parser.add_argument("--out", dest="output_folder", type=str, default=None, help="Output folder (default: input folder)")
    parser.add_argument("--suffix", type=str, default="_feedstock", help="Suffix appended to output .mif files")
    args = parser.parse_args()

    process_folder(args.input_folder, args.output_folder, args.suffix)
