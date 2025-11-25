from __future__ import annotations

import re
from typing import Optional, Tuple, Set

import numpy as np
import pandas as pd

try:
    # Used only to standardize calendars; no training occurs here.
    from forecasting.pooled_ridge import ForecastConfig, FeatureBuilder  # type: ignore
except Exception:  # pragma: no cover - optional import guard
    ForecastConfig = None  # type: ignore
    FeatureBuilder = None  # type: ignore


def _digits_key(x: object) -> str:
    s = "" if pd.isna(x) else str(x).strip().replace(".0", "")
    return re.sub(r"\D", "", s)


def _build_key_map(geo_stores: pd.DataFrame) -> pd.DataFrame:
    """Build a mapping from various store ids to Heatmap store id (hm_key).

    Expects columns: "Store_Number", "Heatmap_Store_Number" on geo_stores.
    Returns a DataFrame with columns: store_key, hm_key (both digit-only strings).
    """
    g = geo_stores.copy()
    g.columns = g.columns.str.strip()
    if not {"Store_Number", "Heatmap_Store_Number"}.issubset(g.columns):
        raise RuntimeError("geo_stores must have columns: Store_Number, Heatmap_Store_Number")
    m = g[["Store_Number", "Heatmap_Store_Number"]].dropna().drop_duplicates()
    m["store_key"] = m["Store_Number"].apply(_digits_key)
    m["hm_key"] = m["Heatmap_Store_Number"].apply(_digits_key)
    key_to_hm = pd.concat(
        [
            m[["store_key", "hm_key"]],
            m[["hm_key"]].assign(store_key=m["hm_key"])[["store_key", "hm_key"]],
        ],
        ignore_index=True,
    ).dropna().drop_duplicates("store_key")
    return key_to_hm


def _attach_hm_key(df: pd.DataFrame, store_col: str, key_to_hm: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[store_col] = out[store_col].astype(str).str.strip()
    out["store_key"] = out[store_col].apply(_digits_key)
    out = out.merge(key_to_hm, on="store_key", how="left")
    out["hm_key"] = out["hm_key"].fillna(out["store_key"])  # fallback if unmapped
    return out


def _ensure_calendar(calendar: pd.DataFrame) -> pd.DataFrame:
    """Standardize fiscal columns and derive days_in_period if possible.

    If forecasting.pooled_ridge's FeatureBuilder is available, use it to normalize
    calendar columns. Otherwise, assume the calendar already has fiscal_year,
    fiscal_period and optional BeginningDate/EndingDate.
    """
    c = calendar.copy()
    if FeatureBuilder is not None and ForecastConfig is not None:
        cfg_tmp = ForecastConfig(feature_list=[], verbose=False)  # type: ignore
        fb_tmp = FeatureBuilder(cfg_tmp)  # type: ignore
        c = fb_tmp._standardize_fiscal_cols(fb_tmp._ensure_calendar(c))
    # Coerce dates if present
    if "BeginningDate" in c.columns:
        c["BeginningDate"] = pd.to_datetime(c["BeginningDate"], errors="coerce")
    if "EndingDate" in c.columns:
        c["EndingDate"] = pd.to_datetime(c["EndingDate"], errors="coerce")
    # days_in_period
    if "days_in_period" not in c.columns and {"BeginningDate", "EndingDate"}.issubset(c.columns):
        c["days_in_period"] = (c["EndingDate"] - c["BeginningDate"]).dt.days + 1
    # Basic guard
    if not {"fiscal_year", "fiscal_period"}.issubset(c.columns):
        raise KeyError(f"Calendar missing fiscal columns. Available: {list(c.columns)}")
    return c


def build_labor_forecasts(
    geo_stores: pd.DataFrame,
    labor: pd.DataFrame,
    calendar: pd.DataFrame,
    sales_history: pd.DataFrame,
    final_fc: pd.DataFrame,
    allowed_hm_keys: Optional[Set[str]] = None,
    excluded_hm_keys: Optional[Set[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convert final store-period sales forecasts to labor hours at store and department levels.

    Inputs
    - geo_stores: store metadata with columns {Store_Number, Heatmap_Store_Number}.
    - labor: granular labor rows with a date column (one of event_date/date/work_date/shift_date),
      a store id column (e.g., Labor_Store_Number / store_number / Store_Number), optional department
      column (dept/department), and hour columns (total_hours or normal/ot/dt hours).
    - calendar: fiscal calendar DataFrame. Should contain fiscal_year, fiscal_period, and ideally
      BeginningDate/EndingDate to derive coverage by day.
    - sales_history: historical sales with columns {store_number, fiscal_year, fiscal_period,
      total_net_sales} or a pre-aggregated column named sales.
    - final_fc: final forecasts already chosen per store/period (pooled vs peer already applied).
      Must contain {store_number, fiscal_year, fiscal_period} and one forecast column among
      {sales_forecast, our_forecast, forecast}. This forecast is treated as sales_forecast.

    Optional gating
    - allowed_hm_keys: if provided, only these Heatmap store ids are kept.
    - excluded_hm_keys: if provided, these Heatmap store ids are removed.

    Returns
    - fc_hours: store-level hours forecast with at least
        {hm_key, fiscal_year, fiscal_period, sales_forecast, hours_forecast}
    - fc_dept: department-level hours forecast with at least
        {hm_key, fiscal_year, fiscal_period, department, hours_forecast_dept}

    Concept
    - Map labor rows to fiscal periods and compute total_hours per store/period and per-department.
    - Learn hours-per-sales ratios per store (median), with chain-level fallback if missing.
    - Learn department shares per store (median dept_hours / store_hours), with chain-level fallback
      and per-store normalization (shares sum to 1).
    - Apply ratios to final sales forecasts to produce store-level hours, then split by department
      using the learned shares.
    """

    fy, fp = "fiscal_year", "fiscal_period"

    # --- Key mapping from geo_stores ---
    key_to_hm = _build_key_map(geo_stores)

    # --- Normalize calendar and build daily coverage if possible ---
    cal_df = _ensure_calendar(calendar)
    cal_daily = None
    if {"BeginningDate", "EndingDate"}.issubset(cal_df.columns):
        cal_daily = (
            cal_df.assign(
                event_date=lambda d: d.apply(
                    lambda r: pd.date_range(r["BeginningDate"], r["EndingDate"], freq="D"), axis=1
                )
            )
            .explode("event_date")[
                ["event_date", fy, fp, "days_in_period"]
            ]
        )
        cal_daily["event_date"] = pd.to_datetime(cal_daily["event_date"], errors="coerce").dt.normalize()

    # --- Normalize labor rows ---
    hr = labor.copy()
    # event_date detection
    for c in ("event_date", "date", "work_date", "shift_date"):
        if c in hr.columns:
            hr["event_date"] = pd.to_datetime(hr[c], errors="coerce").dt.normalize()
            break
    if "event_date" not in hr.columns:
        raise RuntimeError("labor needs a date column (event_date/date/work_date/shift_date).")
    # department
    if "department" not in hr.columns and "dept" in hr.columns:
        hr = hr.rename(columns={"dept": "department"})
    if "department" not in hr.columns:
        hr["department"] = "Unknown"
    # store id detection -> unify to store_number
    store_col = next(
        (c for c in (
            "Labor_Store_Number", "store_number", "Store_Number", "store", "storeid", "store_id", "Store"
        ) if c in hr.columns),
        None,
    )
    if store_col is None:
        raise RuntimeError("labor needs a store id column (e.g., Store_Number or store_number).")
    hr["store_number"] = hr[store_col].astype(str).str.strip()
    # hours column
    if "total_hours" in hr.columns:
        hr["hours"] = pd.to_numeric(hr["total_hours"], errors="coerce")
    else:
        for c in ("normal_hours", "ot_hours", "dt_hours"):
            if c in hr.columns:
                hr[c] = pd.to_numeric(hr[c], errors="coerce")
        hr["hours"] = hr.filter(items=[c for c in ["normal_hours", "ot_hours", "dt_hours"] if c in hr.columns]).sum(axis=1)
        hr["hours"] = pd.to_numeric(hr["hours"], errors="coerce").fillna(0.0)

    # Map labor to fiscal
    if cal_daily is not None:
        hrs = hr.merge(cal_daily, on="event_date", how="left")
        if not {fy, fp}.issubset(hrs.columns):
            raise RuntimeError("Calendar mapping failed; fiscal columns missing.")
    else:
        if not {fy, fp}.issubset(hr.columns):
            raise RuntimeError("No daily calendar to map dates, and labor lacks fiscal_year/fiscal_period.")
        hrs = hr.copy()
        hrs["days_in_period"] = np.nan

    # Coverage by store/period
    if cal_daily is not None:
        cov = (
            hrs.groupby(["store_number", fy, fp], as_index=False)
            .agg(coverage_days=("event_date", "nunique"), days_in_period=("days_in_period", "first"))
        )
        cov["coverage_pct_active"] = (
            (pd.to_numeric(cov["coverage_days"], errors="coerce") / pd.to_numeric(cov["days_in_period"], errors="coerce"))
            .replace([np.inf, -np.inf], np.nan)
            .clip(0, 1)
            .fillna(0)
        )
    else:
        cov = (
            hrs.groupby(["store_number", fy, fp], as_index=False)
            .agg(coverage_days=("store_number", "size"))
            .assign(days_in_period=np.nan, coverage_pct_active=np.nan)
        )

    # Aggregate total hours
    labor_period_store_total = (
        hrs.groupby(["store_number", fy, fp], as_index=False)["hours"].sum()
        .rename(columns={"hours": "total_hours"})
        .merge(cov[["store_number", fy, fp, "coverage_pct_active"]], on=["store_number", fy, fp], how="left")
    )
    labor_period_store_dept = (
        hrs.groupby(["store_number", fy, fp, "department"], as_index=False)["hours"].sum()
        .rename(columns={"hours": "total_hours"})
    )

    # Attach hm_key via geo mapping + filters
    tot_hm = _attach_hm_key(labor_period_store_total, "store_number", key_to_hm)
    dept_hm = _attach_hm_key(labor_period_store_dept, "store_number", key_to_hm)
    if allowed_hm_keys is not None:
        tot_hm = tot_hm[tot_hm["hm_key"].isin(allowed_hm_keys)]
        dept_hm = dept_hm[dept_hm["hm_key"].isin(allowed_hm_keys)]
    if excluded_hm_keys is not None and len(excluded_hm_keys) > 0:
        tot_hm = tot_hm[~tot_hm["hm_key"].isin(excluded_hm_keys)]
        dept_hm = dept_hm[~dept_hm["hm_key"].isin(excluded_hm_keys)]

    # ----------------- Sales history -----------------
    sh = sales_history.copy()
    sh["store_number"] = sh["store_number"].astype(str).str.strip()
    if "sales" not in sh.columns:
        if "total_net_sales" in sh.columns:
            sh = sh.rename(columns={"total_net_sales": "sales"})
        else:
            raise KeyError("sales_history must have either 'sales' or 'total_net_sales' column")
    sh[fy] = pd.to_numeric(sh[fy], errors="coerce").astype("Int64")
    sh[fp] = pd.to_numeric(sh[fp], errors="coerce").astype("Int64")
    # If there are duplicates, aggregate
    sh = (
        sh.groupby(["store_number", fy, fp], as_index=False)["sales"].sum()
    )
    sales_hm = _attach_hm_key(sh, "store_number", key_to_hm)
    if allowed_hm_keys is not None:
        sales_hm = sales_hm[sales_hm["hm_key"].isin(allowed_hm_keys)]
    if excluded_hm_keys is not None and len(excluded_hm_keys) > 0:
        sales_hm = sales_hm[~sales_hm["hm_key"].isin(excluded_hm_keys)]

    # ----------------- Learn ratios (hours per sales) -----------------
    join_train = tot_hm.merge(
        sales_hm[["hm_key", fy, fp, "sales"]],
        on=["hm_key", fy, fp],
        how="inner",
    )
    use_coverage = "coverage_pct_active" in join_train.columns
    if use_coverage:
        jt = join_train[pd.to_numeric(join_train["coverage_pct_active"], errors="coerce") >= 0.90]
        if jt.empty:
            jt = join_train.copy()
    else:
        jt = join_train.copy()
    jt = jt[pd.to_numeric(jt["sales"], errors="coerce") > 0]
    jt["hrs_per_sales"] = pd.to_numeric(jt["total_hours"], errors="coerce") / pd.to_numeric(jt["sales"], errors="coerce")

    ratio_store = (
        jt.groupby("hm_key")["hrs_per_sales"].median().rename("ratio_store").reset_index()
    )
    if len(jt):
        ratio_chain = float(pd.to_numeric(jt["hrs_per_sales"], errors="coerce").median())
        if not np.isfinite(ratio_chain) or ratio_chain <= 0:
            s_sales = float(pd.to_numeric(jt["sales"], errors="coerce").sum())
            s_hours = float(pd.to_numeric(jt["total_hours"], errors="coerce").sum())
            ratio_chain = float(s_hours / s_sales) if s_sales > 0 else 0.0
    else:
        ratio_chain = 0.0

    # ----------------- Dept shares (median dept_hours / store_hours) -----------------
    tot_for_merge = tot_hm[["hm_key", fy, fp, "total_hours"]].rename(columns={"total_hours": "total_hours_store"})
    dept_join = dept_hm.merge(tot_for_merge, on=["hm_key", fy, fp], how="left")
    if use_coverage and "coverage_pct_active" in tot_hm.columns:
        dept_join = dept_join.merge(
            tot_hm[["hm_key", fy, fp, "coverage_pct_active"]],
            on=["hm_key", fy, fp],
            how="left",
        )
        dj = dept_join[pd.to_numeric(dept_join["coverage_pct_active"], errors="coerce") >= 0.90]
        if dj.empty:
            dj = dept_join.copy()
    else:
        dj = dept_join.copy()
    dj = dj[(pd.to_numeric(dj["total_hours_store"], errors="coerce") > 0) & (pd.to_numeric(dj["total_hours"], errors="coerce") >= 0)]
    dj["share"] = pd.to_numeric(dj["total_hours"], errors="coerce") / pd.to_numeric(dj["total_hours_store"], errors="coerce")

    store_share = dj.groupby(["hm_key", "department"]) ["share"].median().reset_index()
    chain_share = dj.groupby("department")["share"].median().rename("share_chain").reset_index()
    if chain_share.empty:
        chain_share = pd.DataFrame({"department": ["Unknown"], "share_chain": [1.0]})

    stores = pd.DataFrame({"hm_key": sorted(tot_hm["hm_key"].unique())})
    depts = pd.DataFrame({"department": sorted(chain_share["department"].unique())})
    grid = (
        stores.assign(_k=1)
        .merge(depts.assign(_k=1), on="_k", how="outer")
        .drop(columns="_k")
        .merge(store_share, on=["hm_key", "department"], how="left")
        .merge(chain_share, on="department", how="left")
    )

    def _norm(v: pd.Series) -> pd.Series:
        s = pd.Series(v, dtype=float).clip(lower=0).fillna(0.0)
        t = s.sum()
        return (s / t) if t > 0 else pd.Series([1.0 / len(s)] * len(s), index=s.index)

    share_final = (
        grid.groupby("hm_key", group_keys=False)
        .apply(lambda g: g.assign(share_final=_norm(g["share"].fillna(g["share_chain"])) .values))
        .reset_index(drop=True)
    )

    # ----------------- Consume FINAL sales forecasts -----------------
    fc = final_fc.copy()
    fc["store_number"] = fc["store_number"].astype(str).str.strip()
    # Standardize forecast column to sales_forecast
    if "sales_forecast" in fc.columns:
        pass
    elif "our_forecast" in fc.columns:
        fc = fc.rename(columns={"our_forecast": "sales_forecast"})
    elif "forecast" in fc.columns:
        fc = fc.rename(columns={"forecast": "sales_forecast"})
    else:
        raise KeyError("final_fc must include one of: sales_forecast, our_forecast, forecast")
    fc[fy] = pd.to_numeric(fc[fy], errors="coerce").astype("Int64")
    fc[fp] = pd.to_numeric(fc[fp], errors="coerce").astype("Int64")
    fc["sales_forecast"] = pd.to_numeric(fc["sales_forecast"], errors="coerce")

    fc_hm = _attach_hm_key(fc, "store_number", key_to_hm)
    if allowed_hm_keys is not None:
        fc_hm = fc_hm[fc_hm["hm_key"].isin(allowed_hm_keys)]
    if excluded_hm_keys is not None and len(excluded_hm_keys) > 0:
        fc_hm = fc_hm[~fc_hm["hm_key"].isin(excluded_hm_keys)]

    # ----------------- Convert sales → hours (store + dept) -----------------
    fc_hours = fc_hm.merge(ratio_store, on="hm_key", how="left")
    fc_hours["ratio_final"] = pd.to_numeric(fc_hours.get("ratio_store"), errors="coerce").fillna(ratio_chain)
    fc_hours["ratio_final"] = fc_hours["ratio_final"].clip(lower=0).fillna(0.0)
    fc_hours["sales_forecast"] = pd.to_numeric(fc_hours["sales_forecast"], errors="coerce").fillna(0.0)
    fc_hours["hours_forecast"] = fc_hours["sales_forecast"] * fc_hours["ratio_final"]

    fc_dept = (
        fc_hours[["hm_key", fy, fp, "hours_forecast"]]
        .merge(share_final[["hm_key", "department", "share_final"]], on="hm_key", how="left")
    )
    fc_dept["department"] = fc_dept["department"].fillna("Unknown")
    fc_dept["share_final"] = pd.to_numeric(fc_dept["share_final"], errors="coerce").fillna(1.0)
    fc_dept["hours_forecast_dept"] = fc_dept["hours_forecast"] * fc_dept["share_final"]

    # Minimal required columns for outputs
    fc_hours_out = fc_hours[["hm_key", fy, fp, "sales_forecast", "hours_forecast"]].copy()
    fc_dept_out = fc_dept[["hm_key", fy, fp, "department", "hours_forecast_dept"]].copy()

    return fc_hours_out, fc_dept_out


def build_labor_cost_forecast(
    final_fc: pd.DataFrame,
    store_state_source: Optional[pd.DataFrame] = None,
    cost_source: Optional[pd.DataFrame] = None,
    ratio_df: Optional[pd.DataFrame] = None,
    allowed_store_numbers: Optional[Set[str]] = None,
    excluded_store_numbers: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """Compute labor COST forecasts from final sales forecasts.

    Inputs
    - final_fc: final per-store forecasts with columns {store_number, fiscal_year, fiscal_period} and one
      forecast column among {sales_forecast, our_forecast, forecast}. Optional column {source} is preserved.
    - store_state_source: optional DataFrame to derive a per-store State column. Two accepted shapes:
        1) Columns {Heatmap_Store_Number, State}  (e.g., geo_stores)
        2) Columns {store_number, <state-like-column>} (e.g., c_heatmap). The first state-like column is used.
    - cost_source: optional historical table to compute per-store labor_cost_ratio when ratio_df is not provided.
        Must contain columns {store_number, total_labor_cost, total_net_sales}.
    - ratio_df: optional precomputed per-store ratios with columns {store_number, labor_cost_ratio}.
      If provided, cost_source is ignored.
    - allowed_store_numbers: optional set of store_number strings to include; others are dropped.
    - excluded_store_numbers: optional set of store_number strings to exclude after inclusion filter.

    Returns
    - DataFrame with columns at least:
        {store_number, fiscal_year, fiscal_period, State, sales_forecast, labor_cost_ratio, pred_total_labor_cost}
      If final_fc included a 'source' column, it is preserved.

    Notes
    - This function does not train models; it only consumes the final forecasts and historical cost/sales to
      derive per-store ratios and apply them to the forecasts.
    """

    fy, fp = "fiscal_year", "fiscal_period"

    # Normalize final forecasts and standardize forecast column name
    fc = final_fc.copy()
    fc["store_number"] = fc["store_number"].astype(str).str.strip()
    if "sales_forecast" not in fc.columns:
        if "our_forecast" in fc.columns:
            fc = fc.rename(columns={"our_forecast": "sales_forecast"})
        elif "forecast" in fc.columns:
            fc = fc.rename(columns={"forecast": "sales_forecast"})
        else:
            raise KeyError("final_fc must include one of: sales_forecast, our_forecast, forecast")
    fc[fy] = pd.to_numeric(fc[fy], errors="coerce").astype("Int64")
    fc[fp] = pd.to_numeric(fc[fp], errors="coerce").astype("Int64")
    fc["sales_forecast"] = pd.to_numeric(fc["sales_forecast"], errors="coerce")

    # Optional inclusion/exclusion by store_number
    if allowed_store_numbers is not None:
        keep = {str(x).strip() for x in allowed_store_numbers}
        fc = fc[fc["store_number"].isin(keep)]
    if excluded_store_numbers is not None and len(excluded_store_numbers) > 0:
        drop = {str(x).strip() for x in excluded_store_numbers}
        fc = fc[~fc["store_number"].isin(drop)]

    # Build State map
    state_map = None
    if store_state_source is not None and len(store_state_source) > 0:
        ss = store_state_source.copy()
        ss.columns = ss.columns.str.strip()
        if {"Heatmap_Store_Number", "State"}.issubset(ss.columns):
            g = ss[["Heatmap_Store_Number", "State"]].copy()
            g["Heatmap_Store_Number"] = g["Heatmap_Store_Number"].astype(str).str.strip()
            state_map = g.rename(columns={"Heatmap_Store_Number": "store_number"})
        elif "store_number" in ss.columns:
            cand = [c for c in ss.columns if c.lower().find("state") >= 0 and c != "store_number"]
            if cand:
                st_col = cand[0]
                g = ss[["store_number", st_col]].copy()
                g["store_number"] = g["store_number"].astype(str).str.strip()
                g[st_col] = g[st_col].astype(str).str.strip()
                # mode state per store
                state_map = (
                    g.dropna(subset=[st_col])
                    .groupby("store_number", as_index=False)[st_col]
                    .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.dropna().iloc[0])
                    .rename(columns={st_col: "State"})
                )
    if state_map is not None:
        base_fc = fc.merge(state_map, on="store_number", how="left")
        base_fc["State"] = base_fc["State"].fillna("Unknown")
    else:
        base_fc = fc.assign(State="Unknown")

    # Determine ratio per store
    if ratio_df is not None and not ratio_df.empty:
        r = ratio_df.copy()
        need = {"store_number", "labor_cost_ratio"}
        if not need.issubset(r.columns):
            raise KeyError("ratio_df must include columns: store_number, labor_cost_ratio")
        r["store_number"] = r["store_number"].astype(str).str.strip()
        r["labor_cost_ratio"] = pd.to_numeric(r["labor_cost_ratio"], errors="coerce")
    else:
        if cost_source is None:
            raise RuntimeError("Provide either ratio_df or cost_source to compute labor_cost_ratio")
        cs = cost_source.copy()
        need = {"store_number", "total_labor_cost", "total_net_sales"}
        if not need.issubset(cs.columns):
            raise KeyError("cost_source must include columns: store_number, total_labor_cost, total_net_sales")
        cs["store_number"] = cs["store_number"].astype(str).str.strip()
        cs["total_labor_cost"] = pd.to_numeric(cs["total_labor_cost"], errors="coerce")
        cs["total_net_sales"] = pd.to_numeric(cs["total_net_sales"], errors="coerce")
        r = (
            cs.groupby("store_number", as_index=False)
            .agg(cost_sum=("total_labor_cost", "sum"), sales_sum=("total_net_sales", "sum"))
        )
        r["labor_cost_ratio"] = (
            (pd.to_numeric(r["cost_sum"], errors="coerce") / pd.to_numeric(r["sales_sum"], errors="coerce"))
            .replace([np.inf, -np.inf], np.nan)
        )
        r = r[["store_number", "labor_cost_ratio"]]

    # Compute forecasted labor cost
    out = base_fc.merge(r, on="store_number", how="left")
    med_ratio = float(pd.to_numeric(out["labor_cost_ratio"], errors="coerce").median(skipna=True)) if len(out) else 0.0
    out["labor_cost_ratio"] = pd.to_numeric(out["labor_cost_ratio"], errors="coerce").fillna(med_ratio).clip(lower=0)
    out["pred_total_labor_cost"] = out["sales_forecast"] * out["labor_cost_ratio"]

    # Minimal ordered columns
    cols = ["store_number", fy, fp, "State", "sales_forecast", "labor_cost_ratio", "pred_total_labor_cost"]
    if "source" in out.columns:
        cols = ["source"] + cols
    return out[cols].copy()


__all__ = ["build_labor_forecasts", "build_labor_cost_forecast"]
