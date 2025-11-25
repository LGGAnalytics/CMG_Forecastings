import json
import os
import pickle
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


# Default, hard-coded list of stores to exclude if none is passed in via config.
# Note: normalization (string trim and leading-zero strip) is applied later.
HARDCODED_EXCLUDE_STORES = [
    364001, 364002, 364003, 364004, 364005,
    364006, 364007, 364008, 364009, 364010,
]


@dataclass
class ForecastConfig:
    """
    Configuration for pooled Ridge forecasting.

    Fields are intentionally simple so they can be JSON-serialized
    and shown in prints.
    """

    feature_list: List[str]
    target: str = "total_net_sales"
    store_col: str = "store_number"
    time_cols: Tuple[str, str, str] = ("event_date", "fiscal_year", "fiscal_period")

    # Model / data rules
    alpha: float = 5.0
    min_history_train: int = 12
    min_history_eval: int = 18
    holdout_k: int = 6

    # Column names for events
    super_bowl_col: str = "Super Bowl"

    # Global list of stores to exclude everywhere (features, train, predict)
    exclude_stores: Optional[Sequence[object]] = None

    # Logging verbosity for prints
    verbose: bool = True

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def hash_key(self) -> str:
        """
        Simple hash-like key for change detection (features + alpha + excludes).
        """
        effective_excl = (
            list(self.exclude_stores)
            if self.exclude_stores is not None and len(self.exclude_stores) > 0
            else list(HARDCODED_EXCLUDE_STORES)
        )
        key = {
            "features": list(self.feature_list),
            "alpha": float(self.alpha),
            "exclude_stores": effective_excl,
        }
        return json.dumps(key, sort_keys=True)

    def excluded_set(self) -> set:
        """
        Use configured list if provided; otherwise fall back to hard-coded defaults.
        Normalizes by stripping spaces and leading zeros.
        """
        src = self.exclude_stores if self.exclude_stores else HARDCODED_EXCLUDE_STORES
        s: set = set()
        for v in src:
            t = str(v).strip()
            s.add(t)
            s.add(t.lstrip("0"))
        return s


class FeatureBuilder:
    """
    Builds model features from period-level inputs.

    This builder prefers 'compute-if-missing' for robustness:
    - Always recomputes p13_sin, p13_cos from fiscal_period
    - Computes is_5w from calendar.days_in_period (>=35 → 1)
    - Computes m1 lags per store if missing
    - Uses existing Super Bowl column if present; otherwise can accept an events map
    - CPI_m1: if missing, uses CPI joined by (fy, fp), then lag(1); FFILL before lag if needed
    """

    def __init__(self, config: ForecastConfig):
        self.cfg = config

    # ---- Public, consolidated builder ----
    def build_feature_table(
        self,
        base_df: pd.DataFrame,
        calendar_df: pd.DataFrame,
        events: Optional[object] = None,      # DataFrame or path to Excel/CSV with Super Bowl
        adv_keep: Optional[pd.DataFrame] = None,  # (unused here but kept for API compat)
        cpi: Optional[object] = None,         # DataFrame (monthly or fiscal) or path to Excel/CSV
        exclude_stores: Optional[Sequence[object]] = None,
    ) -> pd.DataFrame:
        """
        Builds the full feature table in one go, robust to missing pieces.

        Inputs
        ------
        - base_df: period/daily table with at least:
            store_number, event_date, total_net_sales, total_advertising
        - calendar_df: fiscal calendar (BeginningDate, EndingDate, + fiscal year/period columns)
        - events: None, DataFrame, or path; will extract only 'Super Bowl' flags (0/1) by event_date
        - adv_keep: None or DataFrame with ['store_number','Advertising_keep'] (0/1)
        - cpi: None, DataFrame, or path; either monthly with Year+Jan..Dec or fiscal with
               ['fiscal_year','fiscal_period','CPI']
        - exclude_stores: optional list of stores to exclude for feature building.
                          If None, uses cfg.exclude_stores or hard-coded defaults.

        Output columns
        --------------
        - keys: store_number, event_date, fiscal_year, fiscal_period
        - target: total_net_sales
        - features: total_net_sales_m1, adv_m1, CPI_m1, p13_sin, p13_cos, is_5w, Super Bowl
        """
        cfg = self.cfg
        store_col, date_col, fy_col, fp_col = (
            cfg.store_col,
            cfg.time_cols[0],
            cfg.time_cols[1],
            cfg.time_cols[2],
        )

        df = base_df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()

        # Normalize key columns
        df[store_col] = df[store_col].astype(str).str.strip()

        # Optional: exclude specific stores
        if exclude_stores is None:
            # Use the config's normalized set (includes hard-coded defaults if none provided)
            ex = set(self.cfg.excluded_set())
        else:
            ex = {str(s).strip() for s in exclude_stores}
            ex = ex | {s.lstrip("0") for s in ex}

        if ex:
            before = len(df)
            store_norm = df[store_col].astype(str).str.strip().str.lstrip("0")
            df = df[~store_norm.isin(ex)]
            after = len(df)
            if self.cfg.verbose:
                print(
                    f"[FeatureBuilder] excluded stores: {len(ex)} | "
                    f"rows removed: {before - after}"
                )

        df[cfg.target] = pd.to_numeric(df[cfg.target], errors="coerce")
        if "total_advertising" in df.columns:
            df["total_advertising"] = (
                pd.to_numeric(df["total_advertising"], errors="coerce").fillna(0.0)
            )

        # Attach Super Bowl from events if provided
        if events is not None:
            if isinstance(events, str):
                try:
                    ev_df = pd.read_excel(events)
                except Exception:
                    ev_df = pd.read_csv(events)
            else:
                ev_df = events.copy()

            # Tolerant flag column detection (accepts "Super Bowl" or "Super_Bowl")
            def _find_sb_col(cols: List[str]) -> Optional[str]:
                for c in cols:
                    key = str(c).strip().lower().replace(" ", "")
                    if key in ("superbowl", "super_bowl"):
                        return c
                return None

            sb_col = _find_sb_col(list(ev_df.columns))

            # If events are fiscal-period keyed, map to event_date with calendar
            if "event_date" not in ev_df.columns and {fy_col, fp_col}.issubset(ev_df.columns):
                cal_map = calendar_df[[fy_col, fp_col, date_col]].copy()
                cal_map[date_col] = pd.to_datetime(
                    cal_map[date_col], errors="coerce"
                ).dt.normalize()
                ev_df = ev_df.merge(cal_map, on=[fy_col, fp_col], how="left")

            if "event_date" in ev_df.columns and sb_col is not None:
                ev_df[date_col] = pd.to_datetime(
                    ev_df["event_date"], errors="coerce"
                ).dt.normalize()
                keep_cols = [date_col, sb_col]

                # Keep store-specific flag if present; else date-level flag
                if cfg.store_col in ev_df.columns:
                    ev_df[cfg.store_col] = ev_df[cfg.store_col].astype(str).str.strip()
                    keep_cols.append(cfg.store_col)

                ev_df = ev_df[keep_cols].rename(columns={sb_col: "Super Bowl"})
                ev_df["Super Bowl"] = (
                    pd.to_numeric(ev_df["Super Bowl"], errors="coerce")
                    .fillna(0)
                    .astype(int)
                    .clip(0, 1)
                )

                if cfg.store_col in ev_df.columns:
                    df = df.merge(ev_df, on=[date_col, cfg.store_col], how="left")
                else:
                    df = df.merge(ev_df, on=[date_col], how="left")

        # If still missing, default 0
        if "Super Bowl" not in df.columns:
            df["Super Bowl"] = 0

        # Ensure alias column exists if feature_list uses alternate spelling
        if (
            "Super_Bowl" in cfg.feature_list
            and "Super_Bowl" not in df.columns
            and "Super Bowl" in df.columns
        ):
            df["Super_Bowl"] = df["Super Bowl"]
        if (
            "Super Bowl" in cfg.feature_list
            and "Super Bowl" not in df.columns
            and "Super_Bowl" in df.columns
        ):
            df["Super Bowl"] = df["Super_Bowl"]

        # Calendar prep
        cal = self._ensure_calendar(calendar_df)
        cal = self._standardize_fiscal_cols(cal)

        # Ensure an is_5w flag exists on the calendar
        if "is_5w" not in cal.columns:
            if "weeks_in_period" in cal.columns:
                cal["is_5w"] = (
                    pd.to_numeric(cal["weeks_in_period"], errors="coerce") == 5
                ).astype(int)
            elif "days_in_period" in cal.columns:
                cal["is_5w"] = (
                    pd.to_numeric(cal["days_in_period"], errors="coerce") >= 35
                ).astype(int)
            else:
                cal["is_5w"] = 0
                print(
                    "[FeatureBuilder] WARNING: calendar lacks days_in_period/weeks_in_period; "
                    "is_5w defaulted to 0."
                )

        # Ensure fiscal mapping on df
        if not {fy_col, fp_col}.issubset(df.columns):
            # Map by date; carry is_5w (and days_in_period if present but not required)
            carry_cols = [c for c in [fy_col, fp_col, "is_5w"] if c in cal.columns]
            cal_daily = (
                cal.assign(
                    event_date=lambda r: cal.apply(
                        lambda row: pd.date_range(
                            row.get("BeginningDate"),
                            row.get("EndingDate"),
                            freq="D",
                        )
                        if ("BeginningDate" in cal.columns and "EndingDate" in cal.columns)
                        else pd.to_datetime([]),
                        axis=1,
                    )
                )
                .explode("event_date")[["event_date"] + carry_cols]
            )
            df = df.merge(cal_daily, on="event_date", how="left")
        else:
            df = df.merge(
                cal[[fy_col, fp_col, "is_5w"]].drop_duplicates(),
                on=[fy_col, fp_col],
                how="left",
            )

        df = self._compute_seasonality(df)

        # is_5w already merged from calendar; if still missing, default 0
        if "is_5w" not in df.columns:
            df["is_5w"] = 0

        # CPI attach and lag
        if cpi is not None and "CPI_m1" not in df.columns:
            cpi_df = self._prepare_cpi(cpi)
            # map CPI to fiscal period via calendar if needed
            cpi_fiscal = self._map_cpi_to_fiscal(cal, cpi_df)
            df = df.merge(
                cpi_fiscal[[fy_col, fp_col, "CPI"]], on=[fy_col, fp_col], how="left"
            )

        df = self._ensure_cpi_m1(df, None)

        # Lags: target and advertising
        df = df.sort_values([store_col, date_col])

        if "total_net_sales_m1" not in df.columns:
            df["total_net_sales_m1"] = df.groupby(store_col)[cfg.target].shift(1)

        # adv_m1 from total_advertising
        if "adv_m1" not in df.columns:
            if "total_advertising" in df.columns:
                df["adv_m1"] = df.groupby(store_col)["total_advertising"].shift(1)
            else:
                df["adv_m1"] = 0.0
        df["adv_m1"] = pd.to_numeric(df["adv_m1"], errors="coerce").fillna(0.0)

        # Final selection
        features = list(cfg.feature_list)
        keys = [store_col, date_col, fy_col, fp_col]
        needed = keys + [cfg.target] + features
        present = [c for c in needed if c in df.columns]
        out = df[present].copy()

        # Drop rows missing critical lags/target; coerce other features to numeric and fill
        before = len(out)
        req = [cfg.target]
        if "total_net_sales_m1" in out.columns:
            req.append("total_net_sales_m1")
        if "adv_m1" in out.columns:
            req.append("adv_m1")

        out = out.dropna(subset=req)

        # Make remaining features numeric and fill NA to 0.0 (e.g., CPI_m1)
        numeric_feats = [f for f in features if f in out.columns]
        if numeric_feats:
            out[numeric_feats] = out[numeric_feats].apply(
                pd.to_numeric, errors="coerce"
            ).fillna(0.0)

        after = len(out)
        if self.cfg.verbose:
            print(
                f"[FeatureBuilder] final features rows: {after} "
                f"(dropped {before - after})"
            )

        return out

    # ---- Helpers used by the consolidated builder ----
    def _standardize_fiscal_cols(self, cal: pd.DataFrame) -> pd.DataFrame:
        """
        Try to standardize alternative names to expected names in config.
        """
        fy_col, fp_col = self.cfg.time_cols[1], self.cfg.time_cols[2]
        lower = {c.lower(): c for c in cal.columns}
        fy_src = (
            lower.get("fiscal_year")
            or lower.get("year")
            or lower.get("fy")
            or lower.get("fiscalyear")
        )
        fp_src = (
            lower.get("fiscal_period")
            or lower.get("period")
            or lower.get("fp")
            or lower.get("fiscalperiod")
        )
        if fy_src is None or fp_src is None:
            return cal
        if fy_src != fy_col or fp_src != fp_col:
            cal = cal.rename(columns={fy_src: fy_col, fp_src: fp_col})
        return cal

    def _prepare_cpi(self, cpi: object) -> pd.DataFrame:
        """
        Accept path or DataFrame; returns monthly CPI with columns ['cal_month','CPI']
        or fiscal CPI with ['fiscal_year','fiscal_period','CPI'].
        """
        if isinstance(cpi, str):
            try:
                df = pd.read_excel(cpi)
            except Exception:
                df = pd.read_csv(cpi)
        else:
            df = cpi.copy()

        df_cols_lower = [c.lower() for c in df.columns]

        # Case 1: monthly wide with Year + Jan..Dec
        if "year" in df_cols_lower and any(
            m.lower() in df_cols_lower
            for m in ["jan", "feb", "mar", "apr", "may", "jun",
                      "jul", "aug", "sep", "oct", "nov", "dec"]
        ):
            months = [
                "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
            ]
            long_df = df.melt("Year", months, var_name="Month", value_name="CPI")
            month_map = {m: i + 1 for i, m in enumerate(months)}
            long_df["cal_month"] = pd.to_datetime(
                long_df["Year"].astype(int).astype(str)
                + "-"
                + long_df["Month"].map(month_map).astype(str)
            ).dt.to_period("M").dt.to_timestamp("M")
            long_df["CPI"] = pd.to_numeric(long_df["CPI"], errors="coerce").ffill()
            return long_df[["cal_month", "CPI"]]

        # Case 2: already fiscal
        if {self.cfg.time_cols[1], self.cfg.time_cols[2], "CPI"}.issubset(df.columns):
            return df

        # Fallback: empty
        return pd.DataFrame({"cal_month": [], "CPI": []})

    def _map_cpi_to_fiscal(self, cal: pd.DataFrame, cpi_df: pd.DataFrame) -> pd.DataFrame:
        """
        If CPI is monthly with cal_month, map via calendar month end;
        if already fiscal, return as is.
        """
        fy_col, fp_col = self.cfg.time_cols[1], self.cfg.time_cols[2]
        if {fy_col, fp_col, "CPI"}.issubset(cpi_df.columns):
            return cpi_df

        cal2 = cal.copy()
        cal2["EndingDate"] = pd.to_datetime(cal2["EndingDate"], errors="coerce")
        cal2["cal_month"] = cal2["EndingDate"].dt.to_period("M").dt.to_timestamp("M")
        out = cal2[[fy_col, fp_col, "cal_month"]].merge(
            cpi_df, on="cal_month", how="left"
        )
        out = out[[fy_col, fp_col, "CPI"]].drop_duplicates([fy_col, fp_col])
        return out

    def _ensure_calendar(self, cal_df: pd.DataFrame) -> pd.DataFrame:
        cal = cal_df.copy()
        if "BeginningDate" in cal.columns:
            cal["BeginningDate"] = pd.to_datetime(
                cal["BeginningDate"], errors="coerce"
            )
        if "EndingDate" in cal.columns:
            cal["EndingDate"] = pd.to_datetime(cal["EndingDate"], errors="coerce")
        if "days_in_period" not in cal.columns and {
            "BeginningDate",
            "EndingDate",
        }.issubset(cal.columns):
            cal["days_in_period"] = (
                cal["EndingDate"] - cal["BeginningDate"]
            ).dt.days + 1
        return cal

    def _compute_seasonality(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute 13-period seasonality sin/cos from fiscal_period.
        """
        fy_col, fp_col = self.cfg.time_cols[1], self.cfg.time_cols[2]
        out = df.copy()

        # Drop any existing p13 columns
        out = out.drop(columns=["p13_sin", "p13_cos"], errors="ignore")

        fp = pd.to_numeric(out[fp_col], errors="coerce")
        out["p13_sin"] = np.sin(2 * np.pi * fp / 13.0)
        out["p13_cos"] = np.cos(2 * np.pi * fp / 13.0)
        return out

    def _ensure_cpi_m1(
        self, df: pd.DataFrame, cpi_df: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        out = df.copy()
        fy_col, fp_col = self.cfg.time_cols[1], self.cfg.time_cols[2]

        if "CPI_m1" in out.columns:
            return out

        # Try from CPI in out
        if "CPI" in out.columns:
            out["CPI"] = pd.to_numeric(out["CPI"], errors="coerce")
        elif cpi_df is not None and {
            fy_col,
            fp_col,
            "CPI",
        }.issubset(cpi_df.columns):
            cpi = cpi_df[[fy_col, fp_col, "CPI"]].copy()
            out = out.merge(cpi, on=[fy_col, fp_col], how="left")
        else:
            out["CPI"] = np.nan

        out.sort_values([self.cfg.store_col, fy_col, fp_col], inplace=True)
        out["CPI"] = out.groupby(self.cfg.store_col)["CPI"].ffill()
        out["CPI_m1"] = out.groupby(self.cfg.store_col)["CPI"].shift(1)
        out["CPI_m1"] = pd.to_numeric(out["CPI_m1"], errors="coerce").fillna(0.0)
        return out

    def _ensure_super_bowl(
        self,
        df: pd.DataFrame,
        cal: pd.DataFrame,
        events_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Minimal helper for next-period forecast rows.
        """
        out = df.copy()
        col = self.cfg.super_bowl_col
        if col in out.columns:
            out[col] = out[col].fillna(0).astype(int).clip(0, 1)
            return out

        # If not present, we could map from events_df if needed.
        # For now, set 0 by default.
        out[col] = 0
        return out

    def build_forecast_set(
        self,
        history_df: pd.DataFrame,
        calendar_df: pd.DataFrame,
        cpi_df: Optional[pd.DataFrame] = None,
        events_df: Optional[pd.DataFrame] = None,
        flags_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Construct one next-period row per store using last observed values
        and next-period calendar.

        Assumes history_df already contains the necessary columns and spans
        at least 1 row per store.
        """
        cfg = self.cfg
        store_col, date_col, fy_col, fp_col = (
            cfg.store_col,
            cfg.time_cols[0],
            cfg.time_cols[1],
            cfg.time_cols[2],
        )

        df = history_df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values([store_col, date_col])

        # Exclude globally configured stores for forecast too
        ex = self.cfg.excluded_set()
        if ex:
            store_norm = df[store_col].astype(str).str.strip().str.lstrip("0")
            df = df[~store_norm.isin(ex)]

        # last observed per store
        last = df.groupby(store_col).tail(1).copy()

        # derive next period (fy, fp) from current fy/fp
        next_fy = last[fy_col].astype(int)
        next_fp = last[fp_col].astype(int) + 1
        rollover = next_fp > 13
        next_fp = np.where(rollover, 1, next_fp)
        next_fy = np.where(rollover, next_fy + 1, next_fy)

        next_rows = last[[store_col]].copy()
        next_rows[fy_col] = next_fy
        next_rows[fp_col] = next_fp

        cal = self._ensure_calendar(calendar_df)
        cal = self._standardize_fiscal_cols(cal)

        # Ensure is_5w available
        if "is_5w" not in cal.columns:
            if "weeks_in_period" in cal.columns:
                cal["is_5w"] = (
                    pd.to_numeric(cal["weeks_in_period"], errors="coerce") == 5
                ).astype(int)
            elif "days_in_period" in cal.columns:
                cal["is_5w"] = (
                    pd.to_numeric(cal["days_in_period"], errors="coerce") >= 35
                ).astype(int)
            else:
                cal["is_5w"] = 0
                print(
                    "[FeatureBuilder] WARNING: calendar lacks days_in_period/weeks_in_period; "
                    "is_5w defaulted to 0."
                )

        # Compute p13 seasonality for next period
        next_rows = self._compute_seasonality(next_rows)

        # Merge is_5w for next period
        next_rows = next_rows.merge(
            cal[[fy_col, fp_col, "is_5w"]].drop_duplicates(),
            on=[fy_col, fp_col],
            how="left",
        )
        if "is_5w" not in next_rows.columns:
            next_rows["is_5w"] = 0

        # Super Bowl
        next_rows = self._ensure_super_bowl(next_rows, cal, events_df)

        # m1 features from last observed period
        for src_col, lag_col in [
            (cfg.target, "total_net_sales_m1"),
            ("adv", "adv_m1"),
            ("CPI", "CPI_m1"),
        ]:
            if lag_col in last.columns:
                next_rows[lag_col] = last[lag_col].values
            elif src_col in last.columns:
                next_rows[lag_col] = pd.to_numeric(
                    last[src_col], errors="coerce"
                ).values
            else:
                next_rows[lag_col] = 0.0

        # adv_m1 as numeric
        next_rows["adv_m1"] = pd.to_numeric(
            next_rows["adv_m1"], errors="coerce"
        ).fillna(0.0)

        # event_date for next period can be the beginning of the fiscal period from calendar (optional)
        if {"BeginningDate", fy_col, fp_col}.issubset(cal.columns):
            next_rows = next_rows.merge(
                cal[[fy_col, fp_col, "BeginningDate"]].rename(
                    columns={"BeginningDate": date_col}
                ),
                on=[fy_col, fp_col],
                how="left",
            )

        # Keep only needed columns
        needed = [cfg.store_col, fy_col, fp_col, date_col] + cfg.feature_list
        for f_name in cfg.feature_list:
            if f_name not in next_rows.columns:
                next_rows[f_name] = 0.0

        out = next_rows[[c for c in needed if c in next_rows.columns]].copy()

        if self.cfg.verbose:
            print(f"[FeatureBuilder] next-period design rows: {len(out)}")

        return out


class PooledRidgeForecaster:
    """
    Pooled Ridge forecaster with store dummies and simple time split for evaluation.

    - train(): fits on pooled stores with ≥ min_history_train; reports holdout WAPE using
               last k periods for stores ≥ min_history_eval
    - predict(): builds dummy-aligned design and returns forecasts with only the
                 requested key columns
    """

    def __init__(self, config: ForecastConfig):
        self.cfg = config
        self.model: Optional[Ridge] = None
        self.x_columns_: Optional[List[str]] = None
        self.metrics_: Dict[str, float] = {}

    def _build_design(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        cfg = self.cfg
        X = df[cfg.feature_list + [cfg.store_col]].copy()
        X = pd.get_dummies(X, columns=[cfg.store_col], drop_first=True)
        y = pd.to_numeric(df[cfg.target], errors="coerce")
        return X, y

    def train(self, df_features: pd.DataFrame) -> Dict[str, float]:
        cfg = self.cfg
        store_col, date_col = cfg.store_col, cfg.time_cols[0]
        fy_col, fp_col = cfg.time_cols[1], cfg.time_cols[2]

        df = df_features.copy()

        # Exclude stores globally before training
        ex = cfg.excluded_set()
        if ex:
            df[store_col] = df[store_col].astype(str).str.strip()
            store_norm = df[store_col].str.lstrip("0")
            df = df[~store_norm.isin(ex)]

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values([store_col, date_col])

        counts = df.groupby(store_col)[date_col].size()
        train_stores = counts[counts >= cfg.min_history_train].index
        eval_stores = counts[counts >= cfg.min_history_eval].index

        df_train_all = df[df[store_col].isin(train_stores)].copy()

        # Build evaluation split per store (last k rows)
        holdout_frames: List[pd.DataFrame] = []
        train_frames: List[pd.DataFrame] = []

        for st, g in df_train_all.groupby(store_col):
            g = g.sort_values(date_col)
            if st in set(eval_stores) and cfg.holdout_k > 0:
                holdout_frames.append(g.tail(cfg.holdout_k))
                train_frames.append(g.iloc[:-cfg.holdout_k])
            else:
                train_frames.append(g)

        df_train = (
            pd.concat(train_frames, ignore_index=True) if train_frames else pd.DataFrame()
        )
        df_valid = (
            pd.concat(holdout_frames, ignore_index=True)
            if holdout_frames
            else pd.DataFrame()
        )

        if df_train.empty:
            raise ValueError("No training rows after applying min_history_train filter.")

        X_train, y_train = self._build_design(df_train)
        X_valid, y_valid = (None, None)
        if not df_valid.empty:
            X_valid, y_valid = self._build_design(df_valid)
            # align valid columns to train
            X_valid = X_valid.reindex(columns=X_train.columns, fill_value=0)

        # Fit model
        model = Ridge(alpha=cfg.alpha, random_state=0)
        model.fit(X_train, y_train)

        # Evaluate WAPE on holdout if available
        metrics: Dict[str, float] = {
            "stores_train": float(len(train_stores)),
            "stores_eval": float(len(eval_stores)),
            "rows_train": float(len(df_train)),
            "rows_eval": float(len(df_valid)),
            "alpha": float(cfg.alpha),
        }

        if y_valid is not None and len(y_valid) > 0:
            preds = model.predict(X_valid)
            denom = np.sum(np.abs(y_valid.values))
            wape = (
                float(np.sum(np.abs(y_valid.values - preds)) / denom)
                if denom > 0
                else float("nan")
            )
            metrics["wape_holdout"] = wape
            if self.cfg.verbose:
                print(
                    f"[Train] holdout WAPE: {wape:.4f} | rows_eval={len(df_valid)}"
                )
        else:
            if self.cfg.verbose:
                print("[Train] no evaluation holdout (not enough history per store).")

        # Refit on all eligible rows for final model
        X_all, y_all = self._build_design(df_train_all)
        model.fit(X_all, y_all)

        self.model = model
        self.x_columns_ = list(X_all.columns)
        self.metrics_ = metrics

        if self.cfg.verbose:
            print(
                "[Train] stores_train="
                f"{int(metrics['stores_train'])} | stores_eval={int(metrics['stores_eval'])} | "
                f"rows_train={int(metrics['rows_train'])} | rows_eval={int(metrics['rows_eval'])} | "
                f"alpha={cfg.alpha}"
            )

        return metrics

    def predict(self, df_future: pd.DataFrame) -> pd.DataFrame:
        if self.model is None or self.x_columns_ is None:
            raise RuntimeError("Model not trained/loaded.")

        cfg = self.cfg
        store_col, fy_col, fp_col = cfg.store_col, cfg.time_cols[1], cfg.time_cols[2]

        df_future = df_future.copy()

        # Exclude stores globally before prediction
        ex = cfg.excluded_set()
        if ex:
            df_future[store_col] = df_future[store_col].astype(str).str.strip()
            store_norm = df_future[store_col].str.lstrip("0")
            df_future = df_future[~store_norm.isin(ex)]

        # Ensure feature columns present and finite
        for col in cfg.feature_list:
            if col not in df_future.columns:
                df_future[col] = 0.0

        df_future[cfg.feature_list] = (
            df_future[cfg.feature_list]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
        )

        Xf, _ = self._build_design(df_future.assign(**{cfg.target: 0}))
        Xf = Xf.reindex(columns=self.x_columns_, fill_value=0)

        preds = self.model.predict(Xf)
        out = df_future[[store_col, fy_col, fp_col]].copy()
        out["forecast"] = preds.astype(float)
        return out

    def predict_h(
        self,
        fb: FeatureBuilder,
        history_df: pd.DataFrame,
        calendar_df: pd.DataFrame,
        events_df: Optional[pd.DataFrame] = None,
        flags_df: Optional[pd.DataFrame] = None,
        cpi_df: Optional[pd.DataFrame] = None,
        H: int = 6,
    ) -> pd.DataFrame:
        """
        Multi-step forecasting: recursively roll H periods ahead.
        """
        if self.model is None or self.x_columns_ is None:
            raise RuntimeError("Model not trained/loaded.")

        hist = history_df.copy()
        outputs: List[pd.DataFrame] = []

        for h in range(1, int(H) + 1):
            fut = fb.build_forecast_set(
                history_df=hist,
                calendar_df=calendar_df,
                cpi_df=cpi_df,
                events_df=events_df,
                flags_df=flags_df,
            )

            # Ensure feature columns present and finite
            for col in fb.cfg.feature_list:
                if col not in fut.columns:
                    fut[col] = 0.0
                fut[col] = pd.to_numeric(fut[col], errors="coerce").fillna(0.0)

            fc = self.predict(fut)
            fc["horizon"] = h
            outputs.append(fc)

            # Advance history by one step using forecast as the observed total
            next_hist = fut.copy()
            next_hist[fb.cfg.target] = fc["forecast"].values
            next_hist["total_net_sales_m1"] = fc["forecast"].values

            # Carry forward other lags
            for col in ["adv_m1", "CPI_m1"]:
                if col in next_hist.columns:
                    next_hist[col] = pd.to_numeric(
                        next_hist[col], errors="coerce"
                    ).fillna(0.0)

            base_cols = [
                fb.cfg.store_col,
                fb.cfg.time_cols[0],
                fb.cfg.time_cols[1],
                fb.cfg.time_cols[2],
                fb.cfg.target,
                "total_net_sales_m1",
            ]
            feat_cols = [c for c in fb.cfg.feature_list if c in next_hist.columns]
            keep_order: List[str] = []
            for c in base_cols + feat_cols:
                if c in next_hist.columns and c not in keep_order:
                    keep_order.append(c)

            hist = pd.concat([hist, next_hist[keep_order]], ignore_index=True)

        return pd.concat(outputs, ignore_index=True)

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)

        # Save model
        with open(os.path.join(path, "model.pkl"), "wb") as f:
            pickle.dump({"model": self.model, "x_columns": self.x_columns_}, f)

        # Save config + metrics
        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
            f.write(self.cfg.to_json())
        with open(os.path.join(path, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(self.metrics_, f, indent=2)

        print(f"[Save] artifacts written to: {path}")

    @classmethod
    def load(cls, path: str) -> "PooledRidgeForecaster":
        # Load config
        with open(os.path.join(path, "config.json"), "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        cfg = ForecastConfig(**cfg_dict)
        inst = cls(cfg)

        # Load model
        with open(os.path.join(path, "model.pkl"), "rb") as f:
            payload = pickle.load(f)
        inst.model = payload.get("model")
        inst.x_columns_ = payload.get("x_columns")

        # Load metrics if present
        metrics_path = os.path.join(path, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                inst.metrics_ = json.load(f)

        print(f"[Load] artifacts loaded from: {path}")
        return inst

    def needs_retrain(
        self,
        new_feature_list: Optional[Sequence[str]] = None,
        new_alpha: Optional[float] = None,
    ) -> bool:
        """
        Simple check for feature/alpha drift that should trigger retraining.
        """
        features = (
            list(new_feature_list)
            if new_feature_list is not None
            else list(self.cfg.feature_list)
        )
        alpha = float(new_alpha) if new_alpha is not None else float(self.cfg.alpha)
        exclude = (
            list(self.cfg.exclude_stores)
            if self.cfg.exclude_stores is not None
            else []
        )
        proposed = json.dumps(
            {"features": features, "alpha": alpha, "exclude_stores": exclude},
            sort_keys=True,
        )
        return proposed != self.cfg.hash_key()
