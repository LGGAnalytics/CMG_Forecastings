from __future__ import annotations

"""
pipeline_lite.py – compact pipeline for pooled forecasts, peer helpers,
and basic utilities. This module keeps notebook imports stable.

Note: To avoid duplication and risk, business-rule assignment is re-exported
from forecasting.model_selection.assign_store_models.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Set

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from forecasting.model_selection import assign_store_models  # re-export


# ----------------------------- Config + Builder ------------------------------

@dataclass
class ForecastConfig:
    feature_list: List[str]
    target: str = "total_net_sales"
    store_col: str = "store_number"
    time_cols: Tuple[str, str, str] = ("event_date", "fiscal_year", "fiscal_period")
    alpha: float = 5.0
    min_history_train: int = 12
    min_history_eval: int = 18
    holdout_k: int = 6
    exclude_stores: Optional[Sequence[object]] = None
    verbose: bool = True

    def excluded_set(self) -> set:
        src = self.exclude_stores or []
        s: set = set()
        for v in src:
            t = str(v).strip()
            s.add(t)
            s.add(t.lstrip("0"))
        return s


class FeatureBuilder:
    def __init__(self, config: ForecastConfig):
        self.cfg = config

    def _standardize_fiscal_cols(self, cal: pd.DataFrame) -> pd.DataFrame:
        fy_col, fp_col = self.cfg.time_cols[1], self.cfg.time_cols[2]
        lower = {c.lower(): c for c in cal.columns}
        fy_src = lower.get("fiscal_year") or lower.get("year") or lower.get("fy") or lower.get("fiscalyear")
        fp_src = lower.get("fiscal_period") or lower.get("period") or lower.get("fp") or lower.get("fiscalperiod")
        if fy_src and fp_src and (fy_src != fy_col or fp_src != fp_col):
            cal = cal.rename(columns={fy_src: fy_col, fp_src: fp_col})
        return cal

    def _ensure_calendar(self, cal: pd.DataFrame) -> pd.DataFrame:
        out = cal.copy()
        if "BeginningDate" in out.columns:
            out["BeginningDate"] = pd.to_datetime(out["BeginningDate"], errors="coerce")
        if "EndingDate" in out.columns:
            out["EndingDate"] = pd.to_datetime(out["EndingDate"], errors="coerce")
        if "days_in_period" not in out.columns and {"BeginningDate", "EndingDate"}.issubset(out.columns):
            out["days_in_period"] = (out["EndingDate"] - out["BeginningDate"]).dt.days + 1
        return out

    def _seasonality(self, df: pd.DataFrame) -> pd.DataFrame:
        fy_col, fp_col = self.cfg.time_cols[1], self.cfg.time_cols[2]
        out = df.copy()
        fp = pd.to_numeric(out[fp_col], errors="coerce")
        out["p13_sin"] = np.sin(2 * np.pi * fp / 13.0)
        out["p13_cos"] = np.cos(2 * np.pi * fp / 13.0)
        return out

    def build_feature_table(self, base_df: pd.DataFrame, calendar_df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        store_col, date_col, fy_col, fp_col = cfg.store_col, cfg.time_cols[0], cfg.time_cols[1], cfg.time_cols[2]

        df = base_df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
        df[store_col] = df[store_col].astype(str).str.strip()
        ex = cfg.excluded_set()
        if ex:
            df = df[~df[store_col].str.lstrip("0").isin(ex)]
        df[cfg.target] = pd.to_numeric(df[cfg.target], errors="coerce")
        if "total_advertising" in df.columns:
            df["total_advertising"] = pd.to_numeric(df["total_advertising"], errors="coerce").fillna(0.0)

        cal = self._ensure_calendar(calendar_df)
        cal = self._standardize_fiscal_cols(cal)
        if "is_5w" not in cal.columns:
            if "days_in_period" in cal.columns:
                cal["is_5w"] = (pd.to_numeric(cal["days_in_period"], errors="coerce") >= 35).astype(int)
            else:
                cal["is_5w"] = 0

        if not {fy_col, fp_col}.issubset(df.columns):
            daily = (
                cal.assign(event_date=lambda d: d.apply(
                    lambda r: pd.date_range(r.get("BeginningDate"), r.get("EndingDate"), freq="D"), axis=1))
                   .explode("event_date")[["event_date", fy_col, fp_col, "is_5w"]]
            )
            df = df.merge(daily, on="event_date", how="left")
        else:
            df = df.merge(cal[[fy_col, fp_col, "is_5w"]].drop_duplicates(), on=[fy_col, fp_col], how="left")

        df = self._seasonality(df)
        df = df.sort_values([store_col, date_col])
        if "total_net_sales_m1" not in df.columns:
            df["total_net_sales_m1"] = df.groupby(store_col)[cfg.target].shift(1)
        if "adv_m1" not in df.columns:
            # Build lag from the advertising column if present; else fill zeros.
            if "total_advertising" in df.columns:
                df["adv_m1"] = df.groupby(store_col)["total_advertising"].shift(1)
            else:
                df["adv_m1"] = 0.0
        df["adv_m1"] = pd.to_numeric(df["adv_m1"], errors="coerce").fillna(0.0)

        feats = [c for c in cfg.feature_list if c in df.columns]
        out = df[[store_col, date_col, fy_col, fp_col, cfg.target] + feats].copy()
        out = out.dropna(subset=[cfg.target, "total_net_sales_m1"]) if "total_net_sales_m1" in out.columns else out.dropna(subset=[cfg.target])
        if feats:
            out[feats] = out[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return out


# ------------------------------- Forecaster ---------------------------------

class PooledRidgeForecaster:
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
        df = df_features.copy()
        if cfg.excluded_set():
            df = df[~df[store_col].astype(str).str.strip().str.lstrip("0").isin(cfg.excluded_set())]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values([store_col, date_col])

        counts = df.groupby(store_col)[date_col].size()
        train_stores = counts[counts >= cfg.min_history_train].index
        eval_stores = counts[counts >= cfg.min_history_eval].index
        df_train_all = df[df[store_col].isin(train_stores)].copy()

        holdout_frames, train_frames = [], []
        for st, g in df_train_all.groupby(store_col):
            g = g.sort_values(date_col)
            if st in set(eval_stores) and cfg.holdout_k > 0:
                holdout_frames.append(g.tail(cfg.holdout_k))
                train_frames.append(g.iloc[:-cfg.holdout_k])
            else:
                train_frames.append(g)
        df_train = pd.concat(train_frames, ignore_index=True) if train_frames else pd.DataFrame()
        df_valid = pd.concat(holdout_frames, ignore_index=True) if holdout_frames else pd.DataFrame()
        if df_train.empty:
            raise ValueError("No training rows after applying min_history_train filter.")

        X_train, y_train = self._build_design(df_train)
        model = Ridge(alpha=cfg.alpha, random_state=0).fit(X_train, y_train)

        metrics: Dict[str, float] = {"rows_train": float(len(df_train)), "rows_eval": float(len(df_valid)), "alpha": float(cfg.alpha)}
        if not df_valid.empty:
            Xv, yv = self._build_design(df_valid)
            Xv = Xv.reindex(columns=X_train.columns, fill_value=0)
            preds = model.predict(Xv)
            denom = float(np.sum(np.abs(yv.values)))
            metrics["wape_holdout"] = float(np.sum(np.abs(yv.values - preds)) / denom) if denom > 0 else float("nan")

        X_all, y_all = self._build_design(df_train_all)
        model.fit(X_all, y_all)
        self.model, self.x_columns_, self.metrics_ = model, list(X_all.columns), metrics
        return metrics

    def predict(self, df_future: pd.DataFrame) -> pd.DataFrame:
        if self.model is None or self.x_columns_ is None:
            raise RuntimeError("Model not trained/loaded.")
        cfg = self.cfg
        df = df_future.copy()
        if cfg.excluded_set():
            df = df[~df[cfg.store_col].astype(str).str.strip().str.lstrip("0").isin(cfg.excluded_set())]
        for col in cfg.feature_list:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        Xf, _ = self._build_design(df.assign(**{cfg.target: 0}))
        Xf = Xf.reindex(columns=self.x_columns_, fill_value=0)
        preds = self.model.predict(Xf)
        fy_col, fp_col = cfg.time_cols[1], cfg.time_cols[2]
        out = df[[cfg.store_col, fy_col, fp_col]].copy()
        out["forecast"] = preds.astype(float)
        return out


# --------------------------------- Helpers ----------------------------------

def normalize_hist(c_heatmap: pd.DataFrame) -> pd.DataFrame:
    need = {"store_number", "fiscal_year", "fiscal_period", "total_net_sales"}
    missing = list(need - set(c_heatmap.columns))
    if missing:
        raise KeyError(f"c_heatmap missing columns: {missing}")
    df = c_heatmap.copy()
    df["store_number"] = df["store_number"].astype(str).str.strip()
    df["fiscal_year"] = pd.to_numeric(df["fiscal_year"], errors="coerce").astype("Int64")
    df["fiscal_period"] = pd.to_numeric(df["fiscal_period"], errors="coerce").astype("Int64")
    df["actual"] = pd.to_numeric(df["total_net_sales"], errors="coerce")
    return df


def normalize_fc_store(fc_store: pd.DataFrame) -> pd.DataFrame:
    need = {"store_number", "fiscal_year", "fiscal_period", "forecast"}
    missing = list(need - set(fc_store.columns))
    if missing:
        raise KeyError(f"fc_store missing columns: {missing}")
    df = fc_store.copy()
    df["store_number"] = df["store_number"].astype(str).str.strip()
    df["fiscal_year"] = pd.to_numeric(df["fiscal_year"], errors="coerce").astype("Int64")
    df["fiscal_period"] = pd.to_numeric(df["fiscal_period"], errors="coerce").astype("Int64")
    df["forecast"] = pd.to_numeric(df["forecast"], errors="coerce")
    return df


def split_short_long(hist_norm: pd.DataFrame, threshold: int = 18) -> Tuple[Set[str], Set[str]]:
    per_counts = (
        hist_norm.dropna(subset=["fiscal_year", "fiscal_period"])
                 .assign(_per=hist_norm["fiscal_year"] * 100 + hist_norm["fiscal_period"])
                 .groupby("store_number")["_per"].nunique()
    )
    short = set(per_counts[per_counts < threshold].index)
    long = set(per_counts[per_counts >= threshold].index)
    return short, long


def last6_axis_from_fc(fc_norm: pd.DataFrame) -> pd.DataFrame:
    fy, fp = "fiscal_year", "fiscal_period"
    return (
        fc_norm[[fy, fp]].dropna().drop_duplicates().sort_values([fy, fp]).tail(6)
    )


def wape(y: pd.Series, yhat: pd.Series) -> float:
    y = pd.to_numeric(y, errors="coerce")
    yhat = pd.to_numeric(yhat, errors="coerce")
    m = y.notna() & yhat.notna()
    y, yhat = y[m], yhat[m]
    denom = y.abs().sum()
    return float(np.nan) if denom == 0 else float((y - yhat).abs().sum() / denom)


# join_actuals is provided in forecasting.runner_utils; no duplicate here


def build_peer_scaled_short(
    hist_norm: pd.DataFrame,
    fc_norm: pd.DataFrame,
    peer_table_k3_short: pd.DataFrame,
    axis6: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build peer forecasts for short stores by scaling peers' pooled forecasts.

    Returns columns: store_number, fiscal_year, fiscal_period, forecast, source='peer'
    """
    fy, fp = "fiscal_year", "fiscal_period"
    hist = hist_norm[["store_number", fy, fp, "actual"]].copy()
    peers = peer_table_k3_short.copy()
    for c in ["store_number", "peer1", "peer2", "peer3"]:
        if c in peers.columns:
            peers[c] = peers[c].astype(str).str.strip()

    def _avg_first10(store: str) -> float:
        g = (
            hist.loc[hist["store_number"] == store, [fy, fp, "actual"]]
                .dropna().sort_values([fy, fp]).head(10)
        )
        return float("nan") if g.empty else float(g["actual"].mean())

    if axis6 is None:
        axis6 = last6_axis_from_fc(fc_norm)

    rows: List[dict] = []
    for st in peers["store_number"].astype(str):
        r = peers.loc[peers["store_number"] == st]
        if r.empty:
            continue
        r = r.iloc[0]
        avg_short = _avg_first10(st)
        peer10 = [_avg_first10(str(r.get(k, ""))) for k in ("peer1", "peer2", "peer3")]
        peer10 = [v for v in peer10 if pd.notna(v)]
        if pd.isna(avg_short) or not peer10 or np.mean(peer10) == 0:
            continue
        scale = float(avg_short / np.mean(peer10))

        for _, a in axis6.iterrows():
            fy_i, fp_i = int(a[fy]), int(a[fp])
            vals = []
            for k in ("peer1", "peer2", "peer3"):
                pid = str(r.get(k, "")).strip()
                if not pid or pid.lower() == "nan":
                    continue
                g = fc_norm.loc[
                    (fc_norm["store_number"] == pid) & (fc_norm[fy] == fy_i) & (fc_norm[fp] == fp_i), "forecast"
                ]
                if not g.empty and pd.notna(g.iloc[0]):
                    vals.append(float(g.iloc[0]))
            if vals:
                rows.append({
                    "store_number": st, fy: fy_i, fp: fp_i,
                    "forecast": scale * float(np.mean(vals)), "source": "peer"
                })

    return pd.DataFrame(rows, columns=["store_number", fy, fp, "forecast", "source"]) if rows else pd.DataFrame(
        columns=["store_number", fy, fp, "forecast", "source"]
    )


def finalize_forecasts(fc_model: pd.DataFrame, fc_peer: pd.DataFrame, short_set: Set[str]) -> pd.DataFrame:
    fy, fp = "fiscal_year", "fiscal_period"
    m = fc_model.copy()
    m["source"] = "model"
    out = m.merge(fc_peer[["store_number", fy, fp, "forecast"]].rename(columns={"forecast": "peer_pred"}),
                  on=["store_number", fy, fp], how="left")
    use_peer = out["store_number"].isin({str(s).strip() for s in short_set}) & out["peer_pred"].notna()
    out["forecast"] = np.where(use_peer, out["peer_pred"], out["forecast"])  # choose peer for shorts when available
    out["source"] = np.where(use_peer, "peer", out["source"])
    return out.drop(columns=["peer_pred"]) if "peer_pred" in out.columns else out


def run_pooled_ridge(
    heatmap: pd.DataFrame,
    calendar_df: pd.DataFrame,
    feature_list: Optional[List[str]] = None,
    alpha: float = 5.0,
    holdout_k: int = 6,
    min_history_train: int = 12,
    min_history_eval: int = 18,
    verbose: bool = True,
    # Back-compat extras (wired into full builder if provided)
    cpi_df: Optional[pd.DataFrame] = None,
    events_df: Optional[pd.DataFrame] = None,
    H: Optional[int] = None,
    exclude_stores: Optional[Sequence[object]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Train pooled ridge and produce 1-step-ahead forecasts for each observed row.

    Returns (fc_store, feats, metrics).

    Notes
    - Parameters cpi_df, events_df, H are accepted for API compatibility but
      are not used in this lite implementation. If you need them wired, switch
      to the full pooled_ridge.FeatureBuilder with CPI/events.
    """
    # If CPI/events are provided or requested in the features, use the full
    # pooled_ridge builder/forecaster to preserve historical behavior.
    want = set(feature_list or [])
    need_full = bool(cpi_df is not None or events_df is not None or {"CPI_m1", "Super Bowl"} & want)

    if need_full:
        try:
            from forecasting.pooled_ridge import (
                ForecastConfig as PRConfig,
                FeatureBuilder as PRFeatureBuilder,
                PooledRidgeForecaster as PRForecaster,
            )
            ftrs = feature_list or [
                "total_net_sales_m1", "adv_m1", "CPI_m1", "p13_sin", "p13_cos", "is_5w", "Super Bowl"
            ]
            cfg = PRConfig(
                feature_list=ftrs,
                alpha=alpha,
                holdout_k=holdout_k,
                min_history_train=min_history_train,
                min_history_eval=min_history_eval,
                verbose=verbose,
                exclude_stores=exclude_stores,
            )
            fb = PRFeatureBuilder(cfg)
            feats = fb.build_feature_table(heatmap, calendar_df, events=events_df, cpi=cpi_df, exclude_stores=exclude_stores)
            fore = PRForecaster(cfg)
            metrics = fore.train(feats)
            if H is not None and int(H) > 0:
                fc_store = fore.predict_h(
                    fb=fb,
                    history_df=feats,
                    calendar_df=calendar_df,
                    events_df=events_df,
                    cpi_df=cpi_df,
                    H=int(H),
                )
            else:
                fc_store = fore.predict(feats)
            return fc_store, feats, metrics
        except Exception:
            # Fall back to lite path if full module import fails
            pass

    # Lite path (no CPI/events wiring). Minimal default features.
    ftrs = feature_list or ["total_net_sales_m1", "adv_m1", "p13_sin", "p13_cos", "is_5w"]
    cfg = ForecastConfig(
        feature_list=ftrs,
        alpha=alpha,
        holdout_k=holdout_k,
        min_history_train=min_history_train,
        min_history_eval=min_history_eval,
        verbose=verbose,
        exclude_stores=exclude_stores,
    )
    fb = FeatureBuilder(cfg)
    feats = fb.build_feature_table(heatmap, calendar_df)
    fore = PooledRidgeForecaster(cfg)
    metrics = fore.train(feats)
    if H is not None and int(H) > 0:
        fc_store = fore.predict_h(
            fb=fb,
            history_df=feats,
            calendar_df=calendar_df,
            events_df=events_df,
            cpi_df=None,
            H=int(H),
        )
    else:
        fc_store = fore.predict(feats)
    return fc_store, feats, metrics


# ------------------------------- Small helpers -------------------------------

# Filtering helper lives in forecasting.runner_utils; avoid duplicate here


__all__ = [
    "ForecastConfig", "FeatureBuilder", "PooledRidgeForecaster",
    "normalize_hist", "normalize_fc_store", "split_short_long", "last6_axis_from_fc",
    "wape", "build_peer_scaled_short", "finalize_forecasts", "run_pooled_ridge",
    "assign_store_models",
]
