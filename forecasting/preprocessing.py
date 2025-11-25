import sys, subprocess

def pip_install(pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", *pkgs])

# ensure runtime deps
for pkg in ["pyathena", "openpyxl", "pyarrow"]:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        pip_install([pkg])
        
import argparse 
import json
import logging
import os
from typing import List

import pandas as pd
import numpy as np
try:
    import boto3  # optional for local use
except Exception:  # pragma: no cover
    boto3 = None
try:
    from pyathena import connect  # optional; used only in CLI main
except Exception:  # pragma: no cover
    connect = None
try:
    # Optional SageMaker imports; not required for local profiling
    from sagemaker.session import Session
    from sagemaker.feature_store.feature_group import FeatureGroup
    from sagemaker.feature_store.feature_definition import FeatureDefinition, FeatureTypeEnum
except Exception:  # pragma: no cover
    Session = FeatureGroup = FeatureDefinition = FeatureTypeEnum = None

# Your modules (ensure these are in the container/source_dir)
try:
    from pooled_ridge import ForecastConfig, FeatureBuilder
except Exception:
    # Allow package-style import when used as forecasting.preprocessing
    from forecasting.pooled_ridge import ForecastConfig, FeatureBuilder

try:
    from feature_store import *  # optional; used only in CLI main
except Exception:  # pragma: no cover
    pass

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

EXCLUDE = [364001,364002,364003,364004,364005,364006,364007,364008,364009,364010]
FEATURES = ['total_net_sales_m1','adv_m1','CPI_m1','p13_sin','p13_cos','is_5w','Super Bowl']

# ---------------------------------------------------------------------------
# Minimal local helper: build_store_profile
# ---------------------------------------------------------------------------
def build_store_profile(
    c_heatmap: pd.DataFrame,
    include_opex_pct: bool = False,
    compute_event_uplifts: bool = False,
    column_map: dict | None = None,
) -> pd.DataFrame:
    """Create a per-store profile for PCA/peers without SageMaker deps.

    Expected inputs in c_heatmap: store_number, total_net_sales, event_date (or FY/FP).
    Returns columns: store_number, avg_net_sales, std_net_sales, cv_net_sales,
    log_avg_net_sales, n_periods, optional State.
    """
    df = c_heatmap.copy()
    df['store_number'] = df['store_number'].astype(str).str.strip()
    df['total_net_sales'] = pd.to_numeric(df.get('total_net_sales'), errors='coerce')
    # One row per fiscal period or date is fine; just summarize
    g = df.groupby('store_number', as_index=False)
    prof = g['total_net_sales'].agg(avg_net_sales='mean', std_net_sales='std', n_periods='size')
    prof['cv_net_sales'] = (prof['std_net_sales'] / prof['avg_net_sales']).replace([np.inf, -np.inf], np.nan)
    prof['log_avg_net_sales'] = np.log1p(prof['avg_net_sales'])
    # Optional: keep a stable State per store if present
    if 'State' in df.columns:
        mode_state = (
            df.dropna(subset=['State']).groupby('store_number')['State']
              .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0])
        )
        prof = prof.merge(mode_state.rename('State'), on='store_number', how='left')
    return prof

# ---------------------------
# Arg parsing
# ---------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    # Athena config (matches how you're already querying)
    parser.add_argument("--athena-database", type=str, required=True)
    parser.add_argument("--athena-table", type=str, required=True)
    parser.add_argument("--athena-work-group", type=str, default="primary")
    parser.add_argument("--athena-staging-dir", type=str, required=True)

    # Extra inputs (Excel or others) mounted as ProcessingInputs
    parser.add_argument("--calendar-path", type=str, default=None)
    parser.add_argument("--events-path", type=str, default=None)
    parser.add_argument("--cpi-path", type=str, default=None)

    # Output for training step
    parser.add_argument("--output-features-path", type=str,
                        default="/opt/ml/processing/output")

    # Feature Store / AWS
    parser.add_argument("--region", type=str,
                        default=os.getenv("AWS_REGION", "eu-west-1"))
    parser.add_argument("--feature-group-name", type=str, required=True)
    parser.add_argument("--offline-store-s3-uri", type=str, required=True)
    parser.add_argument("--role-arn", type=str, required=True)
    parser.add_argument("--enable-online-store", action="store_true")

    # Required FS columns
    parser.add_argument("--record-identifier-name", type=str, default="record_id")
    parser.add_argument("--event-time-feature-name", type=str, default="event_time")

    # Ingestion parallelism
    parser.add_argument("--max-processes", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=4)


    return parser.parse_args()

# ---------------------------
# IO helpers
# ---------------------------
def read_first_table_from_path(path: str) -> pd.DataFrame:
    """
    Read CSV/Parquet/Excel from a directory or a file.

    In your Pipeline, map ProcessingInput to these dirs:
      - base  -> /opt/ml/processing/base
      - calendar -> /opt/ml/processing/calendar
      - etc.
    """
    if path is None:
        return pd.DataFrame()

    if os.path.isdir(path):
        files = sorted(
            f
            for f in os.listdir(path)
            if f.lower().endswith((".csv", ".parquet", ".xlsx", ".xls"))
        )
        if not files:
            raise FileNotFoundError(f"No supported files found in: {path}")
        path = os.path.join(path, files[0])

    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)

    raise ValueError(f"Unsupported file type: {path}")


# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()

    # Sessions
    if boto3 is None or Session is None:
        raise RuntimeError("SageMaker/boto3 not available in this environment; CLI main is AWS-only.")
    boto_session = boto3.Session(region_name=args.region)
    sagemaker_session = Session(boto_session=boto_session)

    logger.info(
        "Querying Athena table %s.%s via workgroup %s",
        args.athena_database,
        args.athena_table,
        args.athena_work_group,
    )

    if connect is None:
        raise RuntimeError("pyathena not available; cannot query Athena in this environment.")
    conn = connect(
        s3_staging_dir=args.athena_staging_dir,
        region_name=args.region,
        work_group=args.athena_work_group,
    )

    sql = f"SELECT * FROM {args.athena_database}.{args.athena_table}"
    base_df = pd.read_sql(sql, conn)
    logger.info("Loaded %d rows from Athena", len(base_df))

    # 2. Load Excel/other side inputs (from ProcessingInputs)
    calendar_df = read_first_table_from_path(args.calendar_path)
    events_df = read_first_table_from_path(args.events_path)
    cpi_df = read_first_table_from_path(args.cpi_path)

    if calendar_df.empty:
        raise RuntimeError("Calendar data is required but missing/empty.")


    # 3. Build feature table with your existing logic
    cfg = ForecastConfig(feature_list=FEATURES)
    fb = FeatureBuilder(cfg)

    feats = fb.build_feature_table(
        base_df=base_df,
        calendar_df=calendar_df,
        events=events_df if not events_df.empty else None,
        cpi=cpi_df if not cpi_df.empty else None,
    )

    # 4. Final sanitation before training
    feats = sanitize_training_features(feats, FEATURES, cfg.target)

    # 5. Add FS required columns (adapt to your config layout)
    date_col = cfg.time_cols[0]
    fy_col = cfg.time_cols[1]
    fp_col = cfg.time_cols[2]

    feats[date_col] = pd.to_datetime(feats[date_col], errors="coerce")
    feats[args.event_time_feature_name] = (
        feats[date_col].astype("int64") // 10**9
    )

    feats[args.record_identifier_name] = (
        feats[cfg.store_col].astype(str).str.strip()
        + "_"
        + feats[fy_col].astype(int).astype(str).str.zfill(4)
        + "_"
        + feats[fp_col].astype(int).astype(str).str.zfill(2)
    )

    feats = feats.dropna(
        subset=[args.record_identifier_name, args.event_time_feature_name]
    )

    logger.info("Final features: %d rows, %d columns", *feats.shape)

    # 6. Save for training step
    os.makedirs(args.output_features_path, exist_ok=True)
    out_path = os.path.join(args.output_features_path, "training_features.parquet")
    feats.to_parquet(out_path, index=False)
    logger.info("Wrote training features to %s", out_path)

    # 7. Ingest into Feature Store
    fg = get_or_create_feature_group(
        sagemaker_session=sagemaker_session,
        feature_group_name=args.feature_group_name,
        df=feats,
        record_identifier_name=args.record_identifier_name,
        event_time_feature_name=args.event_time_feature_name,
        offline_store_s3_uri=args.offline_store_s3_uri,
        role_arn=args.role_arn,
        enable_online_store=args.enable_online_store,
    )

    logger.info("Ingesting into Feature Group '%s'...", args.feature_group_name)
    fg.ingest(
        data_frame=feats,
        max_processes=args.max_processes,
        max_workers=args.max_workers,
        wait=True,
    )
    logger.info("Feature Store ingestion complete.")

if __name__ == "__main__":
    main()
