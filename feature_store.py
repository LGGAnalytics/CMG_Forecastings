import argparse
import json
import logging
import os
from typing import List

import boto3
import pandas as pd
from sagemaker.session import Session
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.feature_store.feature_definition import FeatureDefinition, FeatureTypeEnum

from pooled_ridge import ForecastConfig, FeatureBuilder  # from your repo

import sys, os, importlib, pandas as pd, numpy as np
import forecasting.pooled_ridge as pr
from forecasting.pooled_ridge import ForecastConfig, FeatureBuilder, PooledRidgeForecaster

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)


# ---------------------------
# Feature Store helpers
# ---------------------------
def infer_feature_definitions(
    df: pd.DataFrame,
    record_identifier_name: str,
    event_time_feature_name: str,
) -> List[FeatureDefinition]:
    """Infer Feature Store feature definitions from a DataFrame schema."""
    feature_definitions: List[FeatureDefinition] = []

    for col, dtype in df.dtypes.items():
        if col == record_identifier_name:
            f_type = FeatureTypeEnum.STRING
        elif col == event_time_feature_name:
            # we'll store unix ts (seconds) as fractional
            f_type = FeatureTypeEnum.FRACTIONAL
        elif pd.api.types.is_integer_dtype(dtype):
            f_type = FeatureTypeEnum.INTEGRAL
        elif pd.api.types.is_float_dtype(dtype):
            f_type = FeatureTypeEnum.FRACTIONAL
        else:
            f_type = FeatureTypeEnum.STRING

        feature_definitions.append(
            FeatureDefinition(feature_name=col, feature_type=f_type)
        )

    return feature_definitions

def get_or_create_feature_group(
    sagemaker_session: Session,
    feature_group_name: str,
    df: pd.DataFrame,
    record_identifier_name: str,
    event_time_feature_name: str,
    offline_store_s3_uri: str,
    role_arn: str,
    enable_online_store: bool,
) -> FeatureGroup:
    """Return existing FG or create it from df schema if it doesn't exist."""
    sm_client = sagemaker_session.sagemaker_client

    try:
        sm_client.describe_feature_group(FeatureGroupName=feature_group_name)
        logger.info("Using existing Feature Group '%s'", feature_group_name)
        return FeatureGroup(name=feature_group_name, sagemaker_session=sagemaker_session)
    except sm_client.exceptions.ResourceNotFound:
        logger.info("Creating new Feature Group '%s'", feature_group_name)

    feature_definitions = infer_feature_definitions(
        df,
        record_identifier_name=record_identifier_name,
        event_time_feature_name=event_time_feature_name,
    )

    fg = FeatureGroup(name=feature_group_name, sagemaker_session=sagemaker_session)
    fg.create(
        role_arn=role_arn,
        feature_definitions=feature_definitions,
        record_identifier_name=record_identifier_name,
        event_time_feature_name=event_time_feature_name,
        offline_store_config={
            "S3StorageConfig": {"S3Uri": offline_store_s3_uri}
        },
        online_store_config={"EnableOnlineStore": enable_online_store},
    )
    fg.wait_for_create()
    logger.info("Feature Group '%s' created.", feature_group_name)
    return fg



# ========= Helpers =========
def std_fiscal_cols(cal: pd.DataFrame) -> pd.DataFrame:
    c = cal.copy()
    lower = {col.lower(): col for col in c.columns}
    fy = lower.get('fiscal_year') or lower.get('year') or lower.get('fy') or lower.get('fiscalyear')
    fp = lower.get('fiscal_period') or lower.get('period') or lower.get('fp') or lower.get('fiscalperiod')
    if fy is None or fp is None:
        raise ValueError("Calendar must include fiscal year/period columns.")
    if fy != 'fiscal_year' or fp != 'fiscal_period':
        c = c.rename(columns={fy: 'fiscal_year', fp: 'fiscal_period'})
    for col in ['BeginningDate','EndingDate']:
        if col in c.columns:
            c[col] = pd.to_datetime(c[col], errors='coerce').dt.normalize()
    return c

def build_events_df_fiscal(events_path: str, cal2: pd.DataFrame) -> pd.DataFrame:
    ev = pd.read_excel(events_path)
    ev['event_date'] = pd.to_datetime(ev['event_date'], errors='coerce').dt.normalize()
    sb_col = None
    for c in ev.columns:
        k = c.strip().lower().replace(' ','')
        if k in ('superbowl','super_bowl'):
            sb_col = c; break
    if sb_col is None:
        raise ValueError("No 'Super Bowl' column in events file.")
    ev = ev[['event_date', sb_col]].rename(columns={sb_col: 'Super Bowl'})
    ev['Super Bowl'] = pd.to_numeric(ev['Super Bowl'], errors='coerce').fillna(0).astype(int).clip(0,1)

    span = cal2[['BeginningDate','EndingDate','fiscal_year','fiscal_period']].dropna().copy()
    span['key'] = 1; ev['key'] = 1
    m = (ev.merge(span, on='key', how='inner')
            .query('BeginningDate <= event_date <= EndingDate'))
    events_df = (m[['fiscal_year','fiscal_period','BeginningDate','Super Bowl']]
                 .drop_duplicates(['fiscal_year','fiscal_period'])
                 .rename(columns={'BeginningDate':'event_date'}))
    return events_df[['event_date','fiscal_year','fiscal_period','Super Bowl']].drop_duplicates()

def sanitize_training_features(df: pd.DataFrame, feature_cols: list, target_col: str) -> pd.DataFrame:
    out = df.copy()
    out['store_number'] = out['store_number'].astype(str).str.strip()
    for c in feature_cols:
        out[c] = pd.to_numeric(out[c], errors='coerce')
    out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out[target_col] = pd.to_numeric(out[target_col], errors='coerce')
    out = out.dropna(subset=[target_col])
    return out