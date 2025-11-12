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
