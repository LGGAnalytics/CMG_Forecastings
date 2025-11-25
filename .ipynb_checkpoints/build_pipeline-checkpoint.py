# pipeline.py

import os
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker import image_uris

region = os.getenv("AWS_REGION", "us-east-1")
sess = sagemaker.Session()
pipeline_session = PipelineSession()
role = "arn:aws:iam::216097839164:role/service-role/AmazonSageMaker-ExecutionRole-20251111T151323"



# --------- Parameters (override at run-time) ---------
athena_staging = ParameterString(
    name="AthenaStagingDir",
    default_value="s3://data-science-lgg/staging/",
)

calendar_s3 = ParameterString(
    name="CalendarData",
    default_value="s3://data-science-lgg/calendar/",
)

cpi_s3 = ParameterString(
    name="CpiData",
    default_value="s3://data-science-lgg/cpi/",
)

events_s3 = ParameterString(
    name="EventsData",
    default_value="s3://data-science-lgg/events/",
)

offline_store_s3 = ParameterString(
    name="OfflineStoreS3Uri",
    default_value="s3://data-science-lgg/offline_store//heatmap/",
)

training_features_s3 = ParameterString(
    name="TrainingFeaturesS3Uri",
    default_value="s3://data-science-lgg/features/",
)

feature_group_name = ParameterString(
    name="FeatureGroupName",
    default_value="lc-heatmap-features",
)

# --------- ScriptProcessor ---------
image_uri = image_uris.retrieve(
    framework="sklearn",
    region=region,
    version="1.2-1",
    py_version="py3",
    instance_type="ml.t3.medium",  # helps choose cpu/gpu tag
)

processor = ScriptProcessor(
    image_uri=image_uri,
    role=role,
    instance_type="ml.t3.medium",
    instance_count=1,
    command=["python"],
    sagemaker_session=pipeline_session,
)

# Arguments passed into processing_feature_store_ingest.py
step_args = processor.run(
    code="preprocessing.py",
    inputs=[
        ProcessingInput(
            source=calendar_s3,
            destination="/opt/ml/processing/calendar",
            input_name="calendar",
        ),
        ProcessingInput(
            source=cpi_s3,
            destination="/opt/ml/processing/calendar",
            input_name="cpi",
        ),
        ProcessingInput(
            source=events_s3,
            destination="/opt/ml/processing/calendar",
            input_name="events",
        ),
    ],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/output",
            destination=training_features_s3,
            output_name="training_features",
        ),
    ],
    arguments=[
        "--athena-database", "little_caesars",
        "--athena-table", "heatmap",
        "--athena-work-group", "primary",
        "--athena-staging-dir", athena_staging,
        "--calendar-path", "/opt/ml/processing/calendar",
        "--feature-group-name", feature_group_name,
        "--offline-store-s3-uri", offline_store_s3,
        "--role-arn", role,
        "--enable-online-store",
        "--region", region,
    ],
)

process_step = ProcessingStep(
    name="BuildFeaturesToFeatureStore",
    step_args=step_args,
)

# For now pipeline only has the processing step.
# Later you add a TrainingStep that consumes process_step properties.
pipeline = Pipeline(
    name="Heatmap-FeatureStore-Pipeline",
    parameters=[
        athena_staging,
        calendar_s3,
        offline_store_s3,
        training_features_s3,
        feature_group_name,
    ],
    steps=[process_step],
    sagemaker_session=pipeline_session,
)


def get_pipeline():
    return pipeline
