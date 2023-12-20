import numpy as np 
import pandas as pd 
from zenml import pipeline, step
from zenml.config import DockerSettings
from materializer.custom_materializer import cs_materializer

from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer

from zenml.integrations.mlflow.services import MLFlowDeploymentServices

from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output

from steps.clean_data import clean_df
from steps.evaluation import model_evaluate
from steps.ingest_data import data_ingestion
from steps.model_train import train_model

docker_settings = DockerSettings(required_integrations=[MLFLOW])

class DeploymentTriggerConfig(BaseParameters):
    """Deployment Trigger config"""
    min_accuracy: float =0.92

@step
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig,
):
    return accuracy >=config.min_accuracy

@pipeline(enable_cache=True, settings={"docker_settings":docker_settings})
def continous_deployment_pipeline(
    min_accuracy: float = 0.92,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT
):
    df = data_ingestion(data_path)
    X_train, X_test, y_train, y_test = clean_df(df)

    model = train_model(X_train,
                        X_test,
                        y_train,
                        y_test)
    
    mse_score, rmse_score, r2 = model_evaluate(model, X_test, y_test)
    deployment_decision = deployment_trigger(r2)
    mlflow_model_deployer_step(
        model=model,
        deployment_decision=deployment_decision,
        workers = workers,
        timeout = timeout

    )
    