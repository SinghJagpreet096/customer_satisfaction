from zenml import pipeline

from steps.ingest_data import data_ingestion
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluation import model_evaluate

@pipeline(enable_cache=True)
def train_pipeline(data_path: str):
    df = data_ingestion(data_path)
    clean_df(df)
    train_model(df)
    model_evaluate(df)

