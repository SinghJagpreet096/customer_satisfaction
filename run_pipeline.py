from pipelines.training_pipeline import train_pipeline
from mlflow import MlflowClient



experiment_tracker = MlflowClient(tracking_uri="http://127.0.0.1:8080").search_experiments

if __name__ == "__main__":
    print(experiment_tracker.artifact_location)
    train_pipeline(data_path="data/olist_customers_dataset.csv")