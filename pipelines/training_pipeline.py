from zenml import pipeline

from steps.ingest_data import data_ingestion
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluation import model_evaluate

@pipeline(enable_cache=False)
def train_pipeline(data_path: str):
    df = data_ingestion(data_path)
    X_train, X_test, y_train, y_test = clean_df(df)

    model = train_model(X_train,
                        X_test,
                        y_train,
                        y_test)
    
    mse_score, rmse_score, r2 = model_evaluate(model, X_test, y_test)
    print(f"MSE: {mse_score} RMSE: {rmse_score} R2: {r2}")

