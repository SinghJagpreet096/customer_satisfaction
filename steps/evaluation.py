import logging
import pandas as pd
from zenml import step
from src.evaluation import MSE, R2, RMSE
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated

@step
def model_evaluate(
    model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series) -> Tuple[
        Annotated[float, "mse_score"],
        Annotated[float, "rmse_score"],
        Annotated[float, "r2"]
    ]:
    try:
        logging.info(f"evaluation begin")

        prediction = model.predict(X_test)
        mse_score = MSE().calculate_scores(y_test,prediction)
        rmse_score = RMSE().calculate_scores(y_test,prediction)
        r2 = R2().calculate_scores(y_test,prediction)
        return mse_score, rmse_score, r2
    except Exception as e:
        logging.error(f"Error in model evaluate: {e}")
        raise e
