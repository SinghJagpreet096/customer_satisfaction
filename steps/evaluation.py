import logging
import pandas as pd
from zenml import step

@step
def model_evaluate(df: pd.DataFrame):
    logging.info(f"evaluation begin")
    pass