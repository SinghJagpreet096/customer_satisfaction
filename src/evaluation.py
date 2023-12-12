import logging
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score


from abc import ABC, abstractmethod

class Evaluation(ABC):

    """
    
    """

    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):

        pass

class MSE(Evaluation):

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"MSE: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error in calculating mse: {e}")
            raise e
        
class R2(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating R2 score")
            return r2_score(y_true, y_pred)
        except Exception as e:
            logging.error(f"Error in calculating R2 score: {e}")
            raise e
        
class RMSE(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating RMSE")
            return np.sqrt(mean_squared_error(y_true,y_pred))
        
        except Exception as e:
            logging.error(f"Error in calculating RMSE: {e}")
            raise e