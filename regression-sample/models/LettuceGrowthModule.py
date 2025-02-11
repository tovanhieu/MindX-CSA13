import os
import logging
import pickle

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import Config


class LettuceGrowthModule:
    def __init__(self, model) -> None:
        if model == "Linear Regression":
            self.model = LinearRegression()
        else:
            self.model = None

        logging.basicConfig(filename=Config.LOG_PATH,
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

    def train(self, X, y):
        try:
            logging.info("Try model fitting")
            self.model.fit(X, y)

            checkpoint_model(self.model)
            logging.info("Model fitting completed and saved to disk.")
            logging.debug(f'Model coefficients: {self.model.coef_}')
            logging.debug(f'Model intercept: {self.model.intercept_}')
            return True
            
        except Exception as e:
            logging.error("Error during model fitting: %s", e)
            return False
        
    def inference(self, X):
        loaded_model = load_model()

        if loaded_model:
            y_preds = loaded_model.predict(X)
            return y_preds
        
        return None
        
    def eval(self, y, y_preds):
        mae = mean_absolute_error(y, y_preds)
        mse = mean_squared_error(y, y_preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_preds)

        return {
            "Mean Absolute Error": mae,
            "Mean Squared Error": mse,
            "Root Mean Squared Error": rmse,
            "R2 Score": r2
        }
    
def checkpoint_model(model):
    with open(Config.MODEL_PATH, 'wb') as file:
        pickle.dump(model, file)

def load_model():
    if os.path.isfile(Config.MODEL_PATH):
        with open(Config.MODEL_PATH, 'rb') as file:
            loaded_model = pickle.load(file)
        return loaded_model
    
    else:
        return None 
