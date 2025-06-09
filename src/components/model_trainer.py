# This module finds the best model and creates a model.pkl file
# Returns the r2_score of the best model

import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,AdaBoostRegressor

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models,save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Splitting the train and test data")
            X_train,X_test,y_train,y_test=(
                train_arr[:,:-1],
                test_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,-1]
            )
           
            models={
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Catboost":CatBoostRegressor(verbose=0), # catboost will print all the training it does, so to not see it, set verbose to 0.
                "XG Boost":XGBRegressor(),
                "Gradient Boost":GradientBoostingRegressor(),
                "Ada Boost":AdaBoostRegressor(),
                "Linear Regression":LinearRegression(),
                "K Nearest Neighbour":KNeighborsRegressor()
            }
            logging.info("Send models for evaluation")

            result=evaluate_models(X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test,models=models)

            logging.info("Received the model report")

            best_score=max(result.values())
            if(best_score<0.6):
                raise CustomException("No best model found")

            else:
                best_model_name=list((result.keys()))[list((result.values())).index(best_score)]
                best_model=models[best_model_name]
                logging.info("Best model has found for both training and testing dataset")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            r2_square=r2_score(y_test,predicted)
            return r2_square


        except Exception as e:
            raise CustomException(e,sys)
        
        