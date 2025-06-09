# The objective of this module is to perform Feature Engineering and Data transformation
# Returns Array of transformed train and test data and the path of preprocessor.pkl file

import os
import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_obj(self):

        # This method is responsible for Transforming the data

        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline=Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy='median')), # Fill in the null values with median of the column.
                    ("scaler",StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')), # Fill in the null values with most frequent / occuring value.
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False)) # Doesnt subtract mean from data during standardization.
                ]
            )

            logging.info(f"Numerical Columns : {numerical_columns}")
            logging.info(f"Categorical Columns : {categorical_columns}")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):  # the train path and test path are sent from data ingestion, it locates the dataset in artifacts folder.
            try:
                train_df=pd.read_csv(train_path)
                test_df=pd.read_csv(test_path)

                logging.info("Read train data and test data")

                preprocessor_obj=self.get_data_transformer_obj()
                logging.info("Obtained preprocessor object")

                target_column_name="math_score"

                input_feature_train=train_df.drop(columns=[target_column_name],axis=1)
                input_featur_test=test_df.drop(columns=[target_column_name],axis=1)

                target_feature_train=train_df[target_column_name]
                target_feature_test=test_df[target_column_name]

                logging.info("Applying transformation on train and test dataframes")

                input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train)
                input_feature_test_arr=preprocessor_obj.transform(input_featur_test)

                ''' np.c_ is used for concatenating feature columns. Here, both the input_feature_train_arr and target_feature_train are combined as a single array
                    so that it can be sent to model trainer, where the data is again split into train and test data for actual training purpose.
                    here, it is only separated to transform the input feature data.
                    That's why only the preprocessor_obj is turned as pickle file, which does not contain the spliting part code.
                '''

                train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train)]
                test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test)]

                logging.info("Presprocessing completed")

                save_object(
                     file_path=self.data_transformation_config.preprocessor_obj_file_path,
                     obj=preprocessor_obj
                )

                logging.info("Saved preprocessor object")

                return (
                     train_arr,
                     test_arr,
                     self.data_transformation_config.preprocessor_obj_file_path
                )


            except Exception as e:
                 raise CustomException(e,sys)