import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_part: str=os.path.join('artifacts','data.csv')

    # train_data_part and other 2 are string variables which stores the path of train.csv, test.csv,data.cvs in artifacts file.

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info('Read data from the dataset as a DataFrame')

            artifact_dir=os.path.dirname(self.ingestion_config.train_data_path)  # There is no particular reason for choosing tran_data_part, because all of it is in the same directory.
            os.makedirs(artifact_dir,exist_ok=True)

            logging.info("Train Test Split Initiated")

            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            df.to_csv(self.ingestion_config.raw_data_part,index=False,header=True)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)

if __name__=='__main__':
    obj=DataIngestion()
    obj.initiate_data_ingestion()