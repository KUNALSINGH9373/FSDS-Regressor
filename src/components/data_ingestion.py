
# reading the datset

import os # to create a file path or join path
import sys # used for system error 
import sys
from src.logger import logging
from src.exception import CustomException # for exception handling
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass #avoids repetitive boilerplate code fro initilization


from src.components.data_transformation import DataTransformation

# data_ingestion.py


 ## Initialize the data ingestion configuration

@dataclass #constructor
class DataIngestionconfig:
    train_data_path:str=os.path.join('artifacts','train.csv')  #  traning test data path
    test_data_path:str=os.path.join('artifacts', 'test.csv') 
    raw_data_path:str=os.path.join('artifacts', 'raw.csv')

## create a class for Data Ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion methods starts')
        try:
            df=pd.read_csv(os.path.join('notebooks','gemstone.csv'))  # Read the dataset into a pandas DataFrame
            logging.info('Dataset read as pandas Dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True) # Create directories if they do not exist
            df.to_csv(self.ingestion_config.raw_data_path,index=False)   # Save the raw dataset
            logging.info('Train test split')
            train_set, test_set=train_test_split(df, test_size=0.30)

            # Save the training and testing datasets
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path), exist_ok=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Ingestion of Data is completed')   

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info('Exception occured at Data Ingestion stage')
            # Raise a custom exception with the original exception and system information
            raise CustomException(e,sys)

