from src.ml.model.s3_estimator import CustomerClusterEstimator
from src.logger import logging
from src.entity.config_entity import DataTransformationConfig , ModelTrainerConfig
from src.constant.training_pipeline import *
from src.entity.config_entity import training_pipeline_config
from src.entity.config_entity import Prediction_config, PredictionPipelineConfig

from src.entity.config_entity import DataTransformationConfig , ModelTrainerConfig
from src.logger import logging
from src.utils.main_utils import MainUtils

from src.exception import CustomerException
import pandas as pd
import numpy as np
import sys

import logging
import sys
from pandas import DataFrame
import pandas as pd





class CustomerData:
    def __init__(self):
        pass
        
    def get_input_dataset(self, column_schema:dict, input_data):
        columns = column_schema.keys()
        input_dataset = pd.DataFrame([input_data], columns=columns)

        for key, value in column_schema.items():
            try:
                # Ensure categorical columns remain strings
                if key in ["Education", "Marital_Status", "Parental_Status"]:
                    input_dataset[key] = input_dataset[key].astype(str)
                else:
                    input_dataset[key] = input_dataset[key].astype(value)
            except ValueError as e:
                raise ValueError(f"Error converting {key} to {value}: {e}")
        
        return input_dataset

    @staticmethod
    def form_input_dataframe(data):
        prediction_config = Prediction_config()
        prediction_schema = prediction_config.__dict__
        column_schema = prediction_schema['prediction_schema']['columns']

        customerData = CustomerData()
        input_dataset = customerData.get_input_dataset(
            column_schema=column_schema,
            input_data=data
        )
        
        return input_dataset
        
        
    


class PredictionPipeline:
    def __init__(self):
        self.utils = MainUtils()
        
    def prepare_input_data(self, input_data:list) -> pd.DataFrame:
        """ 
        method: prepare_input_data 
        
        objective: This method creates a dataframe taking the column names from prediction schema file
                       with the input values for prediction and returns it

        Args:
            input_data (list): input data 

        Raises:
            CustomerException

        Returns:
            customerDataframe: pd.DataFrame: a dataframe containing the input values
        """
        try:
        
            
            customerDataframe = CustomerData.form_input_dataframe(data = input_data)
            # Handle potential NaNs by filling with 0 (consistent with trainer config)
            customerDataframe.fillna(0, inplace=True)
            logging.info("customerDatafram has been created")
            return customerDataframe
        except Exception as e:
            raise CustomerException(e,sys)
        
   
        
    
        
    def get_trained_model(self, ModelTrainerConfig = ModelTrainerConfig):
        """
        method: get_trained_model
        
        objective: this method returns the model

        Args:
            ModelTrainerConfig

        Raises:
            CustomerException: 

        Returns:
            model: latest trained model
        """
        try:
            prediction_config = PredictionPipelineConfig()
            model = CustomerClusterEstimator(
                model_path= prediction_config.local_model_path
            )
                
            return model
                
        except Exception as e:
            raise CustomerException(e, sys) from e
        
    def run_pipeline(self, input_data:list):
        
        """
        method: run_pipeline
        
        objective: run_pipeline method runs the whole prediction pipeline.

        Raises:
            CustomerException: 
        """
        try:
            input_dataframe =  self.prepare_input_data(input_data) 
            model = self.get_trained_model()
            prediction = model.predict(input_dataframe)
            return prediction
            
        except Exception as e:
            raise CustomerException(e, sys)

    def prepare_batch_data(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        method: prepare_batch_data
        objective: Replicate feature engineering from DataTransformation.get_new_features
        """
        try:
            from datetime import datetime
            
            # 1. Age
            if 'Year_Birth' in dataset.columns:
                dataset['Age'] = datetime.today().year - dataset['Year_Birth']
            
            # 2. Education mapping
            if 'Education' in dataset.columns:
                dataset["Education"].replace({"Basic":0,"2n Cycle":1, "Graduation":2, "Master":3, "PhD":4}, inplace=True)
            
            # 3. Marital Status mapping
            if 'Marital_Status' in dataset.columns:
                dataset['Marital Status'] = dataset['Marital_Status'].replace({"Married":1, "Together":1, "Absurd":0, "Widow":0, "YOLO":0, "Divorced":0, "Single":0,"Alone":0})
            
            # 4. Children & Parental Status
            if 'Kidhome' in dataset.columns and 'Teenhome' in dataset.columns:
                dataset['Children'] = dataset['Kidhome'] + dataset['Teenhome']
                dataset["Parental Status"] = np.where(dataset["Children"] > 0, 1, 0)

            # 5. Total Spending
            spending_cols = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
            if all(col in dataset.columns for col in spending_cols):
                dataset['Total_Spending'] = dataset[spending_cols].sum(axis=1)
                # Rename for model consistency
                dataset.rename(columns={
                    "MntWines": "Wines", "MntFruits":"Fruits", "MntMeatProducts":"Meat",
                    "MntFishProducts":"Fish", "MntSweetProducts":"Sweets", "MntGoldProds":"Gold"
                }, inplace=True)

            # 6. Total Promo
            promo_cols = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5"]
            if all(col in dataset.columns for col in promo_cols):
                dataset["Total Promo"] = dataset[promo_cols].sum(axis=1)

            # 7. Days as Customer
            if 'Dt_Customer' in dataset.columns:
                dataset['Dt_Customer'] = pd.to_datetime(dataset['Dt_Customer'], format="%d-%m-%Y", errors="coerce")
                dataset['Days_as_Customer'] = (datetime.today() - dataset['Dt_Customer']).dt.days

            # 8. Channel Renaming
            dataset.rename(columns={
                "NumWebPurchases": "Web", "NumCatalogPurchases":"Catalog",
                "NumStorePurchases":"Store", "NumDealsPurchases":"Discount Purchases",
                "NumWebVisitsMonth": "NumWebVisitsMonth"
            }, inplace=True)

            # 9. Column selection and ordering as expected by preprocessor
            final_cols = [
                "Age","Education","Marital Status","Parental Status",
                "Children","Income","Total_Spending","Days_as_Customer",
                "Recency","Wines","Fruits","Meat","Fish","Sweets","Gold",
                "Web","Catalog","Store","Discount Purchases","Total Promo",
                "NumWebVisitsMonth"
            ]
            
            # Filter for only those columns that exist in dataset and fill missing columns with 0
            for col in final_cols:
                if col not in dataset.columns:
                    dataset[col] = 0
            
            # Select final columns and fill potential NaNs in existing data with 0
            final_df = dataset[final_cols].fillna(0)
            
            return final_df

        except Exception as e:
            raise CustomerException(e, sys)

    def predict_batch(self, input_dataframe: pd.DataFrame):
        """
        method: predict_batch
        
        objective: predict_batch method runs the prediction for a whole dataframe.
        """
        try:
            # Preprocess the raw CSV data to match training features
            prepared_df = self.prepare_batch_data(input_dataframe)
            
            model = self.get_trained_model()
            prediction = model.predict(prepared_df)
            return prediction
        except Exception as e:
            raise CustomerException(e, sys)
            
            
        
            
        

 
        

        