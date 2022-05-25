from Logger.logger import Logger
from File_Schema_Validation.Prediction_File_Schema_Validation_utils.prediction_file_schema_validate_utils import Predict_file_schema_validate
from Data_Transforamation_For_DB.data_transformation_for_db import DataTransformation
from DbOperations.db_operations import DbOperations
from Model.Data_Loader.Prediction.prediction_data_loader import Prediction_data_loader
from Model.Data_Pre_Processing.preprocessing import DataPreProcessing
from Model.Saved_models.model_saver import SaveModel
import os
import pandas as pd

class InitiatePredictonProcess:
    def __init__(self, predict_batch_files_path):
        # creaing log agent
        self.log_agent = Logger()

        # log file path
        self.initiate_prediction_log_file_path = 'Logs/Prediction_Logs/initiate_prediction_log_file.txt'

        # Predict batch File path
        self.predict_batch_files_path = predict_batch_files_path

        # GoodDataFilePath
        self.good_data_file_path = 'Validated_Prediction_Batch_Files/GoodFileSchemaDataFolder'

        # prediction save file path
        self.prediction_save_file_path = 'Prediction_csv/'

    def validate(self):
        try:
            log_file = open(self.initiate_prediction_log_file_path, 'a+')
            self.log_agent.log(log_file, "Initiate validation for Prediction Batch Files")

            # Creating object of Validation calss
            self.log_agent.log(log_file, "Creating object of class Train_file_schema_validate")
            predict_file_schema_validate_obj = Predict_file_schema_validate(self.predict_batch_files_path)

            # Fetching meta properties of training batch files
            self.log_agent.log(log_file, "Fetching meta properties of training batch files")
            accepted_pattern, accepted_len_date, accepted_len_time, accepted_no_cols = predict_file_schema_validate_obj.getValuesFromSchema()

            # Fetching regex for validating file name
            self.log_agent.log(log_file, "Fetching regex pattern for file name")
            re_pattern = predict_file_schema_validate_obj.regex_pattern()

            # validating file names
            self.log_agent.log(log_file, "Validating file names of training batch files")
            predict_file_schema_validate_obj.validate_file_name(re_pattern, accepted_len_date, accepted_len_time)

            # validate no of cols in each file
            self.log_agent.log(log_file, "Validating no of cols of training batch files")
            predict_file_schema_validate_obj.validate_no_cols(accepted_no_cols)

            # validate mssing values in each file
            self.log_agent.log(log_file, "Validating missing values of training batch files")
            predict_file_schema_validate_obj.validate_missing_values()
            self.log_agent.log(log_file, "Validation of training batch files completed")

            log_file.close()
        except Exception as e:
            self.log_agent.log(log_file, "Exception occurred in validate() method of InitiateTask class, "+str(e))
            log_file.close()

    def perform_data_transformation_for_db(self):
        """
        :description: Performs data transformation_ for storing it in Database.
		"""
        try:
            log_file =open(self.initiate_prediction_log_file_path, 'a+')
            self.log_agent.log(log_file, "Starting data transformation process for GoodDataFiles.")
            # creating obj of DataTransformation
            self.log_agent.log(log_file, "Creating object of class DataTransformation")
            transformation_obj = DataTransformation(training= False)

            # performing data transformation
            # self.log_agent.log(log_file, "Calling transform_data_for_db() of DataTransformation class.")
            transformation_obj.transform_data_for_db(self.good_data_file_path)
            self.log_agent.log(log_file, "Data transformation completed.")
            log_file.close()
        except Exception as e:
            self.log_agent.log(log_file, "Exception occurred in perform_data_transformation_for_db() method of DataTransformation class, "+str(e))
            log_file.close()

    def perform_Db_operations(self, db_path, tableName, training = False):
        """
		:description: Stores validated data into Training.db and  creates a acsv file.
		:param db_path: Database path
		:param tableName: Table Name
		:param training: Boolean if True, object will be used for training purpose.
		"""
        try:
            log_file =open(self.initiate_prediction_log_file_path, 'a+')
            self.log_agent.log(log_file, "Starting DbOperations.")
            
            # creating object of DbOperation
            self.log_agent.log(log_file, "Creating object of DbOperations.")
            db_object = DbOperations(db_path, tableName, training)
            
            # get column details from schema file
            self.log_agent.log(log_file, "Fetching column details from schema file.")
            column_details = db_object.getColumnDetails()

            # crete table in db
            self.log_agent.log(log_file, "Creating table: {} in db: {}.".format(tableName, db_path))
            db_object.createTableInDb(column_details)

            # inserting records in db
            self.log_agent.log(log_file, "Inserting records in table : {}".format(tableName))
            db_object.insertFilesInDB()

            # getting csv file from records in db
            self.log_agent.log(log_file, "Fetching records from table : {} in db {}".format(tableName, db_path))
            db_object.getRowsFromDb()

            self.log_agent.log(log_file, "DbOperations Completed.")
            log_file.close()
        except Exception as e:
            self.log_agent.log(log_file, "Exception occured while performing dbOperations, "+str(e))
            log_file.close()

    def preprocess_and_initiate_prediction(self):
        try: 
            log_file = open(self.initiate_prediction_log_file_path, 'a+')
            self.log_agent.log(log_file, "Starting Preprocessing and Prediction of data.")

            # creating object of Training_data_loader
            self.log_agent.log(log_file, "Creating object of Training_data_loader")
            predict_data_loader_obj = Prediction_data_loader()
            
            # Fetching Data Frame
            self.log_agent.log(log_file, "Fetching DataFrame")
            df = predict_data_loader_obj.getPredictionDataFrame()

            # creating object of DataPreprocessing class.
            self.log_agent.log(log_file, "Creating object of DataPreprocessing class.")
            preprocessor = DataPreProcessing()

            # Checking for Null Values in Df
            self.log_agent.log(log_file, "Checking for Null Values in Df")
            preprocessor.containsNull(df)

            # Fetching column names
            self.log_agent.log(log_file, "Fetching column names")
            all_cols = list(preprocessor.getDfColNames(df))
            
            # Converting to Log Normal Form
            self.log_agent.log(log_file, "Converting to Log Normal Form")
            df = preprocessor.convertToLogNormalForm(df, all_cols)

            # Scailing data frame
            # Creating instance of SaveModel() to load model
            model_obj = SaveModel()

            # Loading the Scaler model

            scalar = model_obj.load_model('scaler_model.sav')
            self.log_agent.log(log_file, "Scailing data frame")
            x_pred_scaled = scalar.transform(df)
            # X_scaled

            # Clustering
            self.log_agent.log(log_file, "Initiating cluster prediction operation on prediction data")

            #Loading Cluster model
            cluster_model = model_obj.load_model('Kmeans_cluster.sav')

            # predicting clusters
            cluster_pred = cluster_model.predict(x_pred_scaled)

            # self.log_agent.log(log_file, "Reshaping Clusters array to merge later with scaled data")
            reshaped_cluster_pred = preprocessor.reshape_array(data = cluster_pred, shape = (len(cluster_pred), 1))
            # # reshaped_cluster_y
            
            # column_names.append('Cluster')
            all_cols.append('Cluster')
            self.log_agent.log(log_file, "Concatenating all arrays : x_pred_scaled and  cluster_pred")
            new_array = preprocessor.concatenate_array([x_pred_scaled, reshaped_cluster_pred], axis = 1)

            self.log_agent.log(log_file, "Creating DataFrame of above created new_array")
            df = preprocessor.make_df(new_array,columns= all_cols)
            
            # Predicting the result
            self.log_agent.log(log_file, "Starting Predictions")
            # obtaining the model names

            self.log_agent.log(log_file, "Obtaining the model names")
            model_names = os.listdir(model_obj.save_model_path)  

            self.log_agent.log(log_file, "Loading the models to a dictionary")
            model_dict = {}
            for i,name in enumerate(model_names):
                model_dict[str(name)] = model_obj.load_model(name)

            self.log_agent.log(log_file, "Doing Predictions")

            final_prediction = []
            for i in range(df.shape[0]):
                clust = df['Cluster'][i]  # obtaining the cluster value
                for name in model_names: 
                    if name.find(str(clust)) > -1:  # searching for name in model list
                        temp_load = model_dict[name] # loading the model from dict
                        pred = temp_load.predict([df.iloc[i, :8]])[0] # Prdecition -- giving error
                        self.log_agent.log(log_file, "Cluster: {}, Model_name: {}, Pred: {} ".format(clust, name, pred))
                        final_prediction.append(pred)

            # Fetching Data Frame and compiling with the Predictions made
            self.log_agent.log(log_file, "Fetching DataFrame")
            df = predict_data_loader_obj.getPredictionDataFrame()
            df['Predicted_Cement_Compressive_Strength'] = final_prediction
            
            self.log_agent.log(log_file, "Predictions Completed")

            self.log_agent.log(log_file, "Saving the final file.")
            self.savePrediction(df)
            log_file.close()
        except Exception as e:
            self.log_agent.log(log_file,'Exception Occured in preprocess_and_initiate_predicion() method in InitiateProcess.'+str(e))
            log_file.close()

    def createSeries(self, data):
        try:
            log_file = open(self.initiate_prediction_log_file_path, 'a+')
            temp = pd.Series(data)
            self.log_agent.log(log_file, "Data converted to Series successfully")
            log_file.close()
            return temp
        except Exception as e:
            self.log_agent.log(log_file, "Exception Occured while converting to data to Series, "+str(e))
            log_file.close()

    def savePrediction(self, df):
        try:
            log_file = open(self.initiate_prediction_log_file_path, 'a+')

            df.to_csv(self.prediction_save_file_path+"prediction.csv", index = False)

            self.log_agent.log(log_file, "DataFrame converted to CSV successfully and stored at {} ".format(self.prediction_save_file_path))
            log_file.close()
        except Exception as e:
            self.log_agent.log(log_file, "Exception Occured while saving Prediction file, "+str(e))
            log_file.close()
            
            




    
