from Logger.logger import Logger
import os
import pandas as pd

class DataTransformation:
    def __init__(self, training = True):
        """
        :description: Retuns an instance of DataTransformation class.
        :param training: Boolean, if True, creates instance for Training else if False creates for Prediction"""
        # creating logger instance
        self.log_agent = Logger()

        if training:
            self.db_ingestion_ready_file_path = 'Db_Ingestion_Ready_Files/Training/' 
            self.datatransformation_log_file_path = 'Logs/Training_Logs/data_for_db_transformation_log_file_path.txt'
        else:
            self.db_ingestion_ready_file_path = 'Db_Ingestion_Ready_Files/Prediction/'
            self.datatransformation_log_file_path = 'Logs/Prediction_Logs/data_for_db_transformation_log_file_path.txt'


    def transform_data_for_db(self, validatedGoodDataFilePath):
        """
        :description: Ready the data for storing in database.
        :param validatedGoodData: Picks the files from supplied file path and stores it in directory Db_Ingestion_Ready_Files/ """
        try:
            log_file = open(self.datatransformation_log_file_path, 'a+')

            filename = os.listdir(validatedGoodDataFilePath)

            for file in filename:
                csv = pd.read_csv(validatedGoodDataFilePath+'/'+file)
                csv.fillna('NULL', inplace=True)

                csv.to_csv(self.db_ingestion_ready_file_path+file, index=None, header=True)

                # Logging details
                message = file + " transformed successfully and stored in "+self.db_ingestion_ready_file_path
                
                self.log_agent.log(log_file, message)
                log_file.close()

        except Exception as e:
            log_file = open(self.datatransformation_log_file_path, 'a+')

            message = "Error occurred while transforming file "+str(e)

            self.log_agent.log(log_file, message)
            log_file.close()

