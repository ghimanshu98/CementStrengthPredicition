import pandas as pd
from Logger.logger import Logger

class Predictio_ndata_loader:
    def __init__(self):
        self.train_ready_file_path = "Final_Prediction_CSV_File"    ### incomplete

        # creating log
        self.log_agent = Logger()
        self.train_data_loader_log = "Logs/Model_Logs/Prediction_data_loader_log.txt"

    def getTrainingDataFrame(self):
        try:
            log_file = open(self.preprocessing_log_file_path, 'a+')
            self.log_agent.log(log_file, "Retrieving training file {}".format(self.train_ready_file_path))

            df = pd.read_csv(self.train_ready_file_path)  # obtaining csv -> df

            if df != None:
                self.log_agent.log(log_file, "DataFrame obtained successfully.")
                return df
            else:
                raise Exception('Error occurred while reading file {}'.format(self.train_ready_file_path))
        except Exception as e:
            self.log_agent.log(log_file, str(e))