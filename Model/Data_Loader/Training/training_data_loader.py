import pandas as pd
from Logger.logger import Logger

class Training_data_loader:
    def __init__(self):
        self.train_ready_file_path = "Final_Training_CSV_File/ready_to_train_file.csv"

        # creating log
        self.log_agent = Logger()
        self.train_data_loader_log = "Logs/Model_Logs/train_data_loader_log.txt"

    def getTrainingDataFrame(self):
        try:
            log_file = open(self.train_data_loader_log, 'a+')
            self.log_agent.log(log_file, "Retrieving training file {}".format(self.train_ready_file_path))

            df = pd.read_csv(self.train_ready_file_path) # obtaining csv -> df

            if df.shape[0] != 0:
                self.log_agent.log(log_file, "DataFrame obtained successfully.")
                return df
            else:
                raise Exception('Error occurred while reading file {}'.format(self.train_ready_file_path))
        except Exception as e:
            self.log_agent.log(log_file, str(e))