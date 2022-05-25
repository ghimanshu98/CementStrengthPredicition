import pandas as pd
from Logger.logger import Logger

class Prediction_data_loader:
    def __init__(self):
        self.predict_ready_file_path = "Final_CSV_File/Prediction/ready_csv_file.csv"

        # creating log
        self.log_agent = Logger()
        self.predict_data_loader_log = "Logs/Prediction_Logs/prediction_data_loader_log.txt"

    def getPredictionDataFrame(self):
        try:
            log_file = open(self.predict_data_loader_log, 'a+')
            self.log_agent.log(log_file, "Retrieving Prediction file {}".format(self.predict_ready_file_path))

            df = pd.read_csv(self.predict_ready_file_path)  # obtaining csv -> df

            if df.shape[0] != 0:
                self.log_agent.log(log_file, "DataFrame obtained successfully.")
                return df
            else:
                raise Exception('Error occurred while reading file {}'.format(self.predict_ready_file_path))
        except Exception as e:
            self.log_agent.log(log_file, str(e))