import json
import os
from Logger.logger import Applogger

class Train_file_schema_validate:
    """
    Class is used to cross verify schema of passed training files with the expected schema for training files.
    If any of the files are not matched - Bad Files, then the file will be sent to rejected folder where it will be archived
    """
    def __init__(self, train_batch_file_path):
        """
        Creates an Instance of Train_data_validate class which is used to call defined functions at the time of validation.py
        :param batch_file_path: path of batch training files
        """
        self.train_batch_file_path = train_batch_file_path    # train_batch_file_path
        self.log_agent = Applogger()                       # Logger instance
        self.train_schema_path = 'Data_Validation/files_schema/schema_training.json'    # train schema path

        self.train_file_schema_log_filepath = 'Logs/train_file_schema_validate.txt'

    def getValuesFromSchema(self):
        try:
            # Open json file and convert the data in it into dicitionary
            with open(self.train_schema_path, 'r') as file:
                dic = json.load(file)
            file.close()

            # Extracting data from dictionary
            accepted_pattern = dic['SampleFileName']
            accepted_len_date = dic['LengthOfStampDate']
            accepted_len_time = dic['LengthOfStampTime']
            accepted_no_cols = dic['NumberOfColumns']

            # Logging the obtained values in train_file_schema_validate.txt

            log_file = open(self.train_file_schema_log_filepath, 'a+')

            message = 'Values obtained from Schema :- SampleFileName: ' + str(accepted_pattern) + '\t' + 'LengthOfStampDate: ' + str(accepted_len_date) + '\t' + 'LengthOfStampTime: ' + str(accepted_len_time) + '\t' + 'NumberOfColumns: '+str(accepted_no_cols) + '\n'

            self.log_agent.log(log_file, message)

            log_file.close()

        except Exception as e:
            log_file= open(self.train_file_schema_log_filepath, 'a+')

            message = 'Error in Obtaining values from '+self.train_schema_path + str(e)

            self.log_agent(log_file, message)

        return  accepted_pattern, accepted_len_date,accepted_len_time,accepted_no_cols


temp = Train_file_schema_validate('temp')
temp.getValuesFromSchema()