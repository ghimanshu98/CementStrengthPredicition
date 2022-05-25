import json
import os
from Logger.logger import Logger
import re
import shutil
import pandas as pd

class Train_file_schema_validate:
    
    """
    Class is used to cross verify schema of passed training files with the expected schema for training files.
    If any of the files are not matched - Bad Files, then the file will be sent to rejected folder where it will be archived
    """
    def __init__(self, train_batch_file_path):
        """
        Creates an Instance of Train_schema_validate class which is used to call defined functions at the time of validation.py
        :param batch_file_path: path of batch training files
        """
        self.train_batch_file_path = train_batch_file_path    # train_batch_file_path
        self.log_agent = Logger()                       # Logger instance
        self.train_schema_path = 'File_Schema_Validation/files_schema/schema_training.json'    # train schema path

        self.train_file_schema_log_filepath = 'Logs/Training_Logs/train_file_schema_validate.txt'

    def getValuesFromSchema(self):
        """
        :Description: This method returns the meta information about batch files from schema files.
        :return: accepted_pattern, accepted_len_date, accepted_len_time, accepted_no_cols
        """
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

    def regex_pattern(self):
        """
        :description: Returns the regex pattern
        :return: RegEx pattern for file name"""
        reg = "['cement_strength']+['\_']+['\d']+['\_']+['\d']+\.csv"   # d- signifies numbers
        return reg

    def createGoodFileSchemaDataDirectory(self):
        """
        :description: Internal Function for creating GoodFileSchemaDataDirectory - not to be used alone.
        """
        dir_path = 'Validated_Training_Batch_Files/GoodFileSchemaDataFolder'
        try:
            log_file = open(self.train_file_schema_log_filepath, 'a+')

            # creating dir
            os.mkdir(dir_path)

            # logging in log file
            message = 'Directory ' + dir_path+' Created successfully'
            self.log_agent.log(log_file, message)

            # Closing log file
            log_file.close()

            return dir_path
        except Exception as e:
            log_file = open(self.train_file_schema_log_filepath, 'a+')
            message = 'Directory creation unsuccessfull, error '+ str(e)
            self.log_agent.log(log_file, message)
            log_file.close()

    def createBadFileSchemaDataDirectory(self):
        """
        :description: Internal Function for creating BadFileSchemaDataDirectory - not to be used alone.
        """
        dir_path = 'Validated_Training_Batch_Files/BadFileSchemaDataFolder'
        try:
            log_file = open(self.train_file_schema_log_filepath, 'a+')

            # creating dir
            os.mkdir(dir_path)

            # logging in log file
            message = 'Directory ' + dir_path+' Created successfully'
            self.log_agent.log(log_file, message)

            # Closing log file
            log_file.close()

            return dir_path
        except Exception as e:
            log_file = open(self.train_file_schema_log_filepath, 'a+')
            message = 'Directory creation unsuccessfull, error '+ str(e)
            self.log_agent.log(log_file, message)
            log_file.close()

    def deleteGooDBadFileSchemaDataFolders(self):
        """
        :description: Internal Function for deleting Good_Bad_FileSchemaDataDirectory - not to be used alone.
        """
        try:
            log_file = open(self.train_file_schema_log_filepath,'a+')

            parent_folder = 'Validated_Training_Batch_Files/'

            folder = os.listdir(parent_folder)
            if len(folder) == 0:
                self.log_agent.log(log_file, "No GoodBadFileSchemaDataFolders present.")
                log_file.close()
            else:
                for dir in folder:
                    shutil.rmtree(parent_folder+str(dir))
                
                self.log_agent.log(log_file," GooDBadFileSchemaDataFolders deleted successfully.")
                log_file.close()

        except Exception as e:
            log_file = open(self.train_file_schema_log_filepath, 'a+')

            self.log_agent.log(log_file, 'Error occurred while deleting folders '+ str(e))

    def validate_file_name(self, regex, LenOfFileDate, LenOfFileTime):
        """
        :description: Method is used to validate batch file names and it cretess a copy of the correct mathcing name file in directory Validated_Training_Batch_Files/GoodFileSchemaDataFolder and forrejected files in dir Validated_Training_Batch_Files/BadFileSchemaDataFolder.
        :param regex: takes in the allowed regec pattern for matching with file names
        :param LenOfFileDate: len of characters specifying Date in file name
        :param LenOfFileTime: len of characters specifying Time in file name
        """
        
        # deleting Good_BadFileSchema directory in start to avoid any issue that might be caused due to some previous unsuccessful run.

        self.deleteGooDBadFileSchemaDataFolders() 

        # Create Good_BadFileSchema directory to hold good and bad files
        good_dir_path = self.createGoodFileSchemaDataDirectory()
        bad_dir_path = self.createBadFileSchemaDataDirectory()

        # Obtaining the list of files in Training_Batch_Files direcctory
        train_batch_files = os.listdir(self.train_batch_file_path)

        try:
            log_file = open(self.train_file_schema_log_filepath, 'a+')
            for filename in train_batch_files:
                # validate pattern
                if(re.match(regex, filename)):
                    # splitting to get date and time
                    splitAtDot = re.split('.csv',filename)
                    splitAtUnderScore = re.split('_', splitAtDot[0])

                    # comparing lenght of date
                    if(len(splitAtUnderScore[2])== LenOfFileDate):
                        # comparing length of time
                        if(len(splitAtUnderScore[3])== LenOfFileTime):
                            # copy the file t Good File Schema_Data path
                            shutil.copy(self.train_batch_file_path+'/'+filename, good_dir_path)

                            # Log the change
                            message = filename +'is validated and accepted, Copying it from '+ self.train_batch_file_path+'/'+filename+ 'to' + good_dir_path + '\n'
                            self.log_agent.log(log_file, message)

                        else:
                            # copy the file to Bad File Schema_Data path
                            shutil.copy(self.train_batch_file_path+'/'+filename, bad_dir_path)

                            # Log the change
                            message = filename +' is rejected, Copying it from '+ self.train_batch_file_path+'/'+filename+ ' to ' + bad_dir_path + '\n'
                            self.log_agent.log(log_file, message)
                    
                    else:
                        # copy the file to Bad File Schema_Data path
                        shutil.copy(self.train_batch_file_path+'/'+filename, bad_dir_path)

                        # Log the change
                        message = filename +' is rejected, Copying it from '+ self.train_batch_file_path+'/'+filename+ ' to ' + bad_dir_path + '\n'
                        self.log_agent.log(log_file, message)
                
                else:
                        # copy the file to Bad File Schema_Data path
                        shutil.copy(self.train_batch_file_path+'/'+filename, bad_dir_path)

                        # Log the change
                        message = filename +' is rejected, Copying it from '+ self.train_batch_file_path+'/'+filename+ ' to ' + bad_dir_path + '\n'
                        self.log_agent.log(log_file, message)

            # closing log file
            log_file.close()
        except Exception as e:
            log_file = open(self.train_file_schema_log_filepath, 'a+')
            message = 'Exception occurred while Validating Training_batch_files schema ' + str(e)
            self.log_agent.log(log_file, message)
            log_file.close()

    def validate_no_cols(self, accepted_no_cols):
        """
        :description: Method is used to validate umber of cols in each batch file and it keeps a the correct file in directory Validated_Training_Batch_Files/GoodFileSchemaDataFolder and for rejected files it moves them to directory  Validated_Training_Batch_Files/BadFileSchemaDataFolder.
        :param accepted_no_cols: Number of cols accepted
        """
        try:
            good_file_schema_data_dir_path = 'Validated_Training_Batch_Files/GoodFileSchemaDataFolder' 
            filenames = os.listdir(good_file_schema_data_dir_path)
            for file in filenames:
                csv = pd.read_csv(good_file_schema_data_dir_path+"/"+file)
                cols_in_file = len(csv.columns)
                if cols_in_file == accepted_no_cols:
                    log_file = open(self.train_file_schema_log_filepath, 'a+')
                    message = file+" contains "+ str(cols_in_file)+" columns which is equal to " + str(accepted_no_cols) + " columns, so file is good."

                    self.log_agent.log(log_file, message)
                    log_file.close()
                else:
                    shutil.move(good_file_schema_data_dir_path+'/'+file, 'Validated_Training_Batch_Files/BadFileSchemaDataFolder')
                    
                    #logging the change
                    log_file = open(self.train_file_schema_log_filepath, 'a+')

                    message = file+" contains "+ str(cols_in_file)+" columns which is less than accepeted " + str(accepted_no_cols) + " columns, so moving file to Validated_Training_Batch_Files/BadFileSchemaDataFolder"

                    self.log_agent.log(log_file, message)
                    log_file.close()
        except Exception as e:
            log_file = open(self.train_file_schema_log_filepath, 'a+')
            message = "Error occurred while Validating no. of cols : "+ str(e)

            self.log_agent.log(log_file, message)
            log_file.close()
        
    def validate_missing_values(self):
        """
        :description: Method is used to validate missing values in each batch file and it keeps a the correct file in directory Validated_Training_Batch_Files/GoodFileSchemaDataFolder and for rejected files it moves them to directory  Validated_Training_Batch_Files/BadFileSchemaDataFolder.
        :param accepted_no_cols: Number of cols accepted
        """
        try:
            good_file_schema_data_dir_path = 'Validated_Training_Batch_Files/GoodFileSchemaDataFolder' 
            filenames = os.listdir(good_file_schema_data_dir_path)
            for file in filenames:
                csv = pd.read_csv(good_file_schema_data_dir_path+'/'+file)
                for columns in csv:
                    if (len(csv[columns]) != csv[columns].count()):
                        # if col has missing values then move the file to bad file folder
                        shutil.move(good_file_schema_data_dir_path+'/'+file, 'Validated_Training_Batch_Files/BadFileSchemaDataFolder')
                        
                        # logging the change
                        log_file = open(self.train_file_schema_log_filepath, 'a+')
                        message = "Invalid length of feature found in file "+ file + "Moving file to  Validated_Training_Batch_Files/BadFileSchemaDataFolder"
                        self.log_agent.log(log_file, message)
                        log_file.close()
                        break
                # logging the change
                log_file = open(self.train_file_schema_log_filepath, 'a+')
                message = "No missing values found in file "+ file
                self.log_agent.log(log_file, message)
                log_file.close()

        except Exception as e:
            log_file = open(self.train_file_schema_log_filepath, 'a+')
            message = "Error occurred while checking for missing values " + str(e)
            self.log_agent.log(log_file, message)
            log_file.close()

    def reset_log(self):
        try:
            os.remove(self.train_file_schema_log_filepath)
            file = open(self.train_file_schema_log_filepath, 'a+')
            file.close()
        except Exception as e:
            print('Error while resetting '+ self.train_file_schema_log_filepath)
