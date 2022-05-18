import numpy as np
from Logger.logger import Logger
from sklearn.preprocessing import StandardScaler
from Model.Saved_models.model_saver import SaveModel


class DataPreProcessing:
    def __init__(self):

        # logger object
        self.log_agent = Logger()
        
        # preprocessing_log_file_path
        self.preprocessing_log_file_path = "Logs/Model_Logs/preprocessing_log_file.txt" 

        # model saver object
        self.model_saver = SaveModel()


    def removCols(self, dataframe, col_names):
        try:
            log_file = open(self.preprocessing_log_file_path, 'a+')
            if len(col_names) != 0:
                dataframe = dataframe.drop(col_names, axis = 1)
                self.log_agent.log(log_file, "Following columns {} got dropped from {} successfully.".format(col_names, dataframe))
                log_file.close()
                return dataframe
        except Exception as e:
            self.log_agent.log(log_file, "Error occurred during removing columns from datafram. "+str(e))
            log_file.close()

    def containsNull(self, dataframe):
        try:
            log_file = open(self.preprocessing_log_file_path, 'a+')
            contain_null = []
            for col in dataframe.columns:
                if dataframe[col].isnull().sum() !=0:
                    contain_null.append(col)
            
            if len(contain_null) != 0:
                self.log_agent.log(log_file, "DataFrame contains following null colums {}".format(contain_null))
            else:
                self.log_agent.log(log_file, "DataFrame contains 0 null colums {}")
            log_file.close()
            return contain_null
        except Exception as e:
            self.log_agent.log(log_file, "Error occurred during checking Null values in colusmns. "+str(e))
            log_file.close()

    def getColNames(self, dataframe):
        try:
            log_file = open(self.preprocessing_log_file_path, 'a+')
            cols = dataframe.columns
            self.log_agent.log(log_file, "Dataframe contains following cols {}".format(cols))
            log_file.close()
            return cols
        except Exception as e:
            self.log_agent.log(log_file, "Error : "+str(e))
            log_file.close()

    def separateDependentIndependentFeatures(self, dataframe, dependent_feature):
        try:
            log_file = open(self.preprocessing_log_file_path, 'a+')
            X = dataframe.drop([dependent_feature], axis = 1)
            Y = dataframe[dependent_feature]
            self.log_agent.log(log_file, "Dependent and Independent features separated successfully.")
            log_file.close()
            return X,Y
        except Exception as e:
            self.log_agent(log_file, "Error occurred during separating features, "+str(e))
            log_file.close()

    def convertToLogNormalForm(self, dataframe, independent_feature):
        """
        :param independent_feature: takes list of independents column names
        """
        try:
            log_file = open(self.preprocessing_log_file_path, 'a+')
            self.log_agent.log(log_file, "Starting Log Conversions for Dataframe")
            for cols in independent_feature:
                dataframe[cols] += 1  # to remove infinity value issue
                dataframe[cols] = np.log(dataframe[cols])
            self.log_agent.log(log_file, "Log Conversions for Dataframe completed")
            log_file.close()
            return dataframe
        except Exception as e:
            self.log_agent.log(log_file, "Error occurred during Log conversion")
            log_file.close()

    def low_upp_fence(self, dataframe, feature_name, const = 1.5):
        try:
            log_file = open(self.preprocessing_log_file_path, 'a+')
            q1 = dataframe[feature_name].quantile(0.25)
            q3 = dataframe[feature_name].quantile(0.75)
            iqr = q3-q1
            lower_fence = q1 - iqr*const
            upper_fence = q3 + iqr*const
            self.log_agent.log(log_file, "Lower Fence : {} and Upper Fence : {} for feature {}".format(lower_fence, upper_fence, feature_name))
            log_file.close()
            return (lower_fence, upper_fence)
        except Exception as e:
            self.log_agent.log(log_file, "Error occurred during calculation of low_upp_fence for feature : {}".format(feature_name))
            log_file.close()


    def handleOutliers(self, dataframe):
        try:
            log_file = open(self.preprocessing_log_file_path, 'a+')
            self.log_agent.log(log_file, "Starting handling outlier process.")
            with_outlier_length = dataframe.shape[0]
            
            for cols in dataframe.columns:
                low , upp = self.low_upp_fence(dataframe[cols], cols)
                dataframe = dataframe.loc[dataframe[cols] >=low].loc[dataframe[cols] <= upp]
            without_outlier_length = dataframe.shape[0]
            
            self.log_agent.log(log_file, "Handled Ouliers , Rows containing outlier values dropped are {}".format(with_outlier_length - without_outlier_length))

            log_file.close()

        except Exception as e:
            self.log_agent.log(log_file, "Error occurred while handling outlier "+ str(e))
            log_file.close()


    def standardScaler(self, X_train, X_test = None):
        try:
            log_file = open(self.preprocessing_log_file_path, 'a+')
            self.log_agent.log(log_file, "Starting standardization of data")
           
            # creating object of standard scaler
            scaler = StandardScaler()
            if X_test != None:
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                # logging
                self.log_agent.log(log_file, "Scailing completed.")
                log_file.close()
                # saving the scaler model
                self.model_saver.save_model(scaler, "scaler_model")
                return (X_train_scaled, X_test_scaled)
            else:
                X_train_scaled = scaler.fit_transform(X_train)
                self.log_agent.log(log_file, "Scailing completed.")
                log_file.close()
                # saving the scaler model
                self.model_saver.save_model(scaler, "scaler_model")
        except Exception as e:
            self.log_agent.log(log_file, "Error occurred while standardizing the data, "+str(e))
            log_file.close()


            

            






             

