import numpy as np
from Logger.logger import Logger
from sklearn.preprocessing import StandardScaler
from Model.Saved_models.model_saver import SaveModel
from sklearn.impute import KNNImputer
import pandas as pd
import json
from sklearn.model_selection import train_test_split


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

    def imputeMissingData(self, dataframe):
        try:
            log_file = open(self.preprocessing_log_file_path, 'a+')
            self.log_agent.log(log_file, "Starting Imputing process")
            
            # imputer object:
            imputer = KNNImputer(n_neighbors= 3, missing_values= np.nan, weights= "uniform")
            new_array = imputer.fit_transform(dataframe)
            self.model_saver.save_model(imputer, "KNN_imputer")
            # creating dataFrame using new array
            dataframe = pd.DataFrame(new_array, columns = dataframe.columns)

            self.log_agent.log(log_file, "Data imputaion succesfullly completed")
            log_file.close()
        except Exception as e:
            self.log_agent.log(log_file, "Error occurred during imputaion process, "+str(e))
            log_file.close()


    def getDfColNames(self, dataframe):
        try:
            log_file = open(self.preprocessing_log_file_path, 'a+')
            cols = dataframe.columns
            self.log_agent.log(log_file, "Dataframe contains following cols {}".format(cols))
            log_file.close()
            return cols
        except Exception as e:
            self.log_agent.log(log_file, "Error : "+str(e))
            log_file.close()

    def getColsNames(self, schema_file_path):
        try:
            log_file = open(self.preprocessing_log_file_path, 'a+')
            with open(schema_file_path, 'r') as f:
                dic = json.load(f)
                cols = list(dic['Col_Name'].keys())
                self.log_agent.log(log_file, "Col Names {} obtained successfully froms schema {}".format(cols, schema_file_path))
                log_file.close()
                return cols
        except Exception as e:
            self.log_agent.log(self.preprocessing_log_file_path, "Error while retreiving column names form schema {} ".format(schema_file_path)+str(e))
            #self.log_agent.log(log_file, str(e))
            log_file.close()


    def separateDependentIndependentFeatures(self, dataframe, dependent_feature):
        try:
            log_file = open(self.preprocessing_log_file_path, 'a+')
            X = dataframe.drop(dependent_feature, axis = 1)
            Y = dataframe[dependent_feature]
            self.log_agent.log(log_file, "Dependent and Independent features separated successfully.")
            log_file.close()
            return X,Y
        except Exception as e:
            self.log_agent.log(log_file, "Error occurred during separating features, "+str(e))
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
            q1 = dataframe.quantile(0.25)
            q3 = dataframe.quantile(0.75)
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
            return dataframe

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
                return X_train_scaled
        except Exception as e:
            self.log_agent.log(log_file, "Error occurred while standardizing the data, "+str(e))
            log_file.close()

    def reshape_array(self, data, shape):
        try:
            log_file = open(self.preprocessing_log_file_path, 'a+')
            self.log_agent.log(log_file, "Initiating Reshaping process of Array")
            reshaped = np.reshape(data, newshape= shape)
            self.log_agent.log(log_file, "Reshaping of Array completed")
            log_file.close()
            return reshaped
        except Exception as e:
            self.log_agent.log(log_file, "Error occurred while reshaing array, "+str(e))
            log_file.close()

    def concatenate_array(self, array_list, axis = 1):
        try:
            log_file = open(self.preprocessing_log_file_path, 'a+')
            self.log_agent.log(log_file, "Concatenation of Array started")
            concatenated_array = np.concatenate(array_list, axis = axis)
            self.log_agent.log(log_file, "Concatenation of Array completed")
            log_file.close()
            return concatenated_array
        except Exception as e:
            self.log_agent.log(log_file, "Error occurred while Concatenating Arrays, "+str(e))
            log_file.close()

    def make_df(self, data, columns = None, axis = 1):
        try:
            log_file = open(self.preprocessing_log_file_path, 'a+')
            self.log_agent.log(log_file, "Dataframe Creation started")
            if columns != None:
                new_df = pd.DataFrame(data, columns = columns)
            else:
                new_df = pd.DataFrame(data)
            self.log_agent.log(log_file, "Dataframe creation completed")
            log_file.close()
            return new_df
        except Exception as e:
            self.log_agent.log(log_file, "Error occurred while creating Dataframe "+str(e))
            log_file.close()

    def df_train_test_split(self, x, y, split_ratio = 0.2):
        try:
            log_file = open(self.preprocessing_log_file_path, 'a+')
            self.log_agent.log(log_file, "Initiating train_test_split..")
            x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= split_ratio, random_state= 42)
            self.log_agent.log(log_file, "train_test_split completed")
            log_file.close()
            return x_train, x_test, y_train, y_test 
        except Exception as e:
            self.log_agent.log(log_file, "Error occurred while spliting df, "+str(e))
            log_file.close()
            
    def divideDfBasedOnCluster(self, df, cluster_list):
        try:
            log_file = open(self.preprocessing_log_file_path,'a+')
            df_dict = {}
            for cluster_num in cluster_list:
                cluster_name = 'df_cluster_'+str(cluster_num)
                df_dict[cluster_name] = df.loc[df['Cluster'] == cluster_num]
                df_dict[cluster_name] = self.removCols(df_dict[cluster_name], 'Cluster')
                self.log_agent.log(log_file, "Cluster : {} created for {} from df and stored in df_dict.".format(cluster_name, cluster_num))
            log_file.close()
            return df_dict
        except Exception as e:
            self.log_agent.log(log_file, "Exception occurred while dividing df based on cluster"+str(e))
            log_file.close()
            return None
        
    