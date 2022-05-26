from Logger.logger import Logger
from File_Schema_Validation.Training_File_Schema_Validate_utils.training_file_schema_validate_utils import Train_file_schema_validate 
from Data_Transforamation_For_DB.data_transformation_for_db import DataTransformation
from DbOperations.db_operations import DbOperations
from Model.Data_Loader.Training.training_data_loader import Training_data_loader
from Model.Data_Pre_Processing.preprocessing import DataPreProcessing   
from Model.Clustering.cluster import Cluster
from Model.Model_Selection.model_selector import ModelSelector

class InitiateTrainProcess:
	def __init__(self, train_batch_file_path):
		# Logger object
		self.log_agent = Logger()
		# Log file path
		self.initiate_training_log_file_path = 'Logs/Training_Logs/initiate_training_log_file.txt'
		
		# train_batch_file_path
		self.train_batch_file_path = train_batch_file_path

		# good data file path
		self.good_data_file_path = 'Validated_Training_Batch_Files/GoodFileSchemaDataFolder'

	def validate(self):
		"""
			:description: Initialize File Training Batch Files validation process
		"""
		try:
			log_file = open(self.initiate_training_log_file_path, 'a+')
			self.log_agent.log(log_file, "Starting Batch Files Validation")

			# Object of Train_file_schema_validate
			self.log_agent.log(log_file, "Creating object of class Train_file_schema_validate")
			train_file_schema_validate_obj = Train_file_schema_validate(self.train_batch_file_path)

			# Fetching meta properties of training batch files
			self.log_agent.log(log_file, "Fetching meta properties of training batch files")
			accepted_pattern, accepted_len_date, accepted_len_time, accepted_no_cols = train_file_schema_validate_obj.getValuesFromSchema()
			
			# Fetching regex for validating file name
			self.log_agent.log(log_file, "Fetching regex pattern for file name")
			re_pattern = train_file_schema_validate_obj.regex_pattern()

			# validating file names
			self.log_agent.log(log_file, "Validating file names of training batch files")
			train_file_schema_validate_obj.validate_file_name(re_pattern, accepted_len_date, accepted_len_time)

			# validate no of cols in each file
			self.log_agent.log(log_file, "Validating no of cols of training batch files")
			train_file_schema_validate_obj.validate_no_cols(accepted_no_cols)

			# validate mssing values in each file
			self.log_agent.log(log_file, "Validating missing values of training batch files")
			train_file_schema_validate_obj.validate_missing_values()

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
			log_file =open(self.initiate_training_log_file_path, 'a+')
			self.log_agent.log(log_file, "Starting data transformation process for GoodDataFiles.")
			# creating obj of DataTransformation
			self.log_agent.log(log_file, "Creating object of class DataTransformation")
			transformation_obj = DataTransformation(training = True)

			# performing data transformation
			# self.log_agent.log(log_file, "Calling transform_data_for_db() of DataTransformation class.")
			transformation_obj.transform_data_for_db(self.good_data_file_path)
			self.log_agent.log(log_file, "Data transformation completed.")
			log_file.close()
		except Exception as e:
			self.log_agent.log(log_file, "Exception occurred in perform_data_transformation_for_db() method of DataTransformation class, "+str(e))
			log_file.close()
			
	def perform_Db_operations(self, db_path, tableName, training = True):
		"""
		:description: Stores validated data into Training.db and  creates a acsv file.
		:param db_path: Database path
		:param tableName: Table Name
		:param training: Boolean if True, object will be used for training purpose.
		"""
		try:
			log_file =open(self.initiate_training_log_file_path, 'a+')
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

	def preprocess_and_initiate_training(self):
		try: 
			log_file = open(self.initiate_training_log_file_path, 'a+')
			self.log_agent.log(log_file, "Starting Preprocessing and training of data.")

			# creating object of Training_data_loader
			self.log_agent.log(log_file, "Creating object of Training_data_loader")
			train_data_loader_obj = Training_data_loader()

			# Fetching Data Frame
			self.log_agent.log(log_file, "Fetching DataFrame")
			df = train_data_loader_obj.getTrainingDataFrame()

			# creating object of DataPreprocessing class.
			self.log_agent.log(log_file, "Creating object of DataPreprocessing class.")
			preprocessor = DataPreProcessing()

			# Checking for Null Values in Df
			self.log_agent.log(log_file, "Checking for Null Values in Df")
			preprocessor.containsNull(df)

			# Fetching column names
			self.log_agent.log(log_file, "Fetchig column names")
			all_cols = list(preprocessor.getDfColNames(df))
			
			# Converting to Log Normal Form
			self.log_agent.log(log_file, "Converting to Log Normal Form")
			cols = list(preprocessor.getDfColNames(preprocessor.removCols(df, ['Concrete_compressive _strength'])))
			df = preprocessor.convertToLogNormalForm(df, cols)
			# df = preprocessor.convertToLogNormalForm(df, ['Cement _component_1', 'Blast Furnace Slag _component_2',
			# 'Fly Ash _component_3', 'Water_component_4',
			# 'Superplasticizer_component_5', 'Coarse Aggregate_component_6',
			# 'Fine Aggregate_component_7', 'Age_day'])

			# handling Outliers
			self.log_agent.log(log_file, "Staring handling Outliers")
			df = preprocessor.handleOutliers(df)
			
			# Separating Depenedent and Indepenedent Features
			self.log_agent.log(log_file, "Separating Depenedent and Indepenedent Features")
			X, Y = preprocessor.separateDependentIndependentFeatures(df, ['Concrete_compressive _strength'])

			# Scailing data frame
			self.log_agent.log(log_file, "Scailing data frame")
			X_scaled = preprocessor.standardScaler(X)
			# X_scaled

			# Clustering
			self.log_agent.log(log_file, "Creating object of Cluster Class")
			cluster_obj = Cluster()

			self.log_agent.log(log_file, "Staring Clustering of data")
			clusters = cluster_obj.kmeans(X_scaled)
			
			self.log_agent.log(log_file, "Reshaping Clusters array to merge later with scaled data")
			reshaped_cluster_y = preprocessor.reshape_array(data = clusters, shape = (len(clusters), 1))
			# reshaped_cluster_y
			         
			# column_names = preprocessor.getColsNames('File_Schema_Validation/files_schema/schema_training.json')
			
			# column_names.append('Cluster')
			all_cols.append('Cluster')
			self.log_agent.log(log_file, "Concatenating all arrays : X_scaled, Y and  cluster array")
			new_array = preprocessor.concatenate_array([X_scaled, Y, reshaped_cluster_y], axis = 1)

			self.log_agent.log(log_file, "Creating DataFrame of above created array")
			df = preprocessor.make_df(new_array,columns= all_cols)
			
			# Creating df based upon clusters and training for each clsuter

			self.log_agent.log(log_file, "Obtaining unique Values in cluster")
			cluster_values = tuple(df['Cluster'].unique())

			self.log_agent.log(log_file, "Dividing df based upon cluster")
			dic = preprocessor.divideDfBasedOnCluster(df, cluster_values)

			self.log_agent.log(log_file, "Creating object fo ModelSelector class")
			model_selector_obj = ModelSelector()

			self.log_agent.log(log_file, "Starting Training DataFrame")			
			for key in dic.keys():
					key_name = str(key)
					temp_x, temp_y = preprocessor.separateDependentIndependentFeatures(dic.get(key_name), ['Concrete_compressive _strength'])
					model_selector_obj.selectModel(temp_x, temp_y, key_name)

			self.log_agent.log(log_file, "Training Completed.")
			log_file.close()
		except Exception as e:
			self.log_agent.log(log_file,'Exception Occured in preprocess_and_initiate_training() method in InitiateProcess.')
			log_file.close()
