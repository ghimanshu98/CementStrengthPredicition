from Model.Data_Loader.Training.training_data_loader import Training_data_loader
train_data_loader_obj = Training_data_loader()
df = train_data_loader_obj.getTrainingDataFrame()
# df.head(5)
from Model.Data_Pre_Processing.preprocessing import DataPreProcessing
preprocessor = DataPreProcessing()
# check null

preprocessor.containsNull(df)
preprocessor.getDfColNames(df)
df = preprocessor.convertToLogNormalForm(df, ['Cement _component_1', 'Blast Furnace Slag _component_2',
       'Fly Ash _component_3', 'Water_component_4',
       'Superplasticizer_component_5', 'Coarse Aggregate_component_6',
       'Fine Aggregate_component_7', 'Age_day'])

# df.head(5)
df = preprocessor.handleOutliers(df)

# df.head(5)
X, Y = preprocessor.separateDependentIndependentFeatures(df, ['Concrete_compressive _strength'])
# Y
X_scaled = preprocessor.standardScaler(X)
# X_scaled
# len(X_scaled)



# Clustering

from Model.Clustering.cluster import Cluster

cluster_obj = Cluster()
pred = cluster_obj.kmeans(X_scaled)
# pred
reshaped_cluster_y = preprocessor.reshape_array(data = pred, shape = (len(pred), 1))
# reshaped_cluster_y
column_names = preprocessor.getColsNames('File_Schema_Validation/files_schema/schema_training.json')
column_names.append('Cluster')
new_array = preprocessor.concatenate_array([X_scaled, Y, reshaped_cluster_y], axis = 1)
df = preprocessor.make_df(new_array,columns= column_names)
# df



X, Y = preprocessor.separateDependentIndependentFeatures(df, "Concrete_compressive _strength")

# Cluster division

di = tuple(df['Cluster'].unique())
dic = preprocessor.divideDfBasedOnCluster(df, di)

# # Model Selection

# from Model.Model_Selection.model_selector import ModelSelector

# model_selector_obj = ModelSelector()

# result = model_selector_obj.selectModel(X.drop(['Cluster'], axis = 1), Y, "whole")