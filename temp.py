from File_Schema_Validation.Training_File_Schema_Validate_utils.training_file_schema_validate_utils import Train_file_schema_validate

from File_Schema_Validation.Prediction_File_Schema_Validation_utils.prediction_file_schema_validate_utils import Predict_file_schema_validate

temp = Train_file_schema_validate(train_batch_file_path= 'Training_Batch_Files')

reg = temp.regex_pattern()

temp.validate_file_name(reg, 8,6)
temp.validate_no_cols(9)
temp.validate_missing_values()


temp2 = Predict_file_schema_validate(predict_batch_file_path= "Prediction_Batch_files")

temp2.validate_file_name(reg, 8,6)
temp2.validate_no_cols(8)
temp2.validate_missing_values()

# Literal['[\'cement_strength\']+[\'\\_\'\']+[\\d_]+[\\d]+\\.csv']
# Literal['[\'cement_strength\']+[\'\\_\'\']+[\\d_]+[\\d]+\\.csv']