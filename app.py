from crypt import methods
from flask import Flask, request, render_template, Response
import os
from flask_cors import CORS, cross_origin
from start_train_process import InitiateTrainProcess
from start_prediction_process import InitiatePredictonProcess


# creating Flask app
app = Flask(__name__)

train_db_path = 'Database/trainingDb.db'
train_table_name = 'GoodDataTable'

pred_db_bath = 'Database/predictionDb.db'
pred_table_name = 'GoodDataTable'

# For homepage
@app.route('/', methods = ['GET'])
@cross_origin()
def homepage():
    if request.method == 'GET':
        render_template('homepage.html')
    else:
        pass

# For Training
@app.route('/train', methods=['POST'])
@cross_origin()
def train():
    try:
        if request.json['folderPath'] is not None:
        
            path = request.json['folderPath']

            # calling the train methods
            train_obj = InitiateTrainProcess(path)

            train_obj.validate()
            train_obj.perform_data_transformation_for_db()
            train_obj.perform_Db_operations(db_path= train_db_path,tableName= train_table_name, training= True)
            train_obj.preprocess_and_initiate_training()
            return "TrainingCompleted"
        else: 
            raise Exception('Invalid File path')
    except Exception as e:
        print('Exception occurred, '+str(e))



# For Prediction
@app.route('/predict', methods = ['POST'])
@cross_origin()
def predict():
    try:
        if request.json['folderPath'] is not None:
        
            path = request.json['folderPath']

            # calling the train methods
            pred_obj = InitiatePredictonProcess(path)

            pred_obj.validate()
            pred_obj.perform_data_transformation_for_db()
            pred_obj.perform_Db_operations(db_path= pred_db_bath,tableName= pred_table_name, training= False)
            pred_obj.preprocess_and_initiate_prediction()
            return "PredictionCompleted"
        else: 
            raise Exception('Invalid File path')
    except Exception as e:
        print('Exception occurred, '+str(e))

# For Reset
# @app.route('/reset', methods = ['POST'])
# @cross_origin()
# def predict():
#     try:
#         if request.json['folderPath'] is not None:
        
#             reset = request.json['True']

#             # removing logs and processed files
#             logs = os.listdir('Logs')
#             for names in logs:
#                 if 
            
#         else: 
#             raise Exception('Invalid Voolean Value')
#     except Exception as e:
#         print('Exception occurred, '+str(e))


if __name__ == '__main__':
    app.run()