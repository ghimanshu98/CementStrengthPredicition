from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from Logger.logger import Logger
from sklearn.model_selection import GridSearchCV

class ModelTuner:
    def __init__(self):
        # log_object
        self.log_agent = Logger()
        # log path
        self.model_tuner_log_file_path = "Logs/Model_Logs/model_tuner_log_file.txt"

    def find_best_param(self, x_train, y_train, model_type):
        try:
            log_file = open(self.model_tuner_log_file_path, 'a+')
            self.log_agent.log(log_file,"Initiating find_best_param process ..")
            if model_type == "RandomForestReg":
                params = {
                    "n_estimators" : [3,5,25,50,75,100],
                    "max_depth" : [2,5,10, None], 
                    "min_samples_split" : [2,5,10,20]
                }
                estimator = RandomForestRegressor(random_state= 42)
                best_param = self.gridSearch(estimator, x_train, y_train, params)[0]
                if best_param != None:
                    self.log_agent.log(log_file, "find_best_param successfully completed")
                    log_file.close()
                    return best_param
                else:
                    raise Exception("self.gridSearch() returned None")
            elif model_type == "ElasticNetReg":
                params = {
                    "alpha" : [0.2, 0.5 , 1 , 1.5],
                    "l1_ratio" : [0.2,0.4,0.5,0.6,0.8]
                }
                estimator = ElasticNet(random_state= 42)
                best_param = self.gridSearch(estimator, x_train, y_train, params)[0]
                if best_param !=None:
                    self.log_agent.log(log_file, "find_best_param successfully completed")
                    log_file.close()
                    return best_param
                else:
                    raise Exception("self.gridSearch() returned None")
        except Exception as e:
            self.log_agent.log(log_file,"Error occured while finding best parameter for {} ".format(model_type)+str(e))
            log_file.close()
            return None

    def gridSearch(self, estimator, x_train, y_train, params):
        try:
            log_file = open(self.model_tuner_log_file_path, 'a+')
            self.log_agent.log(log_file, "Starting GridSearch")
            gs_model = GridSearchCV(estimator=estimator, param_grid=params, n_jobs= 3)
            gs_model.fit(x_train, y_train)
            best_param = gs_model.best_params_
            best_score = gs_model.best_score_
            gs_model.best_estimator_
            self.log_agent.log(log_file, "Best Params : {} /nBest Score : {} ".format(best_param, best_score))
            log_file.close()
            return (best_param, best_score)
        except Exception as e:
            self.log_agent.log(log_file, "Error occured while performing GridSearch, "+str(e))
            log_file.close()
            return None
            