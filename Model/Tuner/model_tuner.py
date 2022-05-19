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
            self.log_agent.log("Initiating find_best_param process ..")
            if model_type == "RandomForestReg":
                params = {}
                estimator = RandomForestRegressor()
                best_param = self.gridSearch(estimator, x_train, y_train, params)[0]
                self.log_agent.log(log_file, "find_best_param successfully completed")
                log_file.close()
                return best_param
            elif model_type == "ElasticNetReg":
                params = {}
                estimator = ElasticNet()
                best_param = self.gridSearch(estimator, x_train, y_train, params)[0]
                self.log_agent.log(log_file, "find_best_param successfully completed")
                log_file.close()
                return best_param
        except Exception as e:
            self.log_agent.log(log_file,"Error occured while finding best parameter for {} ".format(model_type)+str(e))
            log_file.close()

    def gridSearch(self, estimator, x_train, y_train, params):
        try:
            log_file = open(self.model_tuner_log_file_path, 'a+')
            self.log_agent.log(log_file, "Starting GridSearch")
            gs_model = GridSearchCV(estimator=estimator, params=params, n_jobs= 3)
            gs_model.fit(x_train, y_train)
            best_param = gs_model.best_params_
            best_score = gs_model.best_score_
            self.log_agent.log(log_file, "Best Params : {} /nBest Score : {} ".format(best_param, best_score))
            log_file.close()
            return (best_param, best_score)
        except Exception as e:
            self.log_agent.log(log_file, "Error occured while performing GridSearch, "+str(e))
            log_file.close()
            