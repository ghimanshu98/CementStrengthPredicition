from Logger.logger import Logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from Model.Saved_models.model_saver import SaveModel
from Model.Tuner.model_tuner import ModelTuner

class ModelSelector:
    def __init__(self):
        # logger object
        self.log_agent = Logger()

        # logger path
        self.model_selector_log_path = "Logs/Model_Logs/model_selector_log_file.txt"

        # save model object
        self.model_saver = SaveModel()

    def selectModel(self, x_train, y_train, cluster_name, x_test = None, y_test = None):
        try:
            log_file = open(self.model_selector_log_path, 'a+')

            self.log_agent.log(log_file,"Initiating model_selector")

            self.log_agent.log(log_file, "Comparing models for cluster, {} ".format(cluster_name))
            # comparing model to get score
            comp_result = self.compareModel(x_train, y_train)

            # model tuner object
            tuner = ModelTuner()

            # tuning model
            if comp_result[0] == "RandomForestReg":
                self.log_agent.log(log_file, "RandomForestReg performed better for cluster - {}, therefore choosing and tuning it".format(cluster_name))
                params = tuner.find_best_param(x_train, y_train, comp_result[0])
                self.randomForestReg(x_train, y_train, model_name=cluster_name, save_model= True, params= params)
                self.log_agent.log(log_file, "Model Selection Process competed.")
                log_file.close()
            elif comp_result[0] == "ElasticNetReg":
                self.log_agent.log(log_file, "ElasticNetReg performed better for cluster - {}, therefore choosing and tuning it".format(cluster_name))
                params = tuner.find_best_param(x_train, y_train, comp_result[0])
                self.elasticNetReg(x_train, y_train, model_name=cluster_name, save_model=True, params= params)
                self.log_agent.log(log_file, "Model Selection Process competed.")
                log_file.close()
        except Exception as e:
            self.log_agent.log(log_file,"Error occurred while selecting model for clsuter {}, ".format(cluster_name)+str(e))
            log_file.close()


    def compareModel(self, x_train, y_train):
        try:
            log_file = open(self.model_selector_log_path, 'a+')
            self.log_agent.log(log_file,"Copmparing Model process started")
            forest_score = self.randomForestReg(x_train, y_train, save_model= False)[2]
            elastic_score = self.elasticNetReg(x_train, y_train, save_model =False)[2]
            if forest_score > elastic_score:
                self.log_agent.log(log_file, "On comparision RandomForestReg performed better with score {} while ElasticNetReg got a score of {} ".format(forest_score, elastic_score))
                log_file.close()
                return ("RandomForestReg", forest_score)
            else:
                self.log_agent.log(log_file, "On comparision ElasticNetReg performed better with score {} while RandomForestReg got a score of {} ".format(forest_score, elastic_score))
                log_file.close()
                return ("ElasticNetReg", elastic_score)
        except Exception as e:
            self.log_agent.log(log_file, "Exception occurred while comparing models, "+str(e))
            log_file.close()
        

    def randomForestReg(self, x_train, y_train, model_name = None, save_model = False, params = None):
        try:
            log_file = open(self.model_selector_log_path, 'a+')
            self.log_agent.log(log_file,"Starting RandomForestRegressor model training..")

            if params == None:
                forest_reg = RandomForestRegressor(n_jobs=3, random_state= 42)
            else:
                pass  # to be completed

            forest_reg.fit(x_train, y_train)
            forest_score = forest_reg.score(x_train, y_train)
            self.log_agent.log(log_file,"RandomForestRegresor model training completed")

            if save_model:
                model_name = "RandomForestReg_"+model_name
                self.model_saver.save_model(forest_reg, model_name)
                self.log_agent.log(log_file,"RandomForestRegresor model : {}  Saved successully".format(model_name))
            log_file.close()

            return ("RandomForestReg", forest_reg, forest_score)
        except Exception as e:
            self.log_agent.log(log_file, "Exception occurred while training RandomForestRegressorModel, "+str(e))
            log_file.close()

    def elasticNetReg(self, x_train, y_train, model_name = None, save_model = False, params = None):
        try:
            log_file = open(self.model_selector_log_path, 'a+')
            self.log_agent.log(log_file,"Starting ElasticNetRegresor model training..")

            if params == None:
                net_reg = ElasticNet(random_state= 42)
            else:
                pass # to be completed

            net_reg.fit(x_train, y_train)
            elastic_score = net_reg.score(x_train, y_train)
            self.log_agent.log(log_file,"ElasticNetRegresor model training completed")

            if save_model:
                model_name = "ElasticNetReg_"+model_name
                self.model_saver.save_model(net_reg, model_name)
                self.log_agent.log(log_file,"ElasticNetRegressor : {}  saved successully".format(model_name))
            log_file.close()

            return ("ElasticNetReg", net_reg, elastic_score)
        except Exception as e:
            self.log_agent.log(log_file, "Exception occurred while training ElasticNetRegresor models, "+str(e))
            log_file.close()


