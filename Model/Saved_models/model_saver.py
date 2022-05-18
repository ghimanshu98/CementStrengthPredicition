import pickle
from Logger.logger import Logger

class SaveModel:
    def __init__(self):
        # creating logger object
        self.log_agent = Logger()
        # log path file
        self.model_save_log_path = "Logs/Model_Logs/model_save_log_file.txt"

        # path for saving models to.
        self.save_model_path = "Model/Saved_models"

    def save_model(self, model_obj, model_name):
        try:
            log_file = open(self.model_save_log_path, 'a+')
            self.log_agent.log(log_file, "Initiating process to save model : {} to path {}".format(model_name, self.save_model_path))

            model_name = model_name+".sav"
            pickle.dump(model_obj, open(self.model_save_model+"/"+model_name, 'wb'))
            
            self.log_agent.log(log_file, "Model : {} saved to path {} successfully".format(model_name, self.save_model_path))
            log_file.close()
        except Exception as e:
            self.log_agent.log(log_file, "Exception occurred during model saving. "+str(e))
            log_file.close()
            
    def load_model(self, model_path):
        try:
            log_file = open(self.model_save_log_path, 'a+')
            self.log_agent.log(log_file, "Loading saved model {}".format(model_path))
            loaded_model = pickle.load(open(model_path, 'rb'))
            if loaded_model != None:
                self.log_agent.log(log_file, "Model {} loaded successfully.".format(loaded_model))
                log_file.close()
                return loaded_model
        except Exception as e:
            self.log_agent.log(log_file, "Problem occurred during model loading. "+str(e))
            log_file.close()
