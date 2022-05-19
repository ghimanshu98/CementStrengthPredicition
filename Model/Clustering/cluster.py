from Logger.logger import Logger
from sklearn.cluster import KMeans
from kneed import KneeLocator
import matplotlib.pyplot as plt
from Model.Saved_models.model_saver import SaveModel

class Cluster:
    def __init__(self):
        # logger object
        self.log_agent = Logger()
        # path for log file
        self.cluster_log_file_path = "Logs/Model_Logs/Cluster_log_file.txt"
    
    def elbow_method(self, x, cluster_lower_range, cluster_upper_range):
        try:
            log_file = open(self.cluster_log_file_path, 'a+')
            self.log_agent.log(log_file, "Initiating elbow method")
            wcss = []
            for i in range(cluster_lower_range,cluster_upper_range):
                clusterer = KMeans(n_clusters= i, n_jobs= 3, random_state= 42)
                clusterer.fit(x)
                wcss.append(clusterer.inertia_)
            
            # plotting elbow plot
            plt.plot(range(cluster_lower_range, cluster_upper_range), wcss)
            plt.title("Elbow Method")
            plt.xlabel("Nuber of Clusters")
            plt.ylabel("WCSS")
            plt.savefig("Model/Clustering/Elbow_Method.png")

            self.log_agent.log(log_file, "Elbow method finished, Elbow Plot stored at Model/Clustering/Elbow_Method.png")
            log_file.close()
            return wcss
        except Exception as e:
            self.log_agent.log(log_file, "Exception occurred while plotting elbow method, "+str(e))
            log_file.close()
            

    def knee_finder(self,x, cluster_lower_range = 1, cluster_upper_range = 30):
        try:
            log_file = open(self.cluster_log_file_path, 'a+')
            self.log_agent.log(log_file, "Initiating Knee Method")
            wcss = self.elbow_method(x,cluster_lower_range, cluster_upper_range)
            kn = KneeLocator(range(cluster_lower_range, cluster_upper_range), wcss, curve="convex", direction="decreasing" )
            self.log_agent.log(log_file, "Optimum number of clusters for data X is {}".format(kn.knee))
            log_file.close()
            return kn.knee
        except Exception as e:
            self.log_agent.log(log_file, "Error occurred while finding knee for data, "+str(e))
            log_file.close()
             

    def kmeans(self, x):
        try:
            log_file = open(self.cluster_log_file_path, 'a+')
            self.log_agent.log(log_file, "Initiaing KMeans..")
            # Finding optimum number of cluster
            n_clusters = self.knee_finder(x)
            clusterer = KMeans(n_clusters = n_clusters, random_state=42, n_jobs = 3)
            cluster_prediction_y = clusterer.fit_predict(x)

            # saving model
            model_saver = SaveModel()
            model_saver.save_model(clusterer, "Kmeans_cluster")

            self.log_agent.log(log_file, "Kmans completed, Model Saved.")
            log_file.close()
            return cluster_prediction_y
        except Exception as e:
            self.log_agent.log(log_file, "Error occurred while performing Kmeans CLustering, "+str(e))
            log_file.close()


    
