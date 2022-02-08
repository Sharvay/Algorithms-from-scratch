import numpy as np
from scipy import spatial
from scipy import stats
from collections import Counter


class KNearestNeighborsClf:
    
    def __init__(self,n_neighbors=3,distance_metric='eucledian'):
        """
        Initialize your number of nearest_neighbors and 
        Initialize your distance_metric ie. "eucledian", "manhattan", "cosine_similarity"

        Default n_neighbors=3 
        Default distance_metric='eucledian'
        """

        self.n_neighbors=n_neighbors
        self.distance_metric=distance_metric


    def fit(self,x_train,y_train):
        """
        Takes training data ie.x_train and y_train
        KNearestNeighborsClf.fit() method
        """

        self.x_train=np.asarray(x_train)
        self.y_train=np.asarray(y_train)
        self.len_x_train=self.x_train.shape
        self.len_y_train=self.y_train.shape
        print("KNearestNeighborsClf(n_neighbors={0},distance_metric={1})".format(self.n_neighbors,self.distance_metric))
        

    def predict(self,x_test):
        """
        Takes Test data for predictions considering training data to find the nearest neighbors.
        Returns numpy array of predictions.
        """

        x_test=np.asarray(x_test)
        predictions_list=list()

        # For each point in test data
        for test_pt in x_test:
            classes=[]
            distance_dict=list()
           
            # For each point in training data
            for x_train_pt,y_train_pt in zip(self.x_train,self.y_train):
                
                # calculate distance between 
                dist = self.metric(test_pt,x_train_pt)
                distance_dict.append([dist,y_train_pt])

            # getting k_nearest_neighbors by sorting according to distance
            classes = [x[1] for x in sorted(distance_dict, key = lambda d: d[0])]
            classes=classes[:self.n_neighbors]

            # getting majority class
            predictn = stats.mode(classes)[0][0]
            predictions_list.append(predictn)
        
        return np.array(predictions_list)


    def predict_proba(self,x_test):
        """
        Takes Test data for predictions considering training data to find the nearest neighbors.
        Returns numpy array of predictions.
        """

        x_test=np.asarray(x_test)
        pred_proba=list()

        # For each point in test data
        for test_pt in x_test:
            classes=[]
            distance_dict=list()
           
            # For each point in training data
            for x_train_pt,y_train_pt in zip(self.x_train,self.y_train):
                
                # calculate distance between 
                dist = self.metric(test_pt,x_train_pt)
                distance_dict.append([dist,y_train_pt])
            
            # getting k_nearest_neighbors by sorting according to distance
            classes = [x[1] for x in sorted(distance_dict, key = lambda d: d[0])]
            classes=classes[:self.n_neighbors]

            #print(classes)
            # probabilities for each class
            proba=[Counter(classes)[i]/len(classes) for i in np.unique(self.y_train)]
            pred_proba.append(proba)

        return pred_proba
        


    def metric(self,x1,x2):
        """
        Calculates specified distance 
        """
    
        if self.distance_metric=="eucledian":
            eucl_dist = np.linalg.norm(x1-x2)
            return eucl_dist
        
        elif self.distance_metric=="manhattan":
            manh_dist = np.linalg.norm(x1-x2,ord=1)
            return manh_dist

        elif self.distance_metric=="cosine_similarity":

            """
            cosine similarity from scratch:
            numerator = x.dot(y)
            denominator = (linalg.norm(x,ord=2)*linalg.norm(y,ord=2))
            cos_sim = numerator/denominator   
            cos_dist = 1-cos_sim     
            """
            
            cos_dist = spatial.distance.cosine(x1,x2)
            return cos_dist
 
    