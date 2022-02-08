from sklearn.preprocessing import StandardScaler
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
import seaborn as sns

class PCA:
    def __init__(self,n_components=2):
        
        '''
        Default n_components=2
        '''
        
        self.n_components=n_components
        
    def fit(self,data,standardized_data=True):
       
        '''
        standardized_data=True tells that the data passed is already standardized.
        if standardized_data=False then fit function will standardize the data 
        '''
        
        if standardized_data==False:
            sc=StandardScaler()
            std_data=sc.fit_transform(data)
        else:
            std_data=data
            
        covariance_matrix = (np.matmul(std_data.T,std_data))/std_data.shape[0]
        self.covariance_matrix=covariance_matrix
        self.eig_values,self.eig_vectors=eig(self.covariance_matrix)
        
        sorted_eig_vectors_idx=np.argsort(self.eig_values)[::-1]
        self.n_eigen_vectors=self.eig_vectors[:,sorted_eig_vectors_idx[:self.n_components]]
    
    def eigen_vectors(self):
        '''
        Returns eigen vectors
        '''
        return self.n_eig_vectors
    
    def eigen_values(self):
        '''
        Returns eigen values
        '''
        return np.sort(self.eig_values)[::-1][:self.n_components]
    
    def explained_variance(self):
        '''
        Returns variance explained by each of the components.
        '''
        
        self.exp_var=list()
        for i in np.sort(self.eig_values)[::-1]:
            ev=i/np.sum(self.eig_values)
            self.exp_var.append(ev)
            
        return np.cumsum(self.exp_var)
    
    def plot_explained_variance(self):
        '''
        This function returns a graph of : number of components vs variance / information preserved
        '''
        self.exp_var=list()
        for i in np.sort(self.eig_values)[::-1]:
            ev=i/np.sum(self.eig_values)
            self.exp_var.append(ev)
            
        sns.set_style('whitegrid')
        plt.figure(figsize=(10, 6), dpi=70)
        plt.clf()
        plt.plot(np.cumsum(self.exp_var),linewidth=2)
        plt.xlabel('number of components')
        plt.ylabel('variance / information preserved')
        plt.show()
        
    def transform(self,data,standardized_data=True):
        if standardized_data==False:
            sc=StandardScaler()
            std_data=sc.fit_transform(data)
        else:
            std_data=data
            
        self.pca_data=np.matmul(std_data,self.n_eigen_vectors)
        return self.pca_data