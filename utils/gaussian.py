import numpy as np

class Gaussian:
 
    def __init__(self, mean=np.zeros((3,1)), sigma=np.eye(3)):
        
        # mean: mean for the gaussian
        self.mean = np.array(mean)
        # sigma: Covariance matrix 
        self.sigma = np.array(sigma) + np.eye(3)*1e-8
        
        self.sigma_det = np.linalg.det(sigma)
        self.sigma_inv = np.linalg.inv(sigma)

        self.k = 3
        self.TWO_PI_3 = (2*np.pi)**self.k
        self.term1 = 1/np.sqrt(self.TWO_PI_3 * self.sigma_det)

    def compute_probability(self, x):
        x = np.array(x)
        mean = np.tile(self.mean, (x.shape[0], 1))
        middle_matrix = np.dot((x - mean), self.sigma_inv)
            
        return self.term1 * np.exp(-0.5 * np.sum(np.multiply((x-mean), middle_matrix),axis=1))

    def update_parameters(self, data):
        self.mean = np.mean(data, axis=0)
        self.sigma = np.cov(data, rowvar=0) + np.eye(self.k)*1e-8

        self.sigma_det = np.linalg.det(self.sigma)
        self.sigma_inv = np.linalg.inv(self.sigma)
        self.term1 = 1/np.sqrt(self.TWO_PI_3 * self.sigma_det)
