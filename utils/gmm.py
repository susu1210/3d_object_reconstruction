# from gaussian import Gaussian
from utils.gaussian import Gaussian
from sklearn.cluster import KMeans
import numpy as np
from past.builtins import xrange

class GMM:
    """
    Implementation of a Gaussian mixture model
    """
    
    def __init__(self, K):
        self.K = K
        self.gaussians = [Gaussian() for _ in xrange(self.K)]
        self.weights = np.array([1.0/K]*K)

    # X - Array of pixels, not necessarily an image
    def initialize_gmm(self, X):
        clusterer = KMeans(n_clusters=self.K, max_iter=10, random_state=None)
        clusters = clusterer.fit_predict(X)

        num_pixels = float(X.shape[0])


        for i, distribution in enumerate(self.gaussians):
            distribution.update_parameters(X[clusters==i])
            self.weights[i] = np.sum(clusters==i)/num_pixels
        return clusters

    def get_component(self, x):
        components = np.zeros((x.shape[0], len(self.gaussians)))

        for i,g in enumerate(self.gaussians):
            components[:, i] = self.weights[i]*g.compute_probability(x)

        return np.argmax(components, axis=1)

    def update_components(self, X, assignments):
        num_pixels = float(np.sum(assignments != -1))

   
        for i, distribution in enumerate(self.gaussians):
            if X[assignments==i].shape[0] != 0:
                distribution.update_parameters(X[assignments==i])
                self.weights[i] = (np.sum(assignments==i)/num_pixels)
            else:
                distribution.mean = [-1e9,-1e9,-1e9]
                self.weights[i] = 0

    def compute_probability(self, x):
        return np.dot(self.weights, [g.compute_probability(x) for g in self.gaussians])


