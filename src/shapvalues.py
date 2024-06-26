import numpy as np
from itertools import product
from math import factorial

class ShapleyEstimator:
    def __init__(self, model, X, y):
        """
        Initializes the ShapleyEstimator with a trained model and dataset.
        
        Parameters:
        - model: Trained machine learning model.
        - X: numpy.ndarray containing the feature data.
        - y: Array containing the target values.
        """
        self.model = model
        self.p = X.shape[1]
        self.X = X
        self.y = y
        self.mean_values = np.nanmean(X, axis=0)
        self.num_class = len(np.unique(y))
        

    def predict(self, X):
        """
        Predict class probabilities for the provided data using the trained model
        """
        return self.model.predict_proba(X)
    
    def create_mask(self, feature_idx):
        """
        Create masks for feature combinations with and without the specified feature.

        Parameters:
        - feature_idx : int
            Index of the feature to include or exclude in the masks.

        Returns:
        tuple
            Tuple containing two masks:
            - mask_with: Mask with the specified feature included.
            - mask_without: Mask with the specified feature excluded.
        """
        mask = np.array(list(product(range(2), repeat=self.p)))
        mask_with = mask[np.where(mask[:, feature_idx] == 1)]
        mask_without = mask_with.copy()
        mask_without[:,feature_idx] = 0
        return (mask_with, mask_without)
    
    def create_z(self, mask, instance):
        """
        Create a modified instance based on the provided mask and instance.
        """
        S_with_zero = mask*instance
        S = (S_with_zero == 0)*self.mean_values + S_with_zero

        return S
    
    def marginal_contribution(self, feature_idx, instance):
        """        
        Compute the marginal contribution of a feature to the prediction for a given instance.

        Parameters:
        - feature_idx : int
            Index of the feature for which marginal contribution is computed.
        - instance : numpy.ndarray
            Input instance of shape (n_features,) or (1, n_features).

        Returns:
        - float: Marginal contribution of the feature to the prediction.
        """
        
        mask_with, mask_without = self.create_mask(feature_idx)
        z_with = self.create_z(mask_with, instance)
        z_without = self.create_z(mask_without, instance)
        f_with = self.predict(z_with)
        f_without = self.predict(z_without)
        return (f_with - f_without)
    
    def calculate_weight(self, mask_row):
        """
        Calculate the weight associated with a specific subset of features represented by 'mask_row'
        """
        
        z_magnitude = np.sum(mask_row) # the number of features != 0
        weight = factorial(z_magnitude) * factorial(self.p - z_magnitude - 1) / factorial(self.p)
        return weight
    
    def weight_array(self, mask):
        """
        Compute weights for each row in the mask matrix.

        Parameters:
        - mask : numpy.ndarray
            Matrix where each row is a binary mask representing a subset of features.

        Returns:
        - numpy.ndarray - Array of weights corresponding to each row in 'mask'.
        """
        weights  = np.apply_along_axis(self.calculate_weight, 1, mask)
        return weights

    
    def calculate_shapley_values(self, feature_idx, instance):
        """
        Calculate the Shapley values for a single instance.

        Parameters:
        - feature_idx : (int) Index of the feature for which Shapley value is computed.
        - instance : (numpy.ndarray) Input instance for which Shapley value is computed.
        
        Returns:
        - Array of Shapley values for a feature.
        """
        mask_without = self.create_mask(feature_idx)[1]
        weights = self.weight_array(mask_without)
        mc = self.marginal_contribution(feature_idx, instance)
        phi = np.sum(np.multiply(weights.reshape(-1,1), mc),axis=0)

        return phi
    
    def shapley_values_arrays(self, instances):
        """
        Compute Shapley values for multiple instances across multiple features.

        Parameters:
        - instances : (numpy.ndarray) Input instances of shape (num_indices, num_features).

        Returns:
        - numpy.ndarray - Array of Shapley values with shape (num_indices, num_features).
        """
        num_indices, num_features = instances.shape[0], instances.shape[1]
        shapley_values = np.zeros((num_indices, num_features))

        for i in range(num_features):
            phis = np.zeros(num_indices)
            for j, instance in enumerate(instances):
                phi = self.calculate_shapley_values(i,instance)
                phis[j].append(phi)
            shapley_values[:,i] = phis
        return shapley_values


