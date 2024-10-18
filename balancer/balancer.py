import numpy as np
import pandas as pd
import xgboost as xgb
from balancer.global_vars import METHODS

from scipy.stats import variation
from sklearn.decomposition import PCA
import pycatch22

def calc_dataset_size(data):
    """Calculate the number of time series in the dataset."""
    return data.shape[0]

def calc_length(data):
    """Calculate the length of each time series."""
    return data.shape[1] - 1  # Exclude the label column

def calc_average_patterns_per_class(data):
    """Calculate the average count of data occurrences in each class."""
    labels = data[:, 0]  # First column is the label
    unique, counts = np.unique(labels, return_counts=True)
    return np.mean(counts)

def calc_variance(data):
    """Calculate the average variance of each component of the time series."""
    time_series_data = data[:, 1:]  # Exclude the label column
    return np.mean(np.var(time_series_data, axis=1))

def calc_intra_class_variance(data):
    """Calculate the average variance per class (element-wise)."""
    labels = data[:, 0]  # First column is the label
    time_series_data = data[:, 1:]
    unique_labels = np.unique(labels)
    intra_class_variances = []
    for label in unique_labels:
        class_data = time_series_data[labels == label]
        intra_class_variances.append(np.mean(np.var(class_data, axis=1)))
    return np.mean(intra_class_variances)

def calc_imbalance_degree(data):
    """Calculate the class imbalance degree (ID)."""
    labels = data[:, 0]  # First column is the label
    unique, counts = np.unique(labels, return_counts=True)
    return variation(counts)  # Coefficient of variation of class counts

def calc_bhattacharyya_coefficient(data):
    """Calculate the Gaussian Bhattacharyya Coefficient (GBC)."""
    labels = data[:, 0]
    time_series_data = data[:, 1:]
    unique_labels = np.unique(labels)
    pca = PCA(n_components=2)

    # Get data for each class
    class_data = [time_series_data[labels == label] for label in unique_labels]

    if len(class_data) < 2:
        return 0  # If only one class exists, there's no overlap

    # Calculate the mean and covariance for each class after PCA
    transformed_data = [pca.fit_transform(cd) for cd in class_data]
    means = [np.mean(td, axis=0) for td in transformed_data]
    covariances = [np.cov(td.T) for td in transformed_data]

    # Compute the Bhattacharyya distance between each pair of classes
    gbc_sum = 0
    count = 0
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            mean_diff = means[i] - means[j]
            cov_avg = (covariances[i] + covariances[j]) / 2
            try:
                inv_cov_avg = np.linalg.inv(cov_avg)
                distance = (1/8) * mean_diff.T @ inv_cov_avg @ mean_diff + (1/2) * np.log(np.linalg.det(cov_avg) / np.sqrt(np.linalg.det(covariances[i]) * np.linalg.det(covariances[j])))
                gbc_sum += np.exp(-distance)  # Convert distance to a similarity measure
                count += 1
            except np.linalg.LinAlgError:
                continue

    return gbc_sum / count if count > 0 else 0

def calc_catch22_features(data):
    """Calculate the mean of each catch22 feature across the dataset."""
    time_series_data = data[:, 1:]
    sum_features = np.zeros(22)
    for ts in time_series_data:
        sum_features += np.array(pycatch22.catch22_all(ts)["values"])
    return sum_features / time_series_data.shape[0]

import xgboost as xgb
from balancer.global_vars import METHODS

class BALANCER:
    def __init__(self, param_file='models_param.json', train=False, classifiers=None):
        """
        Initialize the BALANCER class.

        Args:
            param_file (str): Path to the JSON file containing model parameters.
            train (bool): Whether to train new models or load existing ones.
            classifiers (list): List of classifiers to use (e.g., ['MLP', 'RF', 'KERNEL']).
        """
        self.methods = METHODS
        self.all_classifiers = ['DTW-KNN', 'KERNEL', 'NN', 'RF', 'SHAPELET', 'TS-RF']
        self.classifiers = classifiers if classifiers is not None else ['DTW-KNN', 'RF', 'KERNEL', 'NN', 'SHAPELET', 'TS-RF']
        if 'MLP' in self.classifiers:
            self.classifiers.remove('MLP')
            self.classifiers.append('NN')

        # Validate classifiers
        for clf in self.classifiers:
            if clf not in self.all_classifiers:
                raise ValueError(f"Invalid classifier '{clf}'. Valid classifiers are: {self.all_classifiers}")

        self.models = {method: xgb.XGBRegressor(enable_categorical=True) for method in self.methods}
        self.param_file = param_file

        if train:
            self._set_model_params()
        else:
            self._load_models()

    def _load_models(self):
        for method in self.methods:
            self.models[method].load_model(f'models/model_{method.lower()}.json')

    def _set_model_params(self):
        import json
        with open(self.param_file, 'r') as file:
            self.meta_model_params = json.load(file)
        
        for method, params in self.meta_model_params.items():
            self.models[method].set_params(**params)

    def _prepare_data(self, X_in):
        """Prepare data for feature extraction and model input."""
        if isinstance(X_in, np.ndarray):
            return X_in
        elif isinstance(X_in, pd.DataFrame):
            return X_in.to_numpy()
        else:
            raise TypeError("Input must be a NumPy array or a pandas DataFrame.")

    def _extract_features(self, X_in):
        """Extract the required features from the dataset."""
        features = {
            'DS': calc_dataset_size(X_in),
            'L': calc_length(X_in),
            'APC': calc_average_patterns_per_class(X_in),
            'DV': calc_variance(X_in),
            'IV': calc_intra_class_variance(X_in),
            'ID': calc_imbalance_degree(X_in),
            'GBC': calc_bhattacharyya_coefficient(X_in)
        }
        order = ['L', 'DS', 'APC', 'DV', 'IV', 'GBC', 'DN5', 'DN10', 'COf1', 'COfi', 'COh', 'COt', 'MD', 'SB', 'SBT', 'PD', 'COE', 'IN', 'FCm', 'DNOp', 'DNOn', 'SPa', 'SBb', 'SBm', 'FCr', 'FCd', 'SPc', 'FCs', 'ID']
        # Add catch22 features
        catch22_features = calc_catch22_features(X_in)
        calc_catch22_features_names = ['DN5', 'DN10', 'COf1', 'COfi', 'COh', 'COt', 'MD', 'SB', 'SBT', 'PD', 'COE', 'IN', 'FCm', 'DNOp', 'DNOn', 'SPa', 'SBb', 'SBm', 'FCr', 'FCd', 'SPc','FCs'] 
        for i, value in enumerate(catch22_features):
            features[f'{calc_catch22_features_names[i]}'] = value
        # Reorder the features
        features = {key: features[key] for key in order}
        # print(features)
        return features
    
    def _expand_data_with_classifiers(self, X_in):
        """
        Expand the input data to include one-hot encoding for each classifier.

        Args:
            X_in (pd.DataFrame): Input data containing feature vectors.

        Returns:
            pd.DataFrame: Expanded data with duplicated rows for each classifier, excluding rows with no active classifiers.
        """
        expanded_data = []

        # Iterate over all classifiers, not just the selected ones
        for clf in self.all_classifiers:
            # Create a copy of the data for each classifier and add a one-hot encoded column for the classifier
            data_copy = X_in.copy()
            data_copy[f'model_{clf}'] = 1 if clf in self.classifiers else 0
            expanded_data.append(data_copy)

        # Concatenate all the expanded data into a single DataFrame
        result = pd.concat(expanded_data, ignore_index=True)

        # Filter out rows that have all zeros in the classifier columns
        classifier_columns = [f'model_{clf}' for clf in self.all_classifiers]
        result = result[(result[classifier_columns].sum(axis=1) > 0)]

        return result
    

    def fit(self, X_in, y_in):
        X_in = self._prepare_data(X_in)
        features = self._extract_features(X_in)
        X_expanded = self._expand_data_with_classifiers(pd.DataFrame([features]))

        for method in self.methods:
            X_train = X_expanded.copy()
            y_train = y_in
            self.models[method].fit(X_train, y_train)

    def predict(self, X_in, full_results=False):
        """
        Predict the improvement in F1 score for each augmentation method.

        Args:
            X_in (np.ndarray or pd.DataFrame): Input dataset for which predictions are made.
            full_results (bool): If True, returns all predictions for each classifier and method.
                                If False, returns the ranking of methods for each classifier.

        Returns:
            pd.DataFrame: Either a full results DataFrame or a ranked DataFrame based on predictions.
        """
        # Check if any classifiers are selected
        if not self.classifiers:
            print("No classifiers selected. Skipping predictions.")
            return pd.DataFrame()

        # Prepare the data and extract features
        X_in = self._prepare_data(X_in)
        features = self._extract_features(X_in)
        X_expanded = self._expand_data_with_classifiers(pd.DataFrame([features]))
        X_expanded.fillna(0, inplace=True)

        predictions = []
        for method in self.methods:
            X_test = X_expanded.copy()
            y_pred = self.models[method].predict(X_test)

            # Create a DataFrame to hold predictions, method, and classifiers
            classifiers = np.repeat(self.classifiers, len(y_pred) // len(self.classifiers))
            y_pred_df = pd.DataFrame({
                'Δ F1 score predicted': y_pred,
                'method': method,
                'classifier': classifiers
            })

            # Aggregate the predictions to get a single value per method-classifier pair
            if not full_results:
                y_pred_df = y_pred_df.groupby(['method', 'classifier'])['Δ F1 score predicted'].mean().reset_index()

            predictions.append(y_pred_df)

        # Combine predictions from all methods
        results = pd.concat(predictions, ignore_index=True)

        if full_results:
            # Pivot the DataFrame to have classifiers as columns and methods as index
            results_pivot = results.pivot(index='method', columns='classifier', values='Δ F1 score predicted')
            results_pivot = results_pivot.sort_index(axis=1) 
            return results_pivot
        else:
            # Rank the techniques for each classifier
            results['rank'] = results.groupby('classifier')['Δ F1 score predicted'].rank(method='dense', ascending=False)
            results = results.sort_values(['classifier', 'method'])
            results = results.reset_index(drop=True)

            # Reshape the results to have classifiers as columns and methods as the index
            results_pivot = results.pivot(index='method', columns='classifier', values='rank')
            results_pivot = results_pivot.sort_index(axis=1)  
            return results_pivot