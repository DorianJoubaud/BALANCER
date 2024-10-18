from scipy.stats import kendalltau
import numpy as np
import pandas as pd

def get_kendalltau(data, pred):
    """Calculate Kendall's Tau rank correlation for a single dataset."""
    # print(data['classifier'].unique())  

    scores = {model: [] for model in data.columns.unique()}  # Handle available models dynamically
    

    for model in data.columns.unique():
        
        if model not in pred.columns.unique():
            continue  # Skip if the model isn't present in the predictions

        data_model = data[model]
        pred_model = pred[model]
        

        if data_model.empty or data_model.nunique() == 1:
            continue  # Skip if there is no variance in the data model values

        # Compute rank correlation between actual and predicted values
        rank = data_model.rank(method='dense', ascending=False).tolist()
        pred_rank = pred_model.rank(method='dense', ascending=False).tolist()
        
        
       
        
        tau, _ = kendalltau(rank, pred_rank)

        scores[model].append(tau)
        

    # #return a dataframe
    return pd.DataFrame(scores, index=['kendall_tau'])

def get_rmse(data, pred):
    """Calculate Root Mean Squared Error (RMSE) for a single dataset for each model."""
    scores = {model: [] for model in data.index.unique()}  # Handle available models dynamically

    for model in data.index.unique():
        if model not in pred.index.unique():
            continue  # Skip if the model isn't present in the predictions

        data_model = data.loc[model]
        pred_model = pred.loc[model]

        if data_model.empty:
            continue  # Skip if no data for this model

        # Calculate RMSE for the current model
        rmse = np.sqrt(np.mean((data_model - pred_model) ** 2))

        scores[model].append(rmse)

    # Return the scores as a DataFrame
    return pd.DataFrame(scores, index=['RMSE'])

def get_mean_rank(model, data):
    """Calculate the mean rank for a specific model in a single dataset."""
    data_model = data[data['model'] == model]
    ranks = data_model['value'].rank(method='dense', ascending=False)
    return ranks.mean()



