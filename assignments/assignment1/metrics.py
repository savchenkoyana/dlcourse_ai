import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0
    
    tp = np.sum((ground_truth == True) * (prediction == True))
    fp = np.sum((ground_truth == False) * (prediction == True))
    tn = np.sum((ground_truth == False) * (prediction == False))
    fn = np.sum((ground_truth == True) * (prediction == False))

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    accuracy = (tp+tn)/(tp+fp+tn+fn)
    f1 = 2*precision*recall/(precision+recall)
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    
    return np.sum(ground_truth == prediction) / prediction.shape[0]
