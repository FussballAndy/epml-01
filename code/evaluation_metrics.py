import numpy as np

def confusion_matrix(test, predicted):
    """
    Returns a tuple containing the confusion matrix components given an array of actual
    classes and an array of predicted classes.
    
    Args:
        test: boolean np.darray of actual classes for each instance of the test dataset
        predicted: boolean np.darray of predicted classes for each instance of the test dataset

    Returns:
        Tuple of number of true negatives, false negatives, true positives, and false positives
    """

    TN = np.sum(~test & ~predicted) # Number of true negatives
    FN = np.sum(test & ~predicted) # Number of false negatives
    TP = np.sum(test & predicted) # Number of true positives
    FP = np.sum(~test & predicted) # Number of false negatives

    return TN, FN, TP, FP

def binary_evaluation(test, predicted):
    """
    Args:
        test: np.darray of actual classes for each instance of the test dataset
        predicted: np.darray of predicted classes for each instance of the test dataset
    
    Returns:
        A dictionary containing accuracy, precision, recall and f1 score for given binary predictions from a classification model.
    """

    # Convert to numpy boolean arrays
    test_bool = test == 1
    predicted_bool = predicted == 1

    TN, FN, TP, FP = confusion_matrix(test_bool, predicted_bool)
    
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = (2 * precision * recall) / (precision + recall)

    return { "accuracy": accuracy, "precision": precision,  "recall": recall, "f1_score": f1_score }