import numpy as np


def evaluation_metric(predictions, labels):
    """
    Performance evaluation based on F-measures
    """
    TP = np.sum((labels == 1) & (predictions == 1))
    TN = np.sum((labels == 0) & (predictions == 0))
    FP = np.sum((labels == 0) & (predictions == 1))
    FN = np.sum((labels == 1) & (predictions == 0))

    precision = (TP / (TP + FP)).round(decimals=3)
    recall = (TP / (TP + FN)).round(decimals=3)  # TPR
    specificity = (TN / (TN + FP)).round(decimals=3)  # TNR

    bAccuracy = ((recall + specificity) / 2).round(decimals=3)
    f1 = (2 * TP / (2 * TP + FP + FN)).round(decimals=3)

    return precision, recall, bAccuracy, f1

