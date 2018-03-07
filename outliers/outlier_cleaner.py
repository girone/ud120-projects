#!/usr/bin/python
import numpy as np

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """

    # your code goes here
    errors = abs(predictions.flatten() - net_worths.flatten())
    # Create indices that would sort the error terms, keep only 90%
    keep = errors.argsort()[:int(errors.size * 0.9)]
    
    cleaned_data = []
    for index in keep:
        cleaned_data.append((ages[index], net_worths[index], errors[index]))

    
    return cleaned_data

