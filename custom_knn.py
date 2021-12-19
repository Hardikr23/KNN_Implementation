def custom_knn(x_train, x_test, y_train, y_test, n_neighbors = 1):
    """
    This function is a custom implementation of the knn algorithm. 
    It accepts the training and testing splits of a datasets and returns 
    number of errors and the error rate on the test set.
    
    Parameters
    ----------
    x_train : numpy.ndarray
        This is a numpy array which contains all the training samples
    x_test : int
        This is a numpy array which contains all the testing samples
    y_train : int
        This is a numpy array which contains all the training labels
    y_test : int
        This is a numpy array which contains all the testing labels
    n_neighbors : int
        Default value = 1
        This is the number of neighbors we want to consider while predicting the label for a test sample

    Returns
    -------
    Returns the number of errors it makes on the test set and the 
    test error ratenumber of errors it makes on the test set and 
    the test error rate
    """
    # external imports
    import sys
    import numpy as np

    # helper function imports
    from helper_functions import calc_distance, find_index_min_distance, max_occured_label

    predicted_labels = []
    y_pred = []
    mis_match_count = 0
    for test_sample in x_test:
        distance = calc_distance(test_sample, x_train)
        index_min_distance = find_index_min_distance(distance, n_neighbors)
        predicted_labels.append([test_sample,index_min_distance])

    for item in predicted_labels:
        item[1] = [y_train[label] for label in item[1]]
        y_pred.append(max_occured_label(item[1]))
        
    for i in range(len(y_pred)):
        if y_pred[i] != y_test[i]:
            mis_match_count += 1
    return(n_neighbors, mis_match_count, np.mean(y_pred != y_test))