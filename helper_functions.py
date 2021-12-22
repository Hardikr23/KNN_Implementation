def max_occured_label(label_list):
    """
    This function basically implements voting for the case of a knn 
    classification algorithm
    It Accepts the list of N predicted labels and returns the label 
    which has occured the maximum times.
    
    Parameters
    ----------
    label_list : list
        It is a list of lists where the all sub lists contains 2 elements.
        The first element is the test sample and the 2nd element is a list which has n nearest neighbors

    Returns
    -------
    max_key : int
        returns the label which has the highest number of occurence in the n nearest neighbors
    """
    label_count = {}
    for item in label_list:
        if item in label_count.keys():
            label_count[item] += 1
        else:
            label_count[item] = 1
    max_val = 0
    max_key = ""
    for key in label_count.keys():
        if label_count[key] > max_val:
            max_val = label_count[key]
            max_key = key
    return max_key

def find_index_min_distance(distance, n_neighbors):
    """
    This function accepts the list of distances of a single test sample 
    with all other training samples and the number of neighbors that we 
    need to consider and returns a list of indexes of N shortest distances
    
    Parameters
    ----------
    distance : int
        A list of dstances of each training sample from the test sample
        
    n_neighbors : int
        This is the number of nearest neighbors we want to consider 
        to predict the label of the test sample
    
    Returns
    -------
    _min_index_list : list
        It is a list of indexes which have the least n neighbors distance
    """
    import math
    import numpy as np

    _min_index_dict = {}
    _min_index_list = []
    for i in range(len(distance)):
            _min_index_dict[i]=distance[i]
    for n in range(n_neighbors):
        _min = math.inf
        least_dist_key = 0
        for keys in _min_index_dict.keys():
            if _min_index_dict[keys] != None:
                if _min_index_dict[keys] < _min:
                    _min = _min_index_dict[keys]
                    least_dist_key = keys
        _min_index_dict[least_dist_key] = None
    for keys in _min_index_dict.keys():
        if _min_index_dict[keys] == None:
            _min_index_list.append(keys)
        
    return _min_index_list   

def calc_distance(test_sample, x_train):

    """
    This function is responsible for calculating the distance of the test
    sample with all the training samples and returns a list of it
    
    Input
    train_sample : one of the test sample
    x_train : entire training set
    
    Returns: list
    A list of dstances of each training sample from the test sample
    """
    import math
    from Levenshtein import distance as lev

    distance =[]
    for train_sample in x_train:
        _distance = 0
        for i in range(len(test_sample)):
            if isinstance(test_sample[i], int): # to calculate distance between numeric columns
                _distance += (test_sample[i]-train_sample[i])**2
            else :
                # Levingston distance
                
                lev_dist = lev(test_sample[i], train_sample[i])
                _distance += lev_dist**2
        distance.append(math.sqrt(_distance))
    return distance

def split_dataset(features, labels, rs):
    """
    This function is responsible for splitting the data into training
    and testing sets. It uses the train_test_split function from sklearn
    package. It accepts the features and labels and returns training and 
    testing  sets for samples and labels.
    
    Parameters
    ----------
    features : numpy.ndarray
        This a numpy array of all the features made from iris_db['data'].
    labels : numpy.ndarray
        This a numpy array of all the features made from iris_db['target'].
    rs : int
        This an arbitrary integer value to set the seed at a fixed point to
        get identical split results every time we run it.

    Returns
    -------
    Returns 4 parameters viz. x_train, y_train, x_test, y_test
    These are splits of the dataset as per industry standards(75% training, 
    25% testing) for training and testing sets.
    """
    from sklearn.model_selection import train_test_split
    return(train_test_split(features, labels, random_state = rs))