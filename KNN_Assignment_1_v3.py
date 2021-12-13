#!/usr/bin/env python
# coding: utf-8

# <h1>Custom Implementation of KNN Algorithm</h1> <br>
# 
# The first section is implementation of KNN Algorithm with generalised value of K.<br>
# So we can use that for k=1 and for other values of K too.
# 
# The second section is just 1NN implementation
# 
# In this project we explore 2 datasets:<br>
# 1. IRIS dataset<br>
# 2. ionosphere dataset<br>
# 
# Tasks performed for each dataset are as follows :
# 1. Loading the datasets
# 2. Exploring the datasets
# 3. Implementing K Nearest Neighbor on both datasets
# 4. Implementing Nearest Neighbor on both datasets
# 
# Additionally, I have added my comments with respect to different observations made while performing the above tasks
# 
# <b>Note : </b>Please replace the path of the ionosphere.txt while loading the dataset (section 2.1 and in 1NN implementation)

# <h1>1. Iris Dataset</h1>

# <h2>1.1 Importing the IRIS Dataset</h2>

# In[1]:


# import required packages to 
# load iris dataset : the dataset on which we want to implement KNN
from sklearn.datasets import load_iris
iris_db = load_iris()
import math


# <h2>1.2 Exploring the IRIS dataset</h2>

# In[2]:


print(type(iris_db))
print(type(iris_db['data']))
print(iris_db['data'].shape)


# We can see in the above output we can observe the types of different data that we will be using in our next steps.<br>
# Next we will have a quick look at the the keys and the data present in each value of the key

# In[3]:


print(iris_db.keys())


# In[4]:


print(iris_db['target'])


# Above we can see the value for the target key. These are basically the indexes which map to the predicted labels

# In[5]:


print(iris_db['target_names'])


# In[6]:


print(iris_db['feature_names'])


# <h3>1.3 Loading the Features and the Labels</h3>

# In[8]:


features = iris_db['data']
labels = iris_db['target']
print(type(features))
print(type(labels))


# Since the number of features are small in the iris dataset we can afford to visually plot each feature with every other feature and check that which feature combination will be best for classification

# <h3>1.4 Plotting the dataset for visual analysis</h3>
# Here we will try to plot a simple scatter plot of the features of the dataset to get an understanding that which features give us the best classifiication of the different species of the iris flower

# In[9]:


def scatter_plot(i, j, x, y, labels, title):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot()
    ax.scatter(x,y,c=labels)
    ax.set_xlabel(i)
    ax.set_ylabel(j)


# In[10]:


for i in range(4):
    for j in range(4):
        if i != j:
            scatter_plot(iris_db["feature_names"][i],iris_db["feature_names"][j], features[:,i], features[:,j], labels, "This graph is of {} vs {}".format(iris_db['feature_names'][i], iris_db['feature_names'][j]))


# From the aboves graphs we can draw some observations :
# Graph of 
# 1. Sepal length x petal length
# 2. petal_width x petal length 
# 3. petal length x petal width
# 
# Show good distinction of each category of the flower.
# Basically the clusters are more concentrated, which is helpful while making prediction about the new input data.

# <h1>2. Ionosphere Dataset </h1>

# <h2>2.1 Importing the Ionosphere Dataset</h2>
# Note : Please replace the path of the ionosphere.txt

# In[11]:


import numpy as np

ionosphere_path = "path/to/file/ionosphere.txt"
ion_all_cols = np.genfromtxt(ionosphere_path, delimiter = ",", usecols = np.arange(35))


# <h2>1.2 Exploring the Ionosphere dataset</h2>

# In[12]:


print(type(ion_all_cols))


# In[13]:


#Let's print the first 5 rows of the dataset and see
print(ion_all_cols[:5])


# As we can see in the above output that all the columns have quantitave data.
# From the dataset definition, we know that the last column of the dataset is the label and zll other columns are feature columns<br>
# Unlike the Iris dataset, here we have the features, labels, etc present all in one numpy array only, as we have loaded it directly from a txt file stored on our local machine

# <h3>1.2.1 Checking for NaN values in each column</h3>

# In[14]:


cols_check = []
for i in range(35):
    # We calculate the sum of each column. If a column 
    # has even one NaN value then the sum will be NaN.
    col_sum = np.sum(ion_all_cols[i])
    cols_check.append(np.isnan(col_sum))
print(cols_check)


# In the above output we can see that all the columns return False for the NaN check.<br>
# So we can say that there are no missing values in the dataset

# <h3>1.2.2 Exploring the stats of each Column</h3>

# In[15]:


# We import stats from scipy package to get a brief idea of all the columns that we have in our dataset
from scipy import stats
# We run the stats.describe with axis 1 tp run it on all the columns
ion_data_stats = stats.describe(ion_all_cols[:34], axis = 1)
print(ion_data_stats)
# We get an object in return which basically gives us different stats metrics on a columns level.


# <h3>Let's explore at a granular level</h3>

# In[16]:


print("The total number of columns : {}".format(ion_data_stats.nobs))


# In[17]:


print("The minimum and maximum values of each of the colums\n")
print("Minimum Values : {}\n".format(ion_data_stats.minmax[0]))
print("Maximum Values : {}".format(ion_data_stats.minmax[1]))


# Here we can see that all the columns have values between -1 and 1.

# In[18]:


print("Mean of a each column : \n{}\n".format(ion_data_stats.mean))
print("Variance of each column : \n{}".format(ion_data_stats.variance))


# We can observe that the variance of each column is very close to 0. Infact, none of them have a variance more than 1, so we can say that all the values are belonging in a very compact range.<br>
# Also, we can say that none of the columns have a range which is remarkably different than the other columns as their mean belong to the same range, -1 to 1. 
# This is in line with min max values in the above cells as all the values are between -1 and 1.
# Hence there will no need to regularize any of the values to match with other columns

# <h3>3. Splitting the dataset</h3>
# <br>
# We use the <b>train_test_split</b> function which helps to split the dataset into training and testing splits as per industry standards i.e. 75% training and 25% testing

# In[19]:


def split_dataset(features, labels, rs):
    """
    This function is responsible for splitting the data into training
    and testing tests. It uses the train_test_split function from sklearn
    package. It accepts the features and labels and retruns training and 
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


# <h4>3.1 Splitting IRIS Dataset</h4>

# In[20]:


x_train, x_test, y_train, y_test = split_dataset(features, labels, 311)


# In[21]:


# Have a quick look at the training samples and labels
print("x_train :\n",x_train[:5])
print("y_train :\n",y_train[:5])


# <h4>3.2 Splitting the Ionosphere Dataset</h4>

# In[22]:


features_ion = np.genfromtxt(ionosphere_path, delimiter = ",", usecols = np.arange(34))
labels_ion = np.genfromtxt(ionosphere_path, delimiter = ",", usecols = 34, dtype = 'int')
xi_train, xi_test, yi_train, yi_test = split_dataset(features_ion, labels_ion, 311)


# <h3>4. Helper Funtions</h3>

# In[23]:


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
    distance =[]
    for train_sample in x_train:
        _distance = 0
        for i in range(len(test_sample)):
            _distance += (test_sample[i]-train_sample[i])**2
        distance.append(math.sqrt(_distance))
    return distance


# In[24]:


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
            


# In[25]:


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


# In[26]:


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
    import sys
    import numpy as np
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


# <h3>5. Running KNN on the Datasets</h3>

# In[27]:


import time

# Running knn on the iris dataset
iris_start = time.time()
print("Iris Dataset")
n_neighbors, mis_match_count, error_rate = custom_knn(x_train, x_test, y_train, y_test, n_neighbors = 1)
print("Neighbors : {}\nNumber of mis matched labels : {} \nError Rate :{}\nPercentage of correct Predictions : {}".format(n_neighbors,mis_match_count,error_rate,(1-error_rate)*100))
print("total time : ",time.time()-iris_start,"s")

# Running knn on the ionosphere dataset
ion_start = time.time()
print("\nIonosphere Dataset")
n_neighbors, mis_match_count, error_rate = custom_knn(xi_train, xi_test, yi_train, yi_test, n_neighbors = 10)
print("Neighbors : {}\nNumber of mis matched labels : {} \nError Rate :{}\nPercentage of correct Predictions : {}".format(n_neighbors,mis_match_count,error_rate,(1-error_rate)*100))
print("total time : ",time.time()-ion_start,"s")


# <h4>Plotting Error Rate for different K values on Iris</h4>

# In[28]:


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
fig = plt.figure(figsize=(20, 5))
ax = plt.axes()

k_vals = []
error_rates = []
for i in range(1,101):
    n_neighbors, mis_match_count, err_rate = custom_knn(x_train, x_test, y_train, y_test, n_neighbors = i)
    k_vals.append(i)
    error_rates.append(err_rate)
ax.set_xlabel("k-values")
ax.set_ylabel("error_rate")
ax.xaxis.set_ticks(np.arange(0,105,2))
figure = ax.plot(k_vals, error_rates);


# Above is the graph of error rates for values of n_neighbors ranging from 1 to 50 on the Iris dataset.<br>
# We can observe that the least error rate is acheived multile times when n_neighbors is between 8 to 35 inclusive.<br>
# But once it crosses 35, the error rate only keeps on increasing.<br>
# 
# We can conclude that for the IRIS dataset, an optimum n_neighbor value between 8 to 35 (precisely : 8,12,14,16-19,33-35) for random state 311

# <h4>Plotting Error Rate for different K values on Iris</h4>

# In[29]:


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(20, 5))
ax = plt.axes()

k_vals = []
error_rates = []
for i in range(1,101):
    n_neighbors, mis_match_count, err_rate = custom_knn(xi_train, xi_test, yi_train, yi_test, n_neighbors = i)
    k_vals.append(i)
    error_rates.append(err_rate)
    
ax.set_xlabel("k-values")
ax.set_ylabel("error_rate")
ax.yaxis.set_ticks(np.arange(0,0.5,0.05))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.03f'))
ax.xaxis.set_ticks(np.arange(0,105,2))
figure = ax.plot(k_vals, error_rates);


# Above is the graph of error rates for values of n_neighbors ranging from 1 to 100 on the Ionosphere dataset.
# We can observe that the least error rate is acheived when n_neighbors is 1.
# The error rate only keeps on increasing as we increase the number of n-neighbors
# 
# We can conclude that for the Ionosphere dataset, an optimum n_neighbor value 1 for random state 311

# <h1>1-NN Implementation</h1><br>
# Below you can find the code for 1NN for iris and ionosphere datasets.

# In[30]:


def nn_calc_distance(test_sample, x_train):
    """
    Input
    train_sample : one of the test sample
    x_train : entire training set
    
    Returns: list
    a list of dstances of each training sample from the test sample
    """
    import math
    distance =[]
    for train_sample in x_train:
        _distance = 0
        for i in range(len(test_sample)):
            _distance += (test_sample[i]-train_sample[i])**2
        distance.append(math.sqrt(_distance))
    return distance


# In[31]:


def nn_find_index_min_distance(distance):
    """
    Input : distance
    Descr : it is a list of all distances from a test sample to every train sample
    
    Return : int
    retrurns the index of the trainingg sample with minimum distance
    """
    _min = distance[0]
    min_index = 0
    for i in range(len(distance)):
        if distance[i] < _min:
            _min = distance[i]
            min_index = i
    return min_index


# In[32]:


def one_nn(x_train, x_test, y_train, y_test):
    import sys
    import numpy as np
    predicted_labels = []
    y_pred = []
    nn_mis_match_count = 0
    for test_sample in x_test:
        distance = nn_calc_distance(test_sample, x_train)
        index_min_distance = nn_find_index_min_distance(distance)
        predicted_labels.append([test_sample,index_min_distance])
    for elem in predicted_labels:
        y_pred.append(y_train[elem[1]])
    for i in range(len(y_pred)):
        if y_pred[i] != y_test[i]:
            nn_mis_match_count += 1
    return (nn_mis_match_count, np.mean(y_pred != y_test))


# In[33]:


start_1_iris= time.time()
nn_mis_match_count, error_rate = one_nn(x_train, x_test, y_train, y_test)
print("Iris\n")
print("Neighbors : 1\nNumber of mis matched labels : {} \nError Rate :{}\nPercentage of correct Predictions : {}".format(nn_mis_match_count,error_rate,(1-error_rate)*100))
print("total time : ",time.time()-start_1_iris)


# In[34]:


import numpy as np
# Please replace the path of the ionosphere.txt 
ionosphere_path = "path/to/file/ionosphere.txt"
features_ion = np.genfromtxt(ionosphere_path, delimiter = ",", usecols = np.arange(34))
labels_ion = np.genfromtxt(ionosphere_path, delimiter = ",", usecols = 34, dtype = 'int')


# In[35]:


start_1_ion= time.time()
nn_mis_match_count, error_rate = one_nn(xi_train, xi_test, yi_train, yi_test)
print("Ionosphere\n")
print("Neighbors : 1\nNumber of mis matched labels : {} \nError Rate :{}\nPercentage of correct Predictions : {}".format(nn_mis_match_count,error_rate,(1-error_rate)*100))
print("total time : ",time.time()-start_1_ion)


# <h3>fin.</h3>
