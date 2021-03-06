import time
def salary_predictor():
    import numpy as np
    import math
    import pandas as pd
    import sys
    from helper_functions import split_dataset
    from custom_knn import custom_knn

    # loading the data
    salary_path = "data/adult.txt"
    salary_all_cols = pd.read_csv(salary_path)
    salary_all_cols["encoded_label"] = salary_all_cols["income"] == '<=50K'
    salary_all_cols["encoded_label"] = salary_all_cols["encoded_label"].astype(int)
    salary_features = salary_all_cols.iloc[:, :14]
    salary_labels = salary_all_cols["encoded_label"]
    print(len(salary_labels))

    # Splitting the dataset with a random seed
    x_train, x_test, y_train, y_test = split_dataset(salary_features, salary_labels, 311)

    n_neighbors, mis_match_count, err_rate = custom_knn(x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy(), n_neighbors = 5)

    print("Neighbors : ",n_neighbors)
    print("Mis-match count :", mis_match_count)
    print("Error Rate :", err_rate)

if __name__ == "__main__":
    start_time = time.time()
    salary_predictor()
    print(time.time() - start_time)