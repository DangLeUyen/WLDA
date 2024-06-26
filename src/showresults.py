import numpy as np
import pandas as pd
from itertools import combinations
from experiment import decision_boundary_experiment
from funcs import cosine_similarity
import os


def show_accuracy_time(data_name, res, missing_test):
    directory = '/Volumes/Macintosh HD/Projects/PAPER/WLDA/src/results/performance/'
    # Check if the directory exists
    if not os.path.exists(directory):
        # Create the directory
        os.makedirs(directory)

    #print(f"Dataset: {data_name} \n")
    pm = pd.DataFrame(np.repeat(" ± ",25).reshape(5,5),
                    index = ["15%", "30%", "45%", "60%", "75%"],
                    columns = ["WLDA", "KNNI", "MICE", "Soft-Impute", "DIMV"])
    #print("The accuracy of KNN, MICE, SOFT-IMPUTE, DIMV, WLDA:")
    acc_avg = pd.DataFrame(np.vstack((np.mean(res[0][0], axis = 0), np.mean(res[1][0], axis = 0),
                        np.mean(res[2][0], axis = 0),np.mean(res[3][0], axis = 0),np.mean(res[4][0], axis = 0))).round(3),
                        index = pm.index,
                        columns = pm.columns)
    acc_std = pd.DataFrame(np.vstack((np.std(res[0][0], axis = 0), np.std(res[1][0], axis = 0),
                        np.std(res[2][0], axis = 0),np.std(res[3][0], axis = 0),np.std(res[4][0], axis = 0))).round(3),
                        index = pm.index,
                        columns = pm.columns)
    accuracydf =  acc_avg.astype(str)+pm+acc_std.astype(str)
    # Save the DataFrame to a CSV file
    file_path = os.path.join(directory, f'{int(missing_test)}/{data_name}_accuracy.csv')

    accuracydf.to_csv(file_path)
    print(f"{data_name} accuracy saved")

    time_avg = pd.DataFrame(np.vstack((np.mean(res[0][1], axis = 0), np.mean(res[1][1], axis = 0),
                        np.mean(res[2][1], axis = 0),np.mean(res[3][1], axis = 0),np.mean(res[4][1], axis = 0))).round(3),
                        index = pm.index,
                        columns = pm.columns)
    time_std = pd.DataFrame(np.vstack((np.std(res[0][1], axis = 0), np.std(res[1][1], axis = 0),
                        np.std(res[2][1], axis = 0),np.std(res[3][1], axis = 0),np.std(res[4][1], axis = 0))).round(3),
                        index = pm.index,
                        columns = pm.columns)
    timedf = time_avg.astype(str)+pm+time_std.astype(str)

        # Save the DataFrame to a CSV file
    file_path = os.path.join(directory, f'{int(missing_test)}/{data_name}_runtime.csv')
    timedf.to_csv(file_path)

    print(f"{data_name} running time saved")


def show_boundary_similarity(Ss, Ms, y, models, missing_range):
    classes = np.unique(y)
    boundaries = [(cls1, cls2) for cls1, cls2 in set(list(combinations(classes, 2)))]
    index = pd.MultiIndex.from_product([boundaries, models],
                            names=['boundary', 'models'])
    pm = pd.DataFrame(columns=missing_range, index=index)
    for mr in missing_range:
        weights = {}
        S, M = Ss[mr], Ms[mr] 
        for i in range(len(S[0])):
            weights[i] = decision_boundary_experiment(S[0][i], M[0][i], y)[1]

        for j, w_values in weights.items():
            temp = []
            for (cls1, cls2) in w_values.keys():
                temp= cosine_similarity(weights[0][(cls1, cls2)], weights[j][(cls1, cls2)])
                pm.loc[((cls1, cls2),models[j]),mr] = temp
    return pm
    