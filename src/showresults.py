import numpy as np
import pandas as pd
from itertools import combinations
from experiment import decision_boundary_experiment
import os


def show_accuracy_time(data_name, res, missing_test):
    directory = ''
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

def show_coefs(Ss, Ms, y, column_names, models):
    classes = np.unique(y)
    boundaries = [(cls1, cls2) for cls1, cls2 in set(list(combinations(classes, 2)))]
    index = pd.MultiIndex.from_product([boundaries, models],
                            names=['boundary', 'models'])
    pm = pd.DataFrame(columns=column_names, index=index)
    w0_df =  pd.DataFrame(columns=['w0'], index=index)
    mr_df =  pd.DataFrame(columns=['missing_rate'], index=index)
    for mr in Ss.keys():
        S, M = Ss[mr], Ms[mr] 
        W0 = {}
        W = {}
        for i, model in enumerate(models):
            temp = decision_boundary_experiment(S[0][i], M[0][i], y)
            W0[model] = temp[0]
            W[model] = temp[1]

        for m, w_values in W.items():
            for (cls1, cls2) in w_values.keys():
                pm.loc[((cls1, cls2),m)] = w_values[(cls1, cls2)]
                w0_df.loc[((cls1, cls2),m)] = W0[m][(cls1, cls2)]

                pm.columns = column_names
                df = pd.concat([pm, w0_df], axis=1)
        
         # Save the DataFrame to a CSV file
        directory = ''
        # Check if the directory exists
        if not os.path.exists(directory):
            # Create the directory
            os.makedirs(directory)
        file_path = os.path.join(directory, f'{int(mr*100)}_coefs.csv')
        df.to_csv(file_path)

def show_wlda_coefs(Ss, Ms, y, column_names):
    classes = np.unique(y)
    boundaries = [(cls1, cls2) for cls1, cls2 in set(list(combinations(classes, 2)))]
    missing_range = Ss.keys()
    index = pd.MultiIndex.from_product([missing_range, boundaries],
                            names=['missing rate', 'boundary'])
    pm = pd.DataFrame(columns=column_names, index=index)
    w0_df =  pd.DataFrame(columns=['w0'], index=index)
    W0 = {}
    W = {}
    for mr in Ss.keys():
        temp = decision_boundary_experiment(Ss[mr][0][1], Ms[mr][0][1], y)
        W0[mr] = temp[0]
        W[mr] = temp[1]
        
    for m, w_values in W.items():
        for (cls1, cls2) in w_values.keys():
            pm.loc[(m, (cls1, cls2)),:] = w_values[(cls1, cls2)]
            w0_df.loc[(m, (cls1, cls2)), :] = W0[m][(cls1, cls2)]

            pm.columns = column_names
            df = pd.concat([pm, w0_df], axis=1)
        
         # Save the DataFrame to a CSV file
        # Check if the directory exists
    directory = ''
    file_path = os.path.join(directory, f'wlda_coefs.csv')
    df.to_csv(file_path)

