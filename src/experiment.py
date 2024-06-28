from funcs import generate_NaN
from wlda import WLDA
from mylda import LDA
from itertools import combinations
from DIMVImputation import DIMVImputation
from fancyimpute import SoftImpute, KNN
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import time
from shapvalues import ShapleyEstimator

def training_test_performance_experiment(X,y,missing_rate,run_time):
    """
    Perform an experiment to evaluate training and test performance of models
    with missing data which is present in both training data and test data.

    Parameters:
    X : (numpy.ndarray) Input features of shape (n_samples, n_features).
    y : (numpy.ndarray) Target labels of shape (n_samples,).
    missing_rate : float - Proportion of missing data
    run_time : int, optional (default=10)
        Number of runs to repeat the experiment.

    Returns:
    tuple
        Tuple containing:
        - acc: List of accuracies for each run.
        - tim: List of times (in seconds) for each run.
    """
    acc = [] # accuracy
    tim = [] # running time
    p = X.shape[1]

    for rt in range(run_time):
        # Generate missing data
        Xnan = generate_NaN(X,missing_rate)
        
        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(Xnan, y, test_size = 0.2, random_state = 0)

        # KNN
        t = time.time()
        knn_imputer = KNN(k=np.sqrt(p).astype(int))
        X_train_knn = knn_imputer.fit_transform(X_train)
        X_test_knn = knn_imputer.fit_transform(X_test)

        knn_model = LDA()
        knn_model.fit(X_train_knn, y_train)
        acc_knn = accuracy_score(y_test, knn_model.predict(X_test_knn).flatten())
        time_knn = time.time() - t

        #MICE
        t = time.time()
        iter_imputer = IterativeImputer(max_iter = 10)
        X_train_mice = iter_imputer.fit_transform(X_train)
        X_test_mice = iter_imputer.transform(X_test)
        mice_model = LDA()
        mice_model.fit(X_train_mice, y_train)
        acc_mice = accuracy_score(y_test, mice_model.predict(X_test_mice).flatten())
        time_mice = time.time() - t

        #SOFT
        t = time.time()
        soft_imputer = SoftImpute(max_iters = 10, verbose = False)
        X_train_soft = soft_imputer.fit_transform(X_train)
        X_test_soft = soft_imputer.fit_transform(X_test)
        soft_model = LDA()
        soft_model.fit(X_train_soft, y_train)
        acc_soft = accuracy_score(y_test, soft_model.predict(X_test_soft).flatten())
        time_soft = time.time() - t

        #DIMV
        t = time.time()
        imputer = DIMVImputation()
        imputer.fit(X_train, initializing=False)
        X_train_dimv = imputer.transform(X_train)
        X_test_dimv = imputer.transform(X_test)
        dimv_model = LDA()
        dimv_model.fit(X_train_dimv, y_train)
        acc_dimv = accuracy_score(y_test, dimv_model.predict(X_test_dimv).flatten())
        time_dimv = time.time() - t

        #WLDA
        t = time.time()
        wlda = WLDA()
        wlda.fit(X_train, y_train)
        acc_wlda = accuracy_score(y_test, wlda.predict(X_test).flatten())
        time_wlda = time.time() - t

        # Append test accuracy
        acc.append([acc_wlda,acc_knn,acc_mice,acc_soft,acc_dimv])
        # Append training time
        tim.append([time_wlda,time_knn,time_mice,time_soft,time_dimv])

    return acc, tim

def training_performance_experiment(X,y,missing_rate,run_time):
    """
    Perform an experiment to evaluate training and test performance of models
    with missing data which is present in only training data.

    Parameters:
    X : (numpy.ndarray) Input features of shape (n_samples, n_features).
    y : (numpy.ndarray) Target labels of shape (n_samples,).
    missing_rate : float - Proportion of missing data
    run_time : int, optional (default=10)
        Number of runs to repeat the experiment.

    Returns:
    tuple
        Tuple containing:
        - acc: List of accuracies for each run.
        - tim: List of times (in seconds) for each run.
    """
    acc = [] #accuracy
    tim = [] #running time
    p = X.shape[1]
    for rt in range(run_time):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        X_train = generate_NaN(X_train,missing_rate)

        # KNN
        t = time.time()
        knn_imputer = KNN(k=np.sqrt(p).astype(int))
        X_train_knn = knn_imputer.fit_transform(X_train)

        knn_model = LDA()
        knn_model.fit(X_train_knn, y_train)
        acc_knn = accuracy_score(y_test, knn_model.predict(X_test).flatten())
        time_knn = time.time() - t

        #MICE
        t = time.time()
        iter_imputer = IterativeImputer(max_iter = 10)
        X_train_mice = iter_imputer.fit_transform(X_train)
        mice_model = LDA()
        mice_model.fit(X_train_mice, y_train)
        acc_mice = accuracy_score(y_test, mice_model.predict(X_test).flatten())
        time_mice = time.time() - t

        #SOFT
        t = time.time()
        soft_imputer = SoftImpute(max_iters = 10, verbose = False)
        X_train_soft = soft_imputer.fit_transform(X_train)
        soft_model = LDA()
        soft_model.fit(X_train_soft, y_train)
        acc_soft = accuracy_score(y_test, soft_model.predict(X_test).flatten())
        time_soft = time.time() - t

        #DIMV
        t = time.time()
        imputer = DIMVImputation()
        imputer.fit(X_train, initializing=False)
        X_train_dimv = imputer.transform(X_train)
        dimv_model = LDA()
        dimv_model.fit(X_train_dimv, y_train)
        acc_dimv = accuracy_score(y_test, dimv_model.predict(X_test).flatten())
        time_dimv = time.time() - t

        #WLDA
        t = time.time()
        wlda = WLDA()
        wlda.fit(X_train, y_train)
        acc_wlda = accuracy_score(y_test, wlda.predict(X_test).flatten())
        time_wlda = time.time() - t

        acc.append([acc_wlda,acc_knn,acc_mice,acc_soft,acc_dimv])
        tim.append([time_wlda,time_knn,time_mice,time_soft,time_dimv])

    return acc, tim

def run_performance_experiment(X, y, missing_rate, runtime, missing_id):
  res = []
  if missing_id == 0:
     for r in missing_rate:
        res.append(training_performance_experiment(X,y,r,runtime))
  else:
     for r in missing_rate:
        res.append(training_test_performance_experiment(X,y,r,runtime))

  return res

def shapley_values_experiment(model, X_train, X_test):
   """Return:
        Dictionary: key - class
                    value - average of shapley values of subsamples which are predicted to key value"""
   pred = model.predict(X_test)
   # Create an instance of ShapleyEstimator
   shapley_estimator = ShapleyEstimator(model.predict_proba, X_train)
   explanation = shapley_estimator.shapley_values_arrays(X_test)

   return explanation, pred

def cov_shap_experiment(X,y,missing_rate):
    """
    Perform an experiment to evaluate covariance matrices under conditions of missing data.

    Parameters:
    X : (numpy.ndarray) Input features of shape (n_samples, n_features).
    y : (numpy.ndarray) Target labels of shape (n_samples,).
    missing_rate : (float) Proportion of missing data

    Returns:
    tuple
        Tuple containing:
        - Ss: List of estimated covariance matrices according to different methods.
        - Ms: List of estimated mean vectors according to different methods.
        - expls: dictionary of shapley values according to different methods.
        - preds: dictionary of predicted values of test data according to different methods.
    """
    p = X.shape[1]
    Ss = []
    Ms = []

    expls = {}
    preds = {}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    # Ground Truth
    grth_model = LDA()
    grth_model.fit(X_train, y_train)
    S0 = grth_model.get_covariance()
    m0 = grth_model.get_means()

    Xtrain_nan = generate_NaN(X_train,missing_rate)

    # KNN
    knn_imputer = KNN(k=np.sqrt(p).astype(int))
    X_train_knn = knn_imputer.fit_transform(Xtrain_nan)
    #Xtest_knn = knn_imputer.fit_transform(X_test)

    knn_model = LDA()
    knn_model.fit(X_train_knn, y_train)
    S_knn = knn_model.get_covariance()
    m_knn = knn_model.get_means()

    #MICE
    iter_imputer = IterativeImputer(max_iter = 10)
    X_train_mice = iter_imputer.fit_transform(Xtrain_nan)
    #Xtest_mice = iter_imputer.transform(X_test)

    mice_model = LDA()
    mice_model.fit(X_train_mice, y_train)
    S_mice = mice_model.get_covariance()
    m_mice = mice_model.get_means()
        
    #SOFT
    soft_imputer = SoftImpute(max_iters = 10, verbose = False)
    X_train_soft = soft_imputer.fit_transform(Xtrain_nan)
    #Xtest_soft = soft_imputer.fit_transform(X_test)

    soft_model = LDA()
    soft_model.fit(X_train_soft, y_train)
    S_soft = soft_model.get_covariance()
    m_soft = soft_model.get_means()
        
    #DIMV
    imputer = DIMVImputation()
    imputer.fit(Xtrain_nan, initializing=False)
    X_train_dimv = imputer.transform(Xtrain_nan)
    #Xtest_dimv = imputer.transform(X_test)

    dimv_model = LDA()
    dimv_model.fit(X_train_dimv, y_train)
    S_dimv = dimv_model.get_covariance()
    m_dimv = dimv_model.get_means()

    #WLDA
    wlda = WLDA()
    wlda.fit(Xtrain_nan, y_train)
    #S1_wlda = wlda.get_covariance()
    S_wlda = wlda.get_weight_covariance(X_test)
    m_wlda = wlda.get_means()

    Ss.append([S0, S_wlda, S_knn, S_mice, S_soft, S_dimv])
    Ms.append([m0, m_wlda, m_knn, m_mice, m_soft, m_dimv])

    # Shapley experiment
    expls['Ground Truth'], preds['Ground Truth'] = shapley_values_experiment(grth_model, X_train, X_test)
    expls['WLDA'], preds['WLDA'] = shapley_values_experiment(wlda, Xtrain_nan, X_test)
    expls['KNNI'] , preds['KNNI']= shapley_values_experiment(knn_model, X_train_knn, X_test)
    expls['MICE'] , preds['MICE'] = shapley_values_experiment(mice_model, X_train_mice, X_test)
    expls['Soft-Impute'], preds['Soft-Impute'] = shapley_values_experiment(soft_model, X_train_soft, X_test)
    expls['DIMV'], preds['DIMV'] = shapley_values_experiment(dimv_model, X_train_dimv, X_test)

    return Ss, Ms , expls, preds

def decision_boundary_experiment(cov, means, y): 
    """
    Perform an experiment to estimate the weights based on covariance matrices and mean vectors.

    Parameters:
    cov : (list of numpy.ndarray) List of estimated covariance matrices for different methods.
    means : (list of numpy.ndarray) List of estimated mean vectors for different methods.
    y : (numpy.ndarray) Target labels of shape (n_samples,).

    Returns:
    tuple
        Tuple containing:
        - w0s: the intercept
        - weights: Coefficients/weights of the features.
    """
    classes = np.unique(y)
    cov_inv = np.linalg.inv(cov)
    
    weights = {}
    w0s = {}
    for cls1, cls2 in set(list(combinations(classes, 2))):
        mean_diff = means[cls1] - means[cls2]
        w = np.round(np.dot(cov_inv, mean_diff),3)
        weights[(cls1, cls2)] = w

        # calculate w0
        term1 = np.dot(np.dot(means[cls1],cov_inv), means[cls1])
        term2 = np.dot(np.dot(means[cls2],cov_inv), means[cls2])
        w0 = np.round(0.5 * (term2 - term1) + np.log(np.sum(y==cls1)/np.sum(y==cls2)),3)

        w0s[(cls1, cls2)] = w0

    return w0s, weights

