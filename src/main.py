from experiment import cov_shap_experiment, run_performance_experiment
from showresults import show_accuracy_time, show_wlda_coefs
from plot import *
from loaddata import *
import warnings
warnings.filterwarnings('ignore')

if __name__=='__main__':

    # List of models used for comparison in the analysis
    models = ['Ground Truth', 'WLDA', 'KNNI', 'MICE', 'Soft-Impute', 'DIMV']
    # List of missing data rates used in the experiment
    # Each value represents the percentage of data that is missing in the dataset
    missing_range = [0.15, 0.3, 0.45, 0.6, 0.75]
    #missing_range = [0.15]

    print('Input your request 0 or 1 (0 - performance and 1 - visualization):')
    request_id = input()
    request_id = int(request_id)

    if request_id == 1:
        X, y, column_names = load_irisData()

        # Set colormap for charts
        colors = ["#FF165D", "#FFFFFF", "#0A6EBD"]
        class_colors = ["#FF165D", "#A459D1", "#0A6EBD"]
        mse_colors = ["#FFFFFF", "#29ADB2"]
        swarm_colors = ["#FF165D", "#A459D1", "#0A6EBD", "#3AA6B9"]

        shapvalues = dict()
        preds = dict()
        Ss = dict()
        Ms = dict()

        for mr in missing_range:
            Ss[mr], Ms[mr], shapvalues[mr], preds[mr] = cov_shap_experiment(X,y,mr)

        ##### Plot decision boundary cosine similarity #######
        show_wlda_coefs(Ss, Ms, y, column_names)

        ##### Plot covariance heatmaps ########
            # correlation heatmaps
        all_mr_heatmaps(Ss, colors, t='correlation')
            # SE heatmaps
        all_mr_heatmaps(Ss, mse_colors, t='mse_corr')
            # subtraction heatmaps
        all_mr_heatmaps(Ss, colors, t='sub_corr')

        each_mr_heatmaps(Ss, colors, t='correlation')
            # each SE heatmaps according to different missing rate
        each_mr_heatmaps(Ss, mse_colors, t='mse_corr')
            # each subtraction heatmaps according to different missing rate
        each_mr_heatmaps(Ss, colors, t='sub_corr')

        ##### Plot shapley values #######
        wlda_shapley_beewarm(shapvalues, swarm_colors)
        wlda_feature_importance(shapvalues, class_colors, column_names)

    
    else:
        print('Input missing_test 0 or 1 (0 - missing values exist in only training data \n and 1 - missing values exist in both training data and test data ):')
        missing_id = input()
        missing_id = int(missing_id)
        runtime=10
        X_iris, y_iris, _ = load_irisData()
        iris_res = run_performance_experiment(X_iris, y_iris, missing_range, runtime, missing_id)
        show_accuracy_time('Iris', iris_res, missing_id)

        X_user, y_user = load_userData()
        user_res = run_performance_experiment(X_user, y_user, missing_range, runtime, missing_id)
        show_accuracy_time('User', user_res, missing_id)

        X_thyroid, y_thyroid = load_thyroidData()
        thyroid_res = run_performance_experiment(X_thyroid, y_thyroid, missing_range, runtime, missing_id)
        show_accuracy_time('Thyroid', thyroid_res, missing_id)

    
    

