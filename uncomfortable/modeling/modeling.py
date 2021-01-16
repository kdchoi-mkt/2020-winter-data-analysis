# Model Training & Performance Metric
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, log_loss, precision_score, recall_score
from statsmodels.api import add_constant

# Progress Bar
from tqdm.notebook import tqdm

# Util Module
import pandas as pd
import datetime

def report_model_result(X, y, SKLearnModel, threshold_list = [0.5], random_state = 3141592, return_model = False, **hyperparameter):
    """Report the result for the model's overall performance. 

    Parameter
    =========
    X: Independent variables
    y: Dependent variable
    SKLearnModel: The machine learning model so that can estimate the probaility for dependent variable using `predict_proba(X)` method.
    threshold_list: The list for threshold used to classifying
    random_state: To control random-ness

    Metric
    ======
    `AUC`
    `Log Loss`
    `Accuracy`
    `Precision`
    `Recall`
    `F1 Score`
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = random_state, test_size = 0.3)
    
    # Model Fitting
    model = SKLearnModel(random_state = random_state, **hyperparameter)
    model = model.fit(X_train, y_train)
    prediction = model.predict_proba(X_test)\
                      .transpose()[-1]
    
    report_df = pd.DataFrame()
    
    for threshold in threshold_list:
        report_df = pd.concat([report_df, report_metric(prediction, y_test, threshold)])

    report_df['model'] = model.__class__
    report_df['hyperparameter'] = str(hyperparameter)
    report_df['random_state'] = random_state
    
    if return_model:
        return report_df, model
    return report_df

def report_metric(y_pred, y_true, threshold):
    """Report performance for given prediction value.
    The performance measurement is described above."""
    metric = dict()
    metric['threshold'] = threshold
    
    fpr, tpr, thres = roc_curve(y_true, y_pred)

    metric['auc'] = auc(fpr, tpr)
    metric['log_loss'] = log_loss(y_true, y_pred)
    
    metric['accuracy'] = accuracy_score(y_true, y_pred > threshold)
    metric['F1_score'] = f1_score(y_true, y_pred > threshold)
    metric['precision'] = precision_score(y_true, y_pred > threshold)
    metric['recall'] = recall_score(y_true, y_pred > threshold)
    metric['# of true'] = (y_pred > threshold).sum()
    metric['# of false'] = (y_pred <= threshold).sum()
    
    return pd.DataFrame([metric.values()], columns = metric.keys())

def GridSearchModel(SKLearnClassifier, hyperparameter_candidate, X, y, random_state = 3141592):
    """Conduct Grid Search to reveal best hyperparameter for model.
    It is developed for DACON, therefore the performance metric is `AUC`.
    
    Especially, the model must have `predict_proba()` method.
    
    Parameter
    =========
    `SKLearnClassifier`: The classification model.
    `hyperparameter_candidate`: hyperparameter candidate. It has form of
    ```
    {
        'n_features': [50, 100, 150, 200],
        'n_jobs': [2],
        'reg_lambda': [0.1, 0.2, 0.3, 0.5],
        ...
    }
    ```
    """
    tqdm.pandas(desc = 'AUC progress bar')
    
    def derive_auc(hyperparam, SKLearnClassifier, X, y, random_state):
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = random_state)
    
        model = SKLearnClassifier(random_state = random_state, **hyperparam)
        model = model.fit(X_train, y_train)
        
        prediction = model.predict_proba(X_test)\
                          .transpose()[-1]
        
        fpr, tpr, thres = roc_curve(y_test, prediction)
        
        return auc(fpr, tpr)
    
    hyperparam_df = pd.DataFrame([hyperparameter_candidate.values()], columns = hyperparameter_candidate.keys())
    derive_auc_local = lambda x: derive_auc(x, SKLearnClassifier, X, y, random_state)
    
    for col in hyperparameter_candidate.keys():
        hyperparam_df = hyperparam_df.explode(col)

    hyperparam_df = hyperparam_df.reset_index(drop = True)
    hyperparam_df['AUC'] = hyperparam_df.progress_apply(lambda x: derive_auc_local(x.to_dict()), axis = 1)
        
    return hyperparam_df

def predict_uncomfortable(test_data, model_name, classifier_model, random_state = 3141592):
    now = datetime.datetime.now().strftime('%m%d')
    X = add_constant(test_data)
    
    test_data['problem'] = classifier_model.predict_proba(X).T[1]
    test_data[['problem']].reset_index().to_csv(f'../resources/output/report_{now}_{model_name}.csv', index = False)
    
    return test_data[['problem']]