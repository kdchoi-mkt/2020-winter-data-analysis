import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, log_loss, precision_score, recall_score

def report_model_result(X, y, SKLearnModel, threshold_list = [0.5], random_state = 3141592, **hyperparameter):
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = random_state)
    
    # Model Fitting
    model = SKLearnModel(random_state = random_state, **hyperparameter)
    model = model.fit(X_train, y_train)
    prediction = model.predict_proba(X_test)\
                      .transpose()[1]
    
    report_df = pd.DataFrame()
    
    for threshold in threshold_list:
        report_df = pd.concat([report_df, report_metric(prediction, y_test, threshold)])

    report_df['model'] = model.__class__
    report_df['hyperparameter'] = str(hyperparameter)
    report_df['random_state'] = random_state
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