LGBM_HYPERPARAM_CANDIDATE = {
    'boosting_type': ['gbdt', 'dart', 'goss'],
    'num_leaves': [20, 30, 40, 50],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'n_estimators': [50, 100, 150, 200],
    'reg_alpha': [0, 0.1, 0.2, 0.3, 0.5],
    'reg_lambda': [0, 0.1, 0.2, 0.3, 0.5],
    'n_jobs': [-1]
}

GBC_HYPERPARAM_CANDIDATE = {
    'loss': ['deviance', 'exponential'],
    'criterion': ['friedman_mse', 'mse', 'mae'],
    'max_features': ['auto', 'sqrt', 'log2'],
    'learning_rate': [.01, .05, .1, .2, .3],
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [10, 20, 30, 40, 50],
    'min_samples_split': [10, 20, 30, 40, 50],
    'min_samples_leaf': [10, 20, 30, 40, 50],
    'n_jobs': [-1],

}

RF_HYPERPARAM_CANDIDATE = {
    'criterion': ['gini', 'entropy'],
    'max_features': ['auto', 'sqrt', 'log2'],
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [10, 20, 30, 40, 50],
    'min_samples_split': [10, 20, 30, 40, 50],
    'min_samples_leaf': [10, 20, 30, 40, 50],
    'n_jobs': [-1],
}


FINAL_LGBM_PARAMETER = {
    'boosting_type': 'gbdt',
    'num_leaves': 50,
    'learning_rate': 0.03,
    'n_estimators': 200,
    'reg_alpha': 0.2,
    'reg_lambda': 0.3,
    'n_jobs': 2
}
