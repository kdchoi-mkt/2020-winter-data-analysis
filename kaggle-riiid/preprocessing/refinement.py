import pandas as pd
import random

def refine_log_data(data_frame: pd.DataFrame,
                    random_seed: int = 3141592,
                    lower_bound: int = 10) -> tuple:
    """Refine the log data to construct cross-sectional data.

    Parameter
    ========
    `data_frame`: The content spending history log data.

    `random_seed`: Control the random seed to cut the log data.
    
    `lower_bound`: The function will cut at least `lower_bound`th log data for each user.
    """
    random.seed(random_seed)

    log_data = shift_question_info(data_frame)
    log_count_data = derive_random_cutoff_data(data_frame, lower_bound)

    log_data = pd.merge(data_frame, log_count_data, on = 'user_id')

    train_log_data  = log_data[log_data['task_container_id'] < log_data['cutoff_position']]
    test_log_data = log_data[log_data['task_container_id'] == log_data['cutoff_position']]

    return train_log_data, test_log_data

def derive_random_cutoff_data(log_data: pd.DataFrame,
                              lower_bound: int) -> pd.DataFrame:
    """Return the user data with cut off information.

    If the total log count for the user is less than the lower bound, return the cut off value as the log count.
    However, the multi-question group does not calculated in the total log count.

    For Example
    ===========
    For the `lower_bound = 10`, 

    | user_id | total_log_count | cutoff_position |
    |---------|-----------------|-----------------|
    |    1    |      400        |       314       |
    |    2    |       2         |        2        |
    """
    def _get_cutoff_position(task_tuple):
        if lower_bound <= len(task_tuple):
            return random.choice(task_tuple[lower_bound - 1:])
        return task_tuple[-1]

    log_data['question_count'] = log_data.groupby(['user_id', 'task_container_id'])['row_id'].transform(func = 'count')
    log_data = log_data[(log_data['question_count'] == 1) & (log_data['content_type_id'] == 0)]
    log_data = log_data.sort_values(['user_id', 'task_container_id'])

    task_data = log_data.groupby(['user_id'])[['task_container_id']]\
                        .aggregate(lambda x: tuple(x))

    task_data['cutoff_position'] = task_data['task_container_id'].apply(lambda x: _get_cutoff_position(x))
    task_data = task_data.drop(columns = ['task_container_id'])

    return task_data

def shift_question_info(log_data: pd.DataFrame) -> pd.DataFrame:
    """Shift the `prior_...` data to match informations.
    Because the multiple question has the same `prior_...` values, I collapsed them by `user_id` and `task_container_id` to treat those variables easily.

    For example
    ===========
    |  Timestamp  |  user_id  |  prior_question_elapsed_time  |  question_elapsed_time  |
    |-------------|-----------|-------------------------------|-------------------------|
    |      0      |     1     |             NaN               |          100000         |
    |   300000    |     1     |           100000              |           50000         |
    |   650000    |     1     |            50000              |           30000         |
    |   800000    |     1     |            30000              |          100000         |
    """
    question_data = log_data[log_data['content_type_id'] == 0]
    identify_columns = ['user_id', 'task_container_id']
    prior_columns = [col for col in log_data.columns if 'prior' in col]
    
    task_data = question_data[identify_columns + prior_columns].drop_duplicates()
    task_data[['question_elapsed_time', 'question_had_explanation']] = task_data.groupby('user_id')[prior_columns].shift(-1)
    
    return pd.merge(log_data, task_data, on = ['user_id', 'task_container_id'], how = 'left')