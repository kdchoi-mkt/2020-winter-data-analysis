import pandas as pd
import random

def refine_log_data(data_frame: pd.DataFrame,
                    random_seed: int = 3141592,
                    upper_bound: int = 10) -> tuple:
    """Refine the log data to construct cross-sectional data.

    Parameter
    ========
    `data_frame`: The content spend history log data.

    `random_seed`: Control the random seed to cut the log data.
    
    `upper_bound`: The function will cut at least `upper_bound`th log data for each user.
    """
    random.seed(random_seed)

    log_count_data = derive_random_cutoff_data(data_frame, upper_bound)
    
    log_data = pd.merge(data_frame, log_count_data, on = 'user_id')
    log_data['row_id_by_user'] = log_data.groupby(['user_id'])['row_id'].rank()

    train_log_data  = log_data[log_data['row_id_by_user'] < log_data['cutoff_position']]
    test_log_data = log_data[log_data['row_id_by_user'] == log_data['cutoff_position']]

    return train_log_data, test_log_data

def derive_random_cutoff_data(log_data: pd.DataFrame,
                              upper_bound: int) -> pd.DataFrame:
    """Return the user data with cut off information.

    If the total log count for the user is less than the upper bound, return the cut off value as the log count.

    For Example
    ===========
    For the `upper_bound = 10`, 

    | user_id | total_log_count | cutoff_position |
    |---------|-----------------|-----------------|
    |    1    |      400        |       314       |
    |    2    |       2         |        2        |
    """
    def _get_cutoff_position(row_count):
        if upper_bound <= row_count:
            return random.randrange(upper_bound, row_count + 1)
        return row_count

    log_count_data = log_data.groupby(['user_id'])\
                             .agg({'row_id': 'count'})\
                             .rename(columns = {'row_id': 'total_log_count'})

    log_count_data['cutoff_position'] = log_count_data['total_log_count'].apply(lambda x: _get_cutoff_position(x))
    log_count_data = log_count_data.drop(columns = ['total_log_count'])

    return log_count_data

