import pandas as pd
from datetime import datetime
from constant import SUPPORT_STRATEGY, QUALITY_COLUMN

def preprocessing(error_data: pd.DataFrame, quality_data: pd.DataFrame) -> pd.DataFrame:
    err_data = derive_err_related_data(error_data)
    quality_data = derive_quality_related_data(quality_data)
    
    return err_data.join(quality_data, how = 'outer')\
                   .fillna(0)

def derive_err_related_data(error_data: pd.DataFrame) -> pd.DataFrame:
    """Derive error related data with the error_data log.

    The error related data has the following variables:
    1. `max_err_count`: The maximum error per day
    2. `min_err_count`: The minimum error occurred per day
    3. `sum_err_count`: The total error occurred
    4. `mean_err_cuont`: The average error per day
    5. `nunique_err_count`: The unique day that error occurred
    6. `distinct_err`: The number of different error occurred
    7. `distinct_err_per_date`: The average distinct error occurred per day
    """
    error_data['time'] = error_data['time'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d%H%M%S'))
    error_data['date'] = error_data['time'].astype('datetime64[D]')

    date_error_data = error_data.groupby(['user_id', 'date', 'errtype'])[['errtype']]\
                                .count()\
                                .rename(columns = {'errtype': 'err_count'})\
                                .reset_index()

    user_error_ind_data = derive_err_related_individual_data(date_error_data)
    user_error_total_data = derive_err_related_total_data(date_error_data)

    user_error_data = user_error_ind_data.join(user_error_total_data)

    return user_error_data

def derive_err_related_individual_data(date_error_data: pd.DataFrame) -> pd.DataFrame:
    """Derive error related data that requires pivot data, i.e. requires one hot encoding."""
    user_error_data = pd.DataFrame()

    ordinary_method = ['max', 'min', 'sum', 'mean']
    for method in ordinary_method:
        error_pivot_data = _derive_pivot_table(date_error_data, method = method)
        user_error_data = user_error_data.join(error_pivot_data, how = 'outer')

    user_error_data = user_error_data.join(
        _derive_pivot_table(date_error_data, value = 'date', method = 'nunique'),
        how = 'outer'
    )

    return user_error_data

def derive_err_related_total_data(date_error_data: pd.DataFrame) -> pd.DataFrame:
    """Derive error related data so that aggregates all errors"""
    
    user_error_gp = date_error_data.groupby(['user_id'])

    user_error_data = pd.DataFrame(
        data = [
            user_error_gp['errtype'].nunique(),
            date_error_data.groupby(['user_id', 'date'])['errtype']\
                           .nunique()\
                           .reset_index()\
                           .groupby(['user_id'])['errtype']\
                           .mean()
        ],
        index = [
            'distinct_err',
            'distinct_err_per_date'
        ]
    ).transpose()

    return user_error_data

# Private Functions

def _derive_pivot_table(data_frame, columns = 'errtype', index = 'user_id', value = 'err_count', method = 'max'):
    """Derive pivot table.
    Because the function is usually used to derive err-related data, the initial value is set by error related.
    
    The column has prefix and it is '{method}_{value}'
    """
    pivot_table = data_frame.pivot_table(values = value, columns = columns, index = index, aggfunc = method)\
                            .fillna(0)
    pivot_table.columns = f"{method}_{value}_" + pivot_table.columns.astype(str)
    return pivot_table

def derive_quality_related_data(quality_data: pd.DataFrame, filling_strategy = 1) -> pd.DataFrame:
    """Derive quality based data.
    The quality data is consisted of their self-diagonstic report data.
    To be specific, in general, once the product conducts self-diagnostic test, it reports 12 log data for each quality features simultaneously.
    However, some data for quality_i is missing, therefore the treatment for missing data is vary.
    Strategy 1. ignore missing value
    Strategy 2. filling missing value as 0
    Strategy 3. filling missing value as previous one
    Note: In 1/12, the funcion does not support strategy 2 and 3


    The quality based data has the following variables:
    1. `product_inspect_count`
    2. `product_experience_count`
    3. `product_quality_mean`
    4. `product_quality_decrease_count`
    5. `product_quality_decrease_ratio`
    6. `product_quality_lower_count`
    7. `product_quality_increase_mean`
    """
    mean_quality_data, var_quality_data = refine_quality_data(quality_data, filling_strategy)
    user_quality_total_data = derive_quality_related_total_data(mean_quality_data, var_quality_data)

    user_quality_data = user_quality_total_data

    return user_quality_data

def refine_quality_data(quality_data, filling_strategy):
    """Preprocess quality data by following steps:    
    0. Check whether filling strategy appropriate or not
    1. Preprocess time variable
    2. Preprocess column data type
    3. Return mean value (performance) and variance value (stability).
    """
    if filling_strategy not in SUPPORT_STRATEGY:
        raise ValueError(f"The filling strategy {filling_strategy} does not support!")
    
    quality_data['time'] = quality_data['time'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d%H%M%S'))
    quality_data['fwver'] = quality_data['fwver'].fillna('missing')
    
    quality_data[QUALITY_COLUMN] = quality_data[QUALITY_COLUMN].astype('str')\
                                                               .apply(lambda row: [x.replace(',', '') for x in row])\
                                                               .astype('float')

    if filling_strategy == 1:
        return quality_data.groupby(['user_id', 'time', 'fwver']).mean(), quality_data.groupby(['user_id', 'time', 'fwver']).var()

def derive_quality_related_total_data(mean_quality_data, var_quality_data):
    """Derive total data."""
    mean_quality_data = mean_quality_data + mean_quality_data.min().min() * -1
    mean_quality_data['quality_L2_norm'] = (mean_quality_data[QUALITY_COLUMN] ** 2).sum(axis = 1) ** 1/2
    mean_quality_data['quality_incremental'] = mean_quality_data['quality_L2_norm'] - mean_quality_data.groupby(['user_id', 'fwver'])['quality_L2_norm'].shift(1)
    mean_quality_data['average_quality'] = mean_quality_data.groupby(['fwver'])['quality_L2_norm'].transform('mean')
    mean_quality_data['is_decremental'] = (mean_quality_data['quality_incremental'] < 0) * 1
    mean_quality_data['is_lower_average'] = (mean_quality_data['quality_L2_norm'] < mean_quality_data['average_quality']) * 1
    mean_quality_data = mean_quality_data.reset_index()

    var_quality_data['stability_L2_norm'] = (var_quality_data[QUALITY_COLUMN] ** 2).sum(axis = 1) ** 1/2
    var_quality_data['stability_L1_norm'] = var_quality_data[QUALITY_COLUMN].sum(axis = 1)

    user_quality_gp = mean_quality_data.groupby(['user_id'])
    user_stability_gp = var_quality_data.groupby(['user_id'])

    user_quality_data = pd.DataFrame(
        data = [
            user_quality_gp['fwver'].count(),
            user_quality_gp['fwver'].nunique(),
            user_quality_gp['quality_L2_norm'].mean(),
            user_quality_gp['is_decremental'].sum(),
            user_quality_gp['is_decremental'].mean(),
            user_quality_gp['is_lower_average'].sum(),
            user_quality_gp['quality_incremental'].mean(),
            user_stability_gp['stability_L1_norm'].mean(),
            user_stability_gp['stability_L2_norm'].mean(),
            user_stability_gp['stability_L2_norm'].max()
        ],
        index = [
            'product_inspect_count',
            'product_experience_count',
            'product_quality_mean',
            'product_quality_decrease_count',
            'product_quality_decrease_ratio',
            'product_quality_lower_count',
            'product_quality_increase_mean',
            'product_stability_L1_mean',
            'product_stability_L2_mean',
            'product_stability_L2_max'
        ]
    ).transpose()

    return user_quality_data