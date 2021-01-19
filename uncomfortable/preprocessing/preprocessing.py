import pandas as pd
import numpy as np
from datetime import datetime
from constant import SUPPORT_STRATEGY, QUALITY_COLUMN

def preprocessing(error_data: pd.DataFrame, quality_data: pd.DataFrame) -> pd.DataFrame:
    fw_model_matcher = derive_fw_model_matcher(error_data)
    
    err_data = derive_err_related_data(error_data)
    quality_data = derive_quality_related_data(quality_data, fw_model_matcher)
    
    total_data = err_data.join(quality_data, how = 'outer')
    total_data['has_quality_data'] = ~pd.isna(total_data['count_quality_L2_norm']) * 1
    
    return total_data.fillna(0)

def derive_fw_model_matcher(error_data: pd.DataFrame) -> pd.DataFrame:
    """Derive fw-model matcher from error data.
    The fw-model matcher has the following form
    
    |  model_nm  |  fwver  |
    |------------|---------|
    |  model_0   |   10    |
    |  model_0   |  8.5.3  |
    ..."""
    return error_data[['model_nm', 'fwver']].drop_duplicates()

def derive_err_related_data(error_data: pd.DataFrame) -> pd.DataFrame:
    """Derive error related data with the error_data log.

    The error related data has the following variables:
    1. `max_err_count`: The maximum error per day
    2. `min_err_count`: The minimum error occurred per day
    3. `sum_err_count`: The total error occurred
    4. `mean_err_cuont`: The average error per day
    5. `nunique_err_count`: The unique day that error occurred
    6. `mean_err_code_nunique`: The mean of distinct code occurred for day
    7. `nunique_errcode`: The distinct error code occurred
    8. `distinct_err`: The number of different error occurred
    9. `distinct_err_per_date`: The average distinct error occurred per day
    """
    error_data['time'] = error_data['time'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d%H%M%S'))
    error_data['date'] = error_data['time'].astype('datetime64[D]')

    user_error_ind_data = derive_err_related_individual_type_data(error_data)
    user_error_total_data = derive_err_related_total_type_data(error_data)
    user_error_model_data = derive_err_related_individual_model_data(error_data)
    
    user_error_data = user_error_ind_data.join(user_error_total_data)\
                                         .join(user_error_model_data)

    return user_error_data

def derive_err_related_individual_model_data(err_data: pd.DataFrame) -> pd.DataFrame:
    """Derive error related data that requires pivot data.
    The function pivots the `model_nm` info."""
    model_err_count = _derive_pivot_table(err_data, columns = 'model_nm', value = 'errtype', method = 'count')
    model_distinct_err = _derive_pivot_table(err_data, columns = 'model_nm', value = 'errtype', method = 'nunique')
    model_distinct_code = _derive_pivot_table(err_data, columns = 'model_nm', value = 'errcode', method = 'nunique')
    model_distinct_date = _derive_pivot_table(err_data, columns = 'model_nm', value = 'date', method = 'nunique')
    
    return model_err_count.join(model_distinct_err, how = 'outer')\
                          .join(model_distinct_code, how = 'outer')\
                          .join(model_distinct_date, how = 'outer')

def derive_err_related_individual_type_data(error_data: pd.DataFrame) -> pd.DataFrame:
    """Derive error related data that requires pivot data, i.e. requires one hot encoding."""
    
    date_error_data = error_data.groupby(['user_id', 'date', 'errtype'])\
                                .agg({'errtype': 'count', 'errcode': 'nunique'})\
                                .rename(columns = {'errtype': 'err_count',
                                                  'errcode': 'err_code_nunique'})\
                                .reset_index()
    
    user_error_data = pd.DataFrame()

    ordinary_method = ['max', 'min', 'sum', 'mean']
    for method in ordinary_method:
        error_pivot_data = _derive_pivot_table(date_error_data, method = method)
        user_error_data = user_error_data.join(error_pivot_data, how = 'outer')

    user_error_data = user_error_data.join(
        _derive_pivot_table(date_error_data, value = 'date', method = 'nunique'),
        how = 'outer'
    ).join(
        _derive_pivot_table(date_error_data, value = 'err_code_nunique', method = 'mean'),
        how = 'outer'
    ).join(
        _derive_pivot_table(error_data, value = 'errcode', method = 'nunique'),
        how = 'outer'
    )

    return user_error_data

def derive_err_related_total_type_data(error_data: pd.DataFrame) -> pd.DataFrame:
    """Derive error related data so that aggregates all errors"""
    
    user_error_gp = error_data.groupby(['user_id'])

    user_error_data = pd.DataFrame(
        data = [
            user_error_gp['date'].nunique(),
            user_error_gp['errtype'].nunique(),
            error_data.groupby(['user_id', 'date'])['errtype']\
                      .nunique()\
                      .reset_index()\
                      .groupby(['user_id'])['errtype']\
                      .mean(),
            user_error_gp['errcode'].nunique(),
            error_data.groupby(['user_id', 'date'])['errcode']\
                      .nunique()\
                      .reset_index()\
                      .groupby(['user_id'])['errcode']\
                      .mean(),
            user_error_gp['model_nm'].nunique(),
            user_error_gp['fwver'].nunique(),
        ],
        index = [
            'distinct_date',
            'distinct_err_type',
            'distinct_err_type_per_date',
            'distinct_err_code',
            'distinct_err_code_per_date',
            'distinct_err_model',
            'distinct_err_fever'
        ]
    ).transpose()

    return user_error_data


def derive_quality_related_data(quality_data: pd.DataFrame, fw_model_matcher: pd.DataFrame, filling_strategy = 1) -> pd.DataFrame:
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
    mean_quality_data, var_quality_data, error_occurred_data = refine_quality_data(quality_data, filling_strategy)

    user_quality_total_data = derive_quality_related_total_data(mean_quality_data, var_quality_data, error_occurred_data, fw_model_matcher)
    user_quality_individual_data = derive_quality_related_individual_data(mean_quality_data, var_quality_data, error_occurred_data)
    
    user_quality_data = user_quality_total_data.join(user_quality_individual_data)

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
        return _quality_strategy_1(quality_data)

def derive_quality_related_total_data(mean_quality_data, var_quality_data, error_occurred_data, fw_model_matcher):
    """Derive total data."""
    # Mean Quality Data Preprocess
    mean_quality_data['quality_L2_norm'] = (mean_quality_data[QUALITY_COLUMN] ** 2).sum(axis = 1) ** 1/2
    mean_quality_data['quality_incremental'] = mean_quality_data['quality_L2_norm'] - mean_quality_data.groupby(['user_id', 'fwver'])['quality_L2_norm'].shift(1)
    mean_quality_data['average_quality'] = mean_quality_data.groupby(['fwver'])['quality_L2_norm'].transform('mean')
    mean_quality_data['is_decremental'] = (mean_quality_data['quality_incremental'] < 0) * 1
    mean_quality_data['is_lower_average'] = (mean_quality_data['quality_L2_norm'] < mean_quality_data['average_quality']) * 1
    mean_quality_data = mean_quality_data.reset_index()
    
    # Variance Quality Data Preprocess
    var_quality_data['stability_L2_norm'] = (var_quality_data[QUALITY_COLUMN] ** 2).sum(axis = 1) ** 1/2
    var_quality_data['stability_L1_norm'] = var_quality_data[QUALITY_COLUMN].sum(axis = 1)
    var_quality_data = var_quality_data.reset_index()

    # Error Quality Data Preprocess
    error_occurred_data['total_error'] = (error_occurred_data == -1).sum(axis = 1)
    error_occurred_data = error_occurred_data.reset_index()
    
    # Definr GroupBy Object to improve readability.
    user_quality_gp = mean_quality_data.groupby(['user_id'])
    user_stability_gp = var_quality_data.groupby(['user_id'])
    user_error_gp = error_occurred_data.groupby(['user_id'])

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
            user_stability_gp['stability_L2_norm'].max(),
            user_error_gp['total_error'].mean()
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
            'product_stability_L2_max',
            'product_error_mean'
        ]
    ).transpose()
    
    # This is for model pivot data
    # Before to derive model pivot data, we have to match fwver with model_nm
    mean_quality_data = pd.merge(mean_quality_data, fw_model_matcher, on = 'fwver', how = 'outer')
    var_quality_data = pd.merge(var_quality_data, fw_model_matcher, on = 'fwver', how = 'outer')
    error_occurred_data = pd.merge(error_occurred_data, fw_model_matcher, on = 'fwver', how = 'outer')

    mean_quality_data['model_nm'] = mean_quality_data['model_nm'].fillna('model_missing')
    var_quality_data['model_nm'] = var_quality_data['model_nm'].fillna('model_missing')
    error_occurred_data['model_nm'] = error_occurred_data['model_nm'].fillna('model_missing')
        
    model_quality_count_data = _derive_pivot_table(mean_quality_data, columns = 'model_nm', value = 'quality_L2_norm', method = 'count')
    model_quality_mean_data = _derive_pivot_table(mean_quality_data, columns = 'model_nm', value = 'quality_L2_norm', method = 'mean')
    model_stability_mean_data = _derive_pivot_table(var_quality_data, columns = 'model_nm', value = 'stability_L2_norm', method = 'mean')
    model_error_count_data = _derive_pivot_table(error_occurred_data, columns = 'model_nm', value = 'total_error', method = 'sum')
    model_error_mean_data = _derive_pivot_table(error_occurred_data, columns = 'model_nm', value = 'total_error', method = 'mean')
    
    model_quality_data = model_quality_count_data.join(model_quality_mean_data, how = 'outer')\
                                                 .join(model_stability_mean_data, how = 'outer')\
                                                 .join(model_error_count_data, how = 'outer')\
                                                 .join(model_error_mean_data, how = 'outer')
    
    return user_quality_data.join(model_quality_data, how = 'outer')

def derive_quality_related_individual_data(mean_quality_data, var_quality_data, error_occurred_data):
    """Derive Quality Individual Data
    In fact, we have to remove some columns so that has zero columns!
    """
    
    quality_mean = _derive_multicolumn_agg(mean_quality_data, method = 'mean')
    quality_sum = _derive_multicolumn_agg(mean_quality_data, method = 'sum')
    quality_var = _derive_multicolumn_agg(mean_quality_data, method = 'var')

#     The feature for each quality columns is not so good...
#     stability_mean = _derive_multicolumn_agg(var_quality_data, prefix = 'stability', method = 'mean')
#     stability_sum = _derive_multicolumn_agg(var_quality_data, prefix = 'stability', method = 'sum')
    
    error_count = _derive_multicolumn_agg(error_occurred_data, prefix = 'error', method = 'sum')
    error_mean = _derive_multicolumn_agg(error_occurred_data, prefix = 'error', method = 'mean')
    
    return quality_mean.join(quality_sum)\
                       .join(quality_var)\
                       .join(error_count)\
                       .join(error_mean)
    
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

def _derive_multicolumn_agg(data_frame, group_col: list = ['user_id'], target_col: list = QUALITY_COLUMN, prefix: str = "", method = 'mean'):
    """Derive multicolumn aggregated dataframe. If the variance is 0, it automatically remove it.
    
    prefix = "{prefix}_{target_col}_{method}"
    Ex) prefix: quality_1_mean, stability_quality_3_var, ...
    """
    if len(prefix) > 0:
        prefix = prefix + "_"
    
    if method == 'mean':
        objection = data_frame.groupby(group_col)[target_col].mean()
    elif method == 'sum':
        objection = data_frame.groupby(group_col)[target_col].sum()
    elif method == 'var':
        objection = data_frame.groupby(group_col)[target_col].var()
    else:
        raise ValueError(f"The method {method} does not support!")
    
    objection.columns = prefix + objection.columns + f"_{method}"
    valid_series = objection.var()
    valid_column = valid_series[valid_series != 0].index
        
    return objection[valid_column]

def _quality_strategy_1(quality_data):
    """Quality Strategy 1: Do not fill NA Value & -1 is Error log
    """
    
    mean_quality_data = quality_data.replace(-1, np.nan)\
                                    .groupby(['user_id', 'time', 'fwver'])\
                                    .mean()
    
    var_quality_data = quality_data.groupby(['user_id', 'time', 'fwver'])\
                                   .var()
    
    error_occurred_data = quality_data.groupby(['user_id', 'time', 'fwver'])\
                                      .min()
    
    return mean_quality_data, var_quality_data, error_occurred_data