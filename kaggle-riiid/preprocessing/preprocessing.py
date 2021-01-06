import pandas as pd
import numpy as np

def preprocessing(data_frame: pd.DataFrame, lecture_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess users' learning behavior to analyze with machine learning."""

    data_frame = derive_lecture_info(data_frame, lecture_df)
    data_frame = derive_df_question_info(data_frame)
    data_frame = derive_df_user_info(data_frame)

    return data_frame

def derive_lecture_info(log_data: pd.DataFrame,
                           lectures_data: pd.DataFrame) -> pd.DataFrame :
    """
    각 user_id마다 part별 몇 개의 강의를 시청했는지를 part_df에 저장합니다. 
    각 user_id마다 part별로 시청한 lecture의 type_of가 몇 개인지를 type_df에 저장합니다.
    각 user_id마다 시청한 lecture의 서로 다른 tag개수를 tag_df에 저장합니다.
    각 user_id마다 시청한 lecture의 timestamp df의 max 값을 timestamp_df에 저장합니다.
    
    """
    lecture_viewed_data = log_data[log_data['content_type_id'] == 1]
    lecture_viewed_data = pd.merge(lecture_viewed_data, lectures_data, left_on = 'content_id', right_on = 'lecture_id')
    
    lecture_viewed_data = lecture_viewed_data.join(pd.get_dummies(lecture_viewed_data['type_of'], prefix='lecture_type_of'))
    
    part_data = _derive_part_data(lecture_viewed_data)
    total_data = _derive_total_data(lecture_viewed_data)

    return part_data.join(total_data).reset_index()

def _derive_part_data(lecture_viewed_data: pd.DataFrame) -> pd.DataFrame:
    """Derive the part-wise individual feature descrived in the `derive_lecture_info()` docstring."""    
    type_of_list = list(set('lecture_type_of_' + lecture_viewed_data['type_of']))

    tag_df = _derive_pivot_data(lecture_viewed_data, value_col = ['tag'], aggfunc = 'nunique')
    type_df = _derive_pivot_data(lecture_viewed_data, value_col = type_of_list, aggfunc = 'count')
    part_df = _derive_pivot_data(lecture_viewed_data, value_col = ['content_id'], aggfunc = 'count')
    timestamp_df = _derive_pivot_data(lecture_viewed_data, value_col = ['timestamp'], aggfunc = 'max')

    return part_df.join(type_df)\
                  .join(tag_df)\
                  .join(timestamp_df)

def _derive_total_data(lecture_viewed_data: pd.DataFrame) -> pd.DataFrame:
    """Derive the total individual feature data described in the `derive_lecture_info()` docstring"""
    type_of_list = list(set('lecture_type_of_' + lecture_viewed_data['type_of']))
    user_gp = lecture_viewed_data.groupby(['user_id'])

    total_df = pd.DataFrame(
        data = [
            user_gp['tag'].nunique(),
            user_gp['content_id'].count(),
            user_gp['timestamp'].max()
        ],
        index = [
            'total_tag_nunique',
            'total_content_id_count',
            'total_timestamp_max'
        ]
    ).transpose()

    type_of_df = user_gp[type_of_list].sum()
    type_of_df.columns += '_count'

    return total_df.join(type_of_df)

def _derive_pivot_data(data_frame: pd.DataFrame, value_col: list, aggfunc: str, index: str = 'user_id', column: str = 'part') -> pd.DataFrame:
    """Derive pivot table for the data frame with regard to the index, column, values.
    
    In general, the function is used to derive pivot data of `user_id` and `part`, therefore the function initializes the index and column as user_id and part respectively."""
    pivot_data = data_frame.pivot_table(values = value_col, index=[index], columns=[column], aggfunc = aggfunc).fillna(0)
    total_data = pd.DataFrame()

    for col in value_col:
        objective_data = pivot_data[col]
        objective_data.columns = f"{col}_{aggfunc}_{column}_" + objective_data.columns.astype(str)
        total_data = total_data.join(objective_data, how = 'outer')

    return total_data