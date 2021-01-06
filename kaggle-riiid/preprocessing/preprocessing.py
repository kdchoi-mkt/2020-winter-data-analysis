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
    """"""
    lecture_viewed_data = log_data[log_data['content_type_id'] == 1]
    lecture_viewed_data = pd.merge(lecture_viewed_data, lectures_data, left_on = 'content_id', right_on = 'lecture_id')
    
    part_one_hot_df = lecture_viewed_data.join(pd.get_dummies(lecture_viewed_data['part'], prefix="lecture_part"))
    part_list = list(set('lecture_part_' + part_one_hot_df['part'].astype('string')))
    part_data = part_one_hot_df.groupby('user_id')[part_list].sum().reset_index()

    part_typeof_one_hot_df = lecture_viewed_data.join(pd.get_dummies(lecture_viewed_data['type_of'], prefix='lecture_type_of'))
    type_of_list = list(set('lecture_type_of_' + part_typeof_one_hot_df['type_of']))
    part_type_df = part_typeof_one_hot_df.groupby(['user_id', 'part'])[type_of_list].sum().reset_index()
    part_type_df = pd.pivot_table(part_type_df, values=type_of_list, index=['user_id'], columns=['part'], aggfunc=np.sum)
    part_type_df = part_type_df.fillna(0)
    user_df = lecture_viewed_data[['user_id']]
    for col in type_of_list :
        tmp_df = part_type_df[col]
        tmp_df.columns = col + tmp_df.columns.astype('string')
        user_df = user_df.merge(tmp_df.reset_index(), on=['user_id'])

