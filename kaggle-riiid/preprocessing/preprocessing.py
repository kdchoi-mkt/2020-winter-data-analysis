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
    
    # 각 user_id마다 part별 몇 개의 강의를 시청했는지를 part_df에 저장합니다. 
    part_one_hot_df = lecture_viewed_data.join(pd.get_dummies(lecture_viewed_data['part'], prefix="lecture_part"))
    part_list = list(set('lecture_part_' + part_one_hot_df['part'].astype('string')))
    part_df = part_one_hot_df.groupby('user_id')[part_list].sum().reset_index()

    # 각 user_id마다 part별로 시청한 lecture의 type_of가 몇 개인지를 type_df에 저장합니다.
    part_typeof_one_hot_df = lecture_viewed_data.join(pd.get_dummies(lecture_viewed_data['type_of'], prefix='lecture_type_of'))
    type_of_list = list(set('lecture_type_of_' + part_typeof_one_hot_df['type_of']))
    part_type_df = part_typeof_one_hot_df.groupby(['user_id', 'part'])[type_of_list].sum().reset_index()
    part_type_df = pd.pivot_table(part_type_df, values=type_of_list, index=['user_id'], columns=['part']).fillna(0)
    type_df = lecture_viewed_data[['user_id']]
    for col in type_of_list :
        tmp_df = part_type_df[col]
        tmp_df.columns = col + tmp_df.columns.astype('string')
        type_df = type_df.merge(tmp_df.reset_index(), on=['user_id'])

    # 각 user_id마다 시청한 lecture의 서로 다른 tag개수를 tag_df에 저장합니다.
    tag_df = lecture_viewed_data.groupby(['user_id', 'part'])[['tag']].nunique().reset_index()
    tag_df = pd.pivot_table(tag_df, values=['tag'], index=['user_id'], columns=['part']).fillna(0)
    tag_df.columns = 'tag_' + tag_df['tag'].columns.astype('string')
    tag_df = tag_df.reset_index()

    # 각 user_id마다 시청한 lecture의 서로 다른 tag개수를 tag_df에 저장합니다.
    timestamp_df = lecture_viewed_data.groupby(['user_id', 'part'])['timestamp'].max().reset_index()
    timestamp_df = pd.pivot_table(timestamp_df, values=['timestamp'], index=['user_id'], columns=['part']).fillna(0)
    timestamp_df.columns = 'max_timestamp_' + timestamp_df['timestamp'].columns.astype('string')
    timestamp_df = timestamp_df.reset_index()

    part_and_type = pd.merge(part_df, type_df, on='user_id')
    tag_and_timestamp = pd.merge(tag_df, timestamp_df, on='user_id')

    return pd.merge(part_and_type, tag_and_timestamp, on='user_id')