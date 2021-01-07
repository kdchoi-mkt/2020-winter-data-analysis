import pandas as pd
import numpy as np

def preprocessing(log_data: pd.DataFrame, 
                  objection_data: pd.DataFrame,
                  lecture_meta_data: pd.DataFrame, 
                  question_meta_data: pd.DataFrame,
                  question_overall_data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess users' learning behavior to analyze with machine learning.
    
    Parameter
    =========
    `log_data`: The user's history log data.
    `lecture_meta_data`: The overall lecture informations.
    `question_meta_data`: The overall question informations.
    `question_overall_data`: The overall question informations about the questions' correction rate.
    `objection_data`: The objective data that will be used to train or test the Machine Learning performance.
    """

    lecture_data = derive_lecture_info(log_data, lecture_meta_data)
    question_data = derive_question_info(log_data, question_meta_data)
    objection_data = derive_user_info(lecture_data, answer_data, objection_data, question_meta_data, question_overall_data)

    return objection_data

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

def derive_question_info(log_data: pd.DataFrame, question_data: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering the user information with regard to the user's question answered history.

    The feature has the following:
    1. `answered_count`          : The answered count for each Part in the TOEIC
    2. `correct_count`           : The correction count for each Part in the TOEIC
    3. `correct_rate`            : The correction rate for each Part in the TOEIC
    4. `answer_elapsed_time_mean`: The average of elapsed time to solving question for each Part in the TOEIC
    5. `answer_elapsed_time_sum` : The total elapsed time to solving question for each Part in the TOEIC
    6. `seen_explanation_count`  : Seen explanation count for each Part in the TOEIC
    7. `seen_explanation_rate`   : Seen explanation rate for each Part in the TOEIC
    8. `recently_solve_question` : The timestamp of recently solve question for each Part in the TOEIC
    9. `recently_correct_answer` : The indicator wheter the user corrected the answer in the previous question
    10. `solved_question_tag_list`: The list of the tags that solved by user

    In fact, the function also have the total information for each variable described above.

    Parameter
    =========
    `log_data`: The user log history. It must have the following columns:
        + `user_id`
        + `answered_correctly`
        + `timestamp`
        + `question_had_explanation`
        + `question_elapsed_time`

        If you want to derive the last two variables, execute `refine_log_data()` in the `refinement` module.
    
    `question_data`: The question information data frame. You can download the `question.csv` in the Kaggle Competition Site.
    
    Result
    ======
    | user_id |  part | correct_count | ... | total_correct_count |
    |---------|-------|---------------|-----|---------------------|
    |    1    |   1   |       3       | ... |         10          |
    |    1    |   2   |       5       | ... |         10          |
    |    1    |   3   |       2       | ... |         10          |
    |    2    |   2   |       1       | ... |          5          |
    |    2    |   4   |       4       | ... |          5          |
    |   ...   |  ...  |      ...      | ... |         ...         |
    """
    question_log_data = log_data[log_data['content_type_id'] == 0]
    question_log_data = pd.merge(question_log_data, question_data, left_on = 'content_id', right_on = 'question_id')
    question_log_data['tag_list'] = question_log_data['tags'].apply(lambda x: str(x).split(' '))

    # Sort by `row_id` to use `groupby.last()` method.
    # In fact, sort by `row_id` is same as sort by both `user_id` and `timestamp`
    question_log_data = question_log_data.sort_values(['row_id'])

    user_part_gp = question_log_data.groupby(['user_id', 'part'])
    user_gp = question_log_data.groupby(['user_id'])
    
    part_answer_data = _derive_question_cross_sectional_data(user_part_gp)
    total_answer_data = _derive_question_cross_sectional_data(user_gp, prefix = 'total')

    answer_data = pd.merge(part_answer_data, total_answer_data, on = ['user_id'])

    return answer_data

def _derive_question_cross_sectional_data(user_gp: pd.core.groupby.generic.DataFrameGroupBy, prefix: str = "") -> pd.DataFrame:
    """Derive question cross sectional data.
    The information is subordinated to the `derive_df_question_info()` function."""

    if len(prefix) > 0:
        prefix += "_"

    question_cross_sectional_data = pd.DataFrame(
        data = [
            user_gp['answered_correctly'].count(),
            user_gp['answered_correctly'].sum(),
            user_gp['question_had_explanation'].sum(),
            user_gp['question_elapsed_time'].mean(),
            user_gp['question_elapsed_time'].sum(),
            user_gp['timestamp'].max(),
            user_gp['answered_correctly'].last(),
            user_gp['tag_list'].sum()
        ],
        index = [
            'answered_count',
            'correct_count',
            'seen_explanation_count',
            'answer_elapsed_time_mean',
            'answer_elapsed_time_sum',
            'recently_solve_question',
            'recently_correct_answer',
            'solved_question_tag_list'
        ]
    ).transpose()

    question_cross_sectional_data['correct_rate'] = question_cross_sectional_data['correct_count'] / question_cross_sectional_data['answered_count'] * 100
    question_cross_sectional_data['seen_explanation_rate'] = question_cross_sectional_data['seen_explanation_count'] / question_cross_sectional_data['answered_count'] * 100    

    question_cross_sectional_data.columns = prefix + question_cross_sectional_data.columns
    
    non_list_columns = [col for col in question_cross_sectional_data if 'list' not in col]
    question_cross_sectional_data[non_list_columns] = question_cross_sectional_data[non_list_columns].astype('float')

    return question_cross_sectional_data.reset_index()

def derive_user_info(lecture_data, answer_data, objection_data, question_meta_data, question_overall_data):
    """Derive objection data to train machine.
    코드 가용성은 최대한 줄였으며, 코드 복잡도는 최대한으로 올렸습니다."""
    question_pivot_col = 'part'
    question_pivot_index = 'user_id'
    question_pivot_value_list = [col for col in answer_data.columns if ('total' not in col) and (col not in ['user_id', 'part'])]
    total_value_list = [col for col in answer_data.columns if 'total' in col]

    question_pivot_df = pd.DataFrame()
    for col in question_pivot_value_list:
        pivot_table = answer_data.pivot(values = col, index = question_pivot_index, columns = question_pivot_col)
        pivot_table.columns = f"{col}_" + pivot_table.columns.astype(str)
        question_pivot_df = question_pivot_df.join(pivot_table, how = 'outer')

    answer_data = answer_data[total_value_list + ['user_id']].drop_duplicates(['user_id'])\
                                                                 .set_index('user_id')\
                                                                 .join(question_pivot_df)

    total_feature_data = answer_data.join(lecture_data)
    objection_data = objection_data.set_index('user_id')
    objection_data = objection_data.join(total_feature_data)
    objection_data = objection_data.merge(question_meta_data, left_on = 'content_id', right_on = 'question_id')
    objection_data['tag_list'] = objection_data['tags'].apply(lambda x: str(x).split(' '))
    tag_list = [col for col in objection_data.columns if 'list' in col and col != 'tag_list']
    def overlap(row, parse):
        try:
            if parse == 'total':
                compare_set = set(row['total_solved_question_tag_list'])
            else:
                compare_set = set(row[f'solved_question_tag_list_{parse}'])
            question_tag = set(row['tag_list'])

            return len(question_tag.intersection(compare_set)) > 0 
        except:
            return False

    for col in tag_list:
        if 'total' in col:
            parse = 'total'    
        else:
            parse = col[-1]
        objection_data[f'is_solved_tag_on_{parse}'] = objection_data.apply(lambda x: overlap(x, parse), axis = 1)

    question_overall_data = question_overall_data.set_index('content_id')
    question_overall_data.columns = 'question_' + question_overall_data.columns
    objection_data = objection_data.merge(question_overall_data, on = 'content_id')

    objection_data = objection_data.drop(
        columns = ['Unnamed: 0', 'row_id', 'question_id', 'task_container_id', 'user_answer', 
        'prior_question_elapsed_time', 'prior_question_had_explanation', 'question_elapsed_time', 
        'question_had_explanation', 'cutoff_position', 
        'total_solved_question_tag_list',
        'solved_question_tag_list_1',
        'solved_question_tag_list_2',
        'solved_question_tag_list_3',
        'solved_question_tag_list_4',
        'solved_question_tag_list_5',
        'solved_question_tag_list_6',
        'solved_question_tag_list_7',
        'question_id',
        'bundle_id',
        'correct_answer',
        'part',
        'tags',
        'tag_list', 'content_id', 'content_type_id'])



    return objection_data