import pandas as pd
import numpy as np

def preprocessing(data_frame: pd.DataFrame, question_data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess users' learning behavior to analyze with machine learning.
    
    Parameter
    =========
    `data_frame`: The user's history log data.
    `question_data`: The overall question informations.
    """

    data_frame = derive_df_lecture_info(data_frame)
    data_frame = derive_question_info(data_frame, question_data)
    data_frame = derive_df_user_info(data_frame)

    return data_frame

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
            user_gp['answered_correctly'].last()
        ],
        index = [
            'answered_count',
            'correct_count',
            'seen_explanation_count',
            'answer_elapsed_time_mean',
            'answer_elapsed_time_sum',
            'recently_solve_question',
            'recently_correct_answer'
        ]
    ).transpose()

    question_cross_sectional_data['correct_rate'] = question_cross_sectional_data['correct_count'] / question_cross_sectional_data['answered_count'] * 100
    question_cross_sectional_data['seen_explanation_rate'] = question_cross_sectional_data['seen_explanation_count'] / question_cross_sectional_data['answered_count'] * 100    

    question_cross_sectional_data.columns = prefix + question_cross_sectional_data.columns    
    question_cross_sectional_data = question_cross_sectional_data.astype('float')

    return question_cross_sectional_data.reset_index()