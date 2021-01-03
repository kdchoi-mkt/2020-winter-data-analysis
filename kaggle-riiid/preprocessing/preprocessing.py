import pandas as pd
import numpy as np

def preprocessing(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Preprocess users' learning behavior to analyze with machine learning."""

    data_frame = derive_df_lecture_info(data_frame)
    data_frame = derive_df_question_info(data_frame)
    data_frame = derive_df_user_info(data_frame)

    return data_frame

def derive_df_user_info(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering the user information with regard to the user's question answered history.

    The feature has the following:
    1. The answered count for each Part in the TOEIC
    2. The correction count for each Part in the TOEIC
    ...
    """
    
    return data_frame