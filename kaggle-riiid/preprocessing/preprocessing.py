import pandas as pd
import numpy as np

def preprocessing(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Preprocess users' learning behavior to analyze with machine learning.
    
    The input data is form of log data (Panel Data), but the output data will be the form of cross-sectional data.
    """

    data_frame = derive_df_lecture_info(data_frame)
    data_frame = derive_df_question_info(data_frame)
    data_frame = derive_df_user_info(data_frame)

    return data_frame