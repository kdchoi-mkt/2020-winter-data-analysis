"""간단한 statistic 등을 간편하게 나타낼 수 있게 모듈화한 파일입니다.

MadeBy: `kdchoi-mkt@kaist.ac.kr`
GitHub: `github.com/kdchoi-mkt`
"""

# Data Frame Module
import pandas as pd

# Statistic Module
from scipy.stats import ttest_ind

def indicate_ttest(data_frame: pd.DataFrame, 
                   treat_indicator: str, 
                   dep_var: str):
    """T-test를 시행합니다.

    Parameter
    =========
    `treat_indicator`: treatment 변수 (이분 변수여야함)

    `dep_var`: T-test를 시행할 변수"""

    treat_gp = data_frame.groupby([treat_indicator])[dep_var].mean()
    untreated = data_frame[data_frame[treat_indicator] == 0][dep_var]
    treated = data_frame[data_frame[treat_indicator] == 1][dep_var]

    stat, pval = ttest_ind(untreated, treated, equal_var = False)
    
    print(f"T-test 결과, {dep_var.replace('_', ' ')}에 대한 t-statistic은 {stat:.3f}이며, 유의 정도는 {pval:.3f} 입니다.")
    print(f"{treat_indicator.replace('_', ' ')}가 0일 때 : {treat_gp[0]:.3f}")
    print(f"{treat_indicator.replace('_', ' ')}가 1일 때 : {treat_gp[1]:.3f}")
