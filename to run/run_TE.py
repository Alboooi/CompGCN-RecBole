import sys
sys.path.append('.')

import pandas as pd
from recbole.quick_start import run_recbole

result = run_recbole(model='TransERecBole', config_file_list=['config_TE.yaml'])

df = pd.DataFrame(result)
df.to_csv('resuts_TE.csv', index=False)