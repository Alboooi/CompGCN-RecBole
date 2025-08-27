import sys
sys.path.append('.')

import pandas as pd
from recbole.quick_start import run_recbole

result = run_recbole(model='CFKG',  dataset='ml-100k', config_dict={'require_pow': False})
