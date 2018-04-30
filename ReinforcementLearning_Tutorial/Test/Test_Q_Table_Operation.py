import numpy as np
import pandas as pd
import time


np.random.seed(2)  # reproducible


N_STATES = 6   # the length of the 1 dimensional world
ACTIONS = ['left', 'right']     # available actions


table = pd.DataFrame(
        np.random.rand(N_STATES, len(ACTIONS)),     # q_table initial values
        columns=ACTIONS,    # actions's name
    )

print('table:',table)

print('table.iloc[1,1]:',table.iloc[1,0])
#print('table.iloc[1,\'left\']:',table.iloc[1,'left'])
print('table.ix[1,\'left\']:',table.ix[1,'left'])