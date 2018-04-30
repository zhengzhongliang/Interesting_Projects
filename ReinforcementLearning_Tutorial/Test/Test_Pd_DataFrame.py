import numpy as np
import pandas as pd

#states = 6
#actions = ['left','right']

#table = pd.DataFrame(
#        np.zeros((states, len(actions))),     # q_table initial values
#        columns=actions,    # actions's name
#    )

#states = total number of states of the world
#actions = the numebr of available actions on each state

np.random.seed(2)

for i in np.arange(30):
    print(np.round(np.random.uniform()))

#print(table)

#A=np.array([1,2,3,4,5])
#B=np.array([0,1,2,3,4])
#C=np.array([0,0,0,0,0])

#count = np.count_nonzero(C)
#print(count)

#print(np.all(C, out=np.zeros(5)))
#np.all: returns true if non elements equal to 0
