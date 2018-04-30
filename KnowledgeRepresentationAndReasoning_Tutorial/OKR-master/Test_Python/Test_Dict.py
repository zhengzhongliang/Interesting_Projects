import numpy as np

A = {0:'Hello', 1:'Wrold', 2:'!'}

for i in np.arange(3):
    print(A[i])


A= np.arange(50)
print('generate numpy array:',A)
print('square:', np.power(A,2))
