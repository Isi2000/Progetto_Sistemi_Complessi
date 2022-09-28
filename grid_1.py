
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from numpy import linalg
import random
from matplotlib import animation
from numpy import *
#here I try to write the adjacent matrix
class Matrix:
    def __init__(self,n):
        self.n = n
        #this might not be the best way to give n_mol since this way it's fixated
        self.ROW = [i for i in range(self.n)]
        self.data = [self.ROW for a in range(self.n)]
    def a_matrix(self):
        #This gives me the coordinates to access the ROWS
        r_coordinates = [r for r in self.ROW]
        #This funciton gives me all the possible combinations of indices
        def comb(a, b):
            c = []
            for i in a:
                for j in b:
                    c.append(r_[i, j])
            return c
        #z gives me the single cases of my matrix. I compare values on z, but i do not change it
        #B is the matrix i modify the values on
        #C isn't yet the adjacent matrix in a ordered array, ready to be used
        #C_ which is declared later is tha matrix I wanted
        z = (comb(comb(r_coordinates,r_coordinates),comb(r_coordinates,r_coordinates)))
        B = np.copy(z)
        A = []
        #I have to use np methods, since I am not very familiar I used a classical list. There is for sure a better ay to do this
        C = np.zeros(self.n**4)
        #Perhaps there is a more efficient way to do this but this is not that bad
        for i, element in enumerate(z):
            if ((element[0] == element[2]) and ((element[3] - element[1] == 1) or (element[3] - element[1] == -1)) or (element[1] == element[3]) and ((element[2] - element[0] == 1) or (element[2] - element[0] == -1)) ):
                    B[i] = [1,1,1,1]
            else:
                    B[i] = [0,0,0,0]
            A.append(B[i][0])

        C = np.copy(A)
        C_ = C.reshape(self.n**2, self.n**2)
        print(C_)
        return C_

m = Matrix(20)
Z = linalg.matrix_power(m.a_matrix(),20)

#now some data analysis


k = []
for i in range(m.n**2):
        k.append(i)
        k.append(Z[200][i])

j = np.reshape(k, (m.n**2, 2))

print(j)
print(Z)


#performance note: with n=30 it struggles a bit but it is still quite fast (under a minute) [this does not take into acount elevating to the nth power]

plt.matshow(Z)
plt.show()
np.savetxt('adjacent_matrix.txt', X= j, fmt = '%i', newline = '\n')

