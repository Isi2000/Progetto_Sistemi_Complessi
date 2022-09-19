import matplotlib.pyplot as plt
import numpy as np
import random

from matplotlib import cm


def rg():
    a = np.random.normal(0.25, 0.01)
    return a
class Grid:
    def __init__(self,n):
        self.n = n
        #this might not be the best way to give n_mol since this way it's fixated

        self.data = np.zeros(n*n, dtype = int).reshape(n, n)
        self.data1 = np.zeros(n*n, dtype = int).reshape(n, n)
        self.zero = np.zeros(n*n, dtype = int).reshape(n, n)
    #fuction which drops a n_mol into a specific [i][j] grid
    def drop(self, i, j, n_mol):
        self.data[i][j] = self.data[i][j] + n_mol


    #no returns since this does change tha value of [i][j]
    #s_evolve stands for a single evolution

#now we implement the funct with Giova's idea, pay attention to the int
    def s_evolve(self,rep):
        for i in range(rep):
            for x in range(1, self.n - 1):
                for y in range(1, self.n - 1):

                    p_up = rg()
                    self.data1[x][y+1] += round(self.data[x][y] * p_up)
                    p_right = rg()
                    self.data1[x+1][y] += round(self.data[x][y] * p_right)
                    p_down = rg()
                    self.data1[x][y-1] += round(self.data[x][y] * p_down)
                    #I can calculate the left one without a gaussian
                    p_left = rg()
                    self.data1[x - 1][y] += round(self.data[x][y]*p_left)
                    #self.data1[x-1][y]  = self.data1[x-1][y] + self.data[x][y] - round(self.data[x][y] * p_up) - round(self.data[x][y] * p_right) - round(self.data[x][y] * p_down)
            #self.data = self.data1.copy()
            #self.data1 = self.zero.copy()

            for l in range(1, self.n - 1):
                for m in range(1, self.n - 1):
                    self.data[l][m] = self.data1[l][m]
            for l in range(1, self.n - 1):
                for m in range(1, self.n - 1):
                    self.data1[l][m] = self.zero[l][m]


m = Grid(70)

m.drop(35,35,100000)

m.s_evolve(31)



def plot_surface(data):
    n = data.shape[0]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y = range(n), range(n)
    vx, vy = np.meshgrid(x, y, indexing='ij')
    # print(vx, vy)
    ax.plot_surface(vx, vy, data, rstride=3, cstride=3, linewidth=1, antialiased=True, cmap=cm.viridis)

plot_surface(m.data)
plt.show()