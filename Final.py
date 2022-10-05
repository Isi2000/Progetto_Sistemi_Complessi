
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg
from numpy import *

#here I make AM

class Matrix:
    def __init__(self,n):
        self.n = n
        self.ROW = np.array([i for i in range(self.n)], dtype= double)
        self.data = [self.ROW for a in range(self.n)]

    def a_matrix(self):
        #This gives me the coordinates to access the ROWS
        r_coordinates = [r for r in self.ROW]
        #This funciton gives me all the possible combinations of indices
        #r_ is special
        def comb(a, b):
            c = []
            for i in a:
                for j in b:
                    c.append(r_[i, j])
            return c

        z = (comb(comb(r_coordinates, r_coordinates), comb(r_coordinates, r_coordinates)))

        B = np.copy(z)
        A = []
        # I am pretty sure there is a way to do this with np, but this is efficient and it works
        C = np.empty(self.n ** 4)
        for i, element in enumerate(z):
            if ((element[0] == element[2]) and ((element[3] - element[1] == 1) or (element[3] - element[1] == -1)) or (
                    element[1] == element[3]) and ((element[2] - element[0] == 1) or (element[2] - element[0] == -1))):
                # kinda bad but it works and it is not that bad
                B[i] = [1, 1, 1, 1]
            else:
                B[i] = [0, 0, 0, 0]
            A.append(B[i][0])

        C = np.copy(A)
        C_ = C.reshape(self.n ** 2, self.n ** 2)
        # to get immediatly adj_matrix:
        # print(C_)
        return C_

m = Matrix(41)

k = []
Z = linalg.matrix_power(m.a_matrix(), 30)
#plt.matshow(Z)
#plt.show()
f = 0.
for i in range(int(m.n**2)):
    f= f + (Z[int(m.n**2/2)][i])
p=[]
for i in range(int(m.n**2)):
        p.append(i)
        p.append(Z[int(m.n**2/2)][i]*(100000/f))
q = np.reshape(p, (int(m.n**2), 2))
np.savetxt('adjacent_matrix.txt', X= q, fmt = '%1.9f', newline = '\n')

#here I simulate the random walk

class Particle:
    def __init__(self):
        #these are arrays
        self.location = [0,0]
        self.velocity = [1,1]
    def rd_direction(self):
        b = np.random.choice([-1, 1])
        return b
    def move(self):
            rd1 = lambda : random.random()
            if (rd1() < 0.5):
                self.location[0] += self.velocity[0]*self.rd_direction()
            else:
                self.location[1] += self.velocity[1]*self.rd_direction()

    def dynamics(self, steps):
        k = []
        for i in range(steps):
            self.move()
            k.append(self.location[0])
            k.append(self.location[1])

class Particles:
    def __init__(self, n):
        self.n = n
        self.particles = [Particle() for i in range(n)]

    def randw(self, steps1):
        k = []
        b = []
        a = []
        c = []
        t = []
        o = []

        for i in self.particles:
            i.dynamics(steps1)
        for i in range(len(self.particles)):
            k.append(self.particles[i].location[0])
            k.append(self.particles[i].location[1])
        j = np.reshape(k, (self.n, 2))
        z = j.tolist()
        for i in z:
            a.append(i[0])
            a.append(i[1])
            a.append(z.count(i))
        l1 = np.reshape(a, (self.n, 3))
        for i in l1:
            if i[0] == 0:
                c.append(i[1])
                c.append(i[2])
        l2 = np.reshape(c, (int(len(c) / 2), 2))
         #here I am mapping my data on the grid, key point
        grid = [[x, y, 0] for x in range(-20, 21) for y in range(-20, 21)]
        for i, j, k in l1:
            for z in range(len(grid)):
                if grid[z][0] == i and grid[z][1] == j:
                    grid[z][2] = k

        r = []
        for h in grid:
            r.append(h[0])
            r.append(h[1])
            r.append(h[2])
        l4 = np.reshape(r, (len(grid), 3))

        for a in range(len(grid)):
            b.append(a)
        for a in grid:
            c.append(a[2])
        for z in range(len(grid)):
            t.append(b[z])
            t.append(c[z])
            t.append(Z[int(len(grid)/2)][z]*(100000/f))
            o.append(b[z])
            o.append(c[z]-Z[int(len(grid)/2)][z]*(1000/f))

        l3 = np.reshape(t, (len(grid), 3))
        l5 = np.reshape(o, (len(grid), 2))
        np.savetxt('hparticles1.txt', X=l1, fmt='%1.0f', newline='\n')
        np.savetxt('hparticles2.txt', X=l2, fmt='%1.0f', newline='\n')
        np.savetxt('g.txt', X=l3, fmt='%1.0f', newline='\n')
        np.savetxt('gri.txt', X=l4, fmt='%1.0f', newline='\n')
        np.savetxt('o.txt', X=l5, fmt='%1.0f', newline='\n')

m = Particles(100000)
m.randw(30)

