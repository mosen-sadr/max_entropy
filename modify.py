import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os
from random import shuffle

def H1(v):
    return v
def H2(v):
    return v**2
def H3(v):
    return v**3
def H4(v):
    return v**4
def H5(v):
    return v**5
def H6(v):
    return v**6
def zhi(v,l,i):
    return Z(v,l)*v**i
def Z(v, l):
    return np.exp(-H1(v) * l[0] - H2(v) * l[1] - H3(v) * l[2] - H4(v) * l[3] - H5(v) * l[4] - H6(v) * l[5])

address = "myfile.txt"
x = np.loadtxt(address,skiprows=1, unpack=True);
N = len(x[0])-1
Q = []
La = []
for i in range(0,N):
    q = x[:, i][0:6]
    l = x[:, i][6:12]
    accept = 1
    for j in range(len(Q)):
        accept = 0
        for k in range(0,6):
            #if abs(l[k]-La[j][k])/abs(l[k]) > 1e-7:
            #    accept = 1
            #    break
            if abs(q[k]-Q[j][k])/abs(q[k]) > 1e-7:
                accept = 1
                break
    ## check if it goes to inf somewhere
    vv = np.linspace(-10.0, 10.0, num=10000)
    zz = Z(vv, l)

    for k in range(0,6):
        qL = zhi(-10.0,l,k+1)
        qU = zhi( 10.0, l, k+1)
        if abs(qL)>1e-15 or abs(qU)>1e-15:
            accept = 0

    if max(zz) > 1000.0:
        accept = 0
    if accept ==1:
        Q.append(q)
        La.append(l)
ind = list(range(len(Q)))
shuffle(ind)
Q0 = [ Q[ind[i]] for i in range (len(ind))]
La0 = [ La[ind[i]] for i in range (len(ind))]
Q = Q0.copy()
La = La0.copy()
name_file = "modified.txt"
f = open(name_file, "w");
st0="#"
for i in range(0,12):
    if i==0:
        st0 += "{:<13}".format("q" + str(i + 1))
    elif i<6:
        st0 += "{:<14}".format("q" + str(i + 1))
    else:
        st0+="{:<14}".format("lamb"+str(i-6+1))
st0+= "\n"
if os.stat(name_file).st_size ==0:
    f.write(st0);
for i in range(0,len(Q)):
    st = "";
    mo = ['{:.6e}'.format(float(x)) for x in Q[i][:]]
    la = ['{:.6e}'.format(float(x)) for x in La[i][:]]
    for j in range(0,6):
        st += mo[j] + " "
    for j in range(0,6):
        st += la[j] + " "
    st += "\n"
    f.write(st);

print("look!")

print('done!');


