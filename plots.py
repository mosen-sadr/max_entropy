import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

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

def Z(v, l):
    return np.exp(-H1(v) * l[0] - H2(v) * l[1] - H3(v) * l[2] - H4(v) * l[3] - H5(v) * l[4] - H6(v) * l[5])

address = "myfile.txt"
x = np.loadtxt(address,skiprows=1, unpack=True);
P = len(x[0])-1
vv =  np.linspace(-10.0, 10.0, num=1000)

fig, ax = plt.subplots();
[plt.plot(vv,Z(vv,x[:,p][6:12]),'-') for p in range(P)]
#ax.set_ylabel("Lambda"+str(p+1)+" of induced points")
#ax.set_yscale("log")
#plt.legend()
plt.ylim(0.0,5.0)
plt.show()

print("yo")