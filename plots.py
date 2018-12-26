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
def zh1(v,l):
    return Z(v,l)*H1(v)
def zh2(v,l):
    return Z(v,l)*H2(v)
def zh3(v,l):
    return Z(v,l)*H3(v)
def zh4(v,l):
    return Z(v,l)*H4(v)
def zh5(v,l):
    return Z(v,l)*H5(v)
def zh6(v,l):
    return Z(v,l)*H6(v)
def Z(v, l):
    return np.exp(-H1(v) * l[0] - H2(v) * l[1] - H3(v) * l[2] - H4(v) * l[3] - H5(v) * l[4] - H6(v) * l[5])

address = "myfile.txt"
address = 'modified.txt'
x = np.loadtxt(address,skiprows=1, unpack=True);
P = len(x[0])-1
vv =  np.linspace(-10.0, 10.0, num=1000)

fig, ax = plt.subplots();
[plt.plot(vv,Z(vv,x[:,p][6:12]),'-') for p in range(P)]
ax.set_ylabel("Z")
#plt.ylim(0.0,2.0)
plt.show()

fig, ax = plt.subplots();
[plt.plot(vv,zh1(vv,x[:,p][6:12]),'-') for p in range(P)]
ax.set_ylabel("v*Z")
plt.show()

fig, ax = plt.subplots();
[plt.plot(vv,zh2(vv,x[:,p][6:12]),'-') for p in range(P)]
ax.set_ylabel("v^2*Z")
plt.show()

fig, ax = plt.subplots();
[plt.plot(vv,zh3(vv,x[:,p][6:12]),'-') for p in range(P)]
ax.set_ylabel("v^3*Z")
plt.show()

fig, ax = plt.subplots();
[plt.plot(vv,zh4(vv,x[:,p][6:12]),'-') for p in range(P)]
ax.set_ylabel("v^4*Z")
plt.show()

fig, ax = plt.subplots();
[plt.plot(vv,zh5(vv,x[:,p][6:12]),'-') for p in range(P)]
ax.set_ylabel("v^5*Z")
plt.show()

fig, ax = plt.subplots();
[plt.plot(vv,zh6(vv,x[:,p][6:12]),'-') for p in range(P)]
ax.set_ylabel("v^6*Z")
plt.show()
print("yo")