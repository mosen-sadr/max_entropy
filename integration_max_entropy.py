from scipy import integrate
import numpy as np
import sympy as sym

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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

def Z(v, l, mean, sig, a):
    # return np.exp(-H1(v)*l[0] - H2(v)*l[1] - H3(v)*l[2] - H4(v)*l[3] - H5(v)*l[4])
    return np.exp(-H1(v) * l[0] - H2(v) * l[1] - H3(v) * l[2] - H4(v) * l[3] - H5(v) * l[4] - H6(v) * l[5])#+a/(np.sqrt(2.0*np.pi)*sig)*np.exp(-(v-mean)**2.0/2.0/sig**2)
    #return 1.0/(np.sqrt(2.0*np.pi)*2)*np.exp(-(v-3)**2.0/2.0/2**2)-a/(np.sqrt(2.0*np.pi)*sig)*np.exp(-(v-mean)**2.0/2.0/sig**2)
#1.0/varr**0.5*( Z(vv, ll) - mean )

def Z2(v, l):
    return np.exp(-H1(v) * l[0] - H2(v) * l[1] - H3(v) * l[2] - H4(v) * l[3] - H5(v) * l[4] - H6(v) * l[5])

def zh1(v,l, mean, sig, a):
    return Z(v,l, mean, sig, a)*H1(v)
def zh2(v,l, mean, sig, a):
    return Z(v,l, mean, sig, a)*H2(v)
def zh3(v,l, mean, sig, a):
    return Z(v,l, mean, sig, a)*H3(v)
def zh4(v,l, mean, sig, a):
    return Z(v,l, mean, sig, a)*H4(v)
def zh5(v,l, mean, sig, a):
    return Z(v,l, mean, sig, a)*H5(v)
def zh6(v,l, mean, sig, a):
    return Z(v,l, mean, sig, a)*H6(v)

def zh11(v,l, mean, sig, a, mu):
    return Z(v-mu,l, mean, sig, a)*H1(v)
def zhvar(v, l, mean, sig, a, m):
    return Z(v, l, mean, sig, a) * (H2(v-m))

def moments(l,a,b):
    #Q = [[],[],[],[],[]];
    mean = 0.0; varr = 1.0;
    Q = np.zeros((6, 2))
    #  l = [1.0e1, 1.0e1, 1.0e0, 1.0e-1, -1e-3, 1e-6];
    intt = integrate.quad(Z, a, b, args=( l, 0.0, 1.0, 0.0) );
    #  intt2 =  integrate.fixed_quad(Z, a, b, args=( l, 0.0, 1.0, 0.0) ,n=10);
    #  intt3 = integrate.quad(Z2, -10.0, 10.0, l);
    #  zz = Z(np.linspace(a, b, 20), l, 0.0, 1.0, 0.0)
    err = 1;
    if intt[0] is not None and np.isinf(intt[0]) == 0 and intt[0]>1e-15 and intt[0]<1e16:
        Q[0] = integrate.quad(zh1, a, b, args=( l, 0.0, 1.0, 0.0) );
        Q[0,0] = Q[0,0]/intt[0];
        if Q[0,0]<1e16:
            Q[1] = integrate.quad(zh2, a, b, args=( l, 0.0, 1.0, 0.0) );
            Q[1, 0] = Q[1, 0] / intt[0];
            if Q[1,0] < 1e16:
                Q[2] = integrate.quad(zh3, a, b, args=( l, 0.0, 1.0, 0.0) );
                Q[2, 0] = Q[2, 0] / intt[0];
                if Q[2,0] < 1e16:
                    Q[3] = integrate.quad(zh4, a, b, args=( l, 0.0, 1.0, 0.0) );
                    Q[3, 0] = Q[3, 0] / intt[0];
                    if Q[3,0] < 1e16:
                        Q[4] = integrate.quad(zh5, a, b, args=( l, 0.0, 1.0, 0.0) );
                        Q[4, 0] = Q[4, 0] / intt[0];
                        if Q[4,0] < 1e16:
                            Q[5] = integrate.quad(zh6, a, b, args=(l, 0.0, 1.0, 0.0));
                            Q[5, 0] = Q[5, 0] / intt[0];
                            if Q[5, 0] < 1e16:
                                err = 0;
    return Q, intt, err

'''
ll = [0.5,0.1,-0.01,0.1,-0.0001]

min = -60.0; max = 60.0;
#test1 = integrate.quad(Z, -1.0, 1.0, args=( ll, 0.0, 1.0))
#test1 = integrate.quad(Z, -1.0, 1.0, args=( ll, test1[0], 1.0))
m1 = integrate.quad(zh1, min, max, args=( ll,0.0, 1.0, 0.0))
m11 = integrate.quad(zh11, min, max, args=( ll, 0.0, 1.0, 0.0, m1[0]))
v1 = integrate.quad(zhvar, min, max, args=( ll,m1[0], 1.0, 0.0, m1[0]))
#sig = np.sqrt( vv[0] );
mean = -m1[0]
sig = np.sqrt( 1-v1[0] )
m2 = integrate.quad(zh1, min, max, args=( ll, mean, sig, 1.0))
v2 = integrate.quad(zhvar, min, max, args=( ll, mean, sig, 1.0, m2[0]))
#test3 = integrate.quad(zh2, -1.0, 1.0, args=( ll, .0, 1.0))
#testvarr = integrate.quad(zh2, vmin, vmax, ll, mean, varr)

N = 100;
x = (max-min)*np.random.rand(N,1)+min;
y1 = Z(x, ll, mean, sig, 0.0)
y2 = Z(x, ll, mean, sig, 1.0)
fig, ax = plt.subplots();
plt.plot(np.array(x),y1,'r.',label='y1')
plt.plot(np.array(x),y2,'b.',label='y2')
plt.legend()
plt.show()
varr
'''