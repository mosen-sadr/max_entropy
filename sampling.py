'''
This file produces N samples from maximum entropy distribution
with q1=0.0 and q2=1.0
'''
import numpy as np
from scipy import integrate

def Z(v, l):
    return np.exp(-H1(v) * l[0] - H2(v) * l[1] - H3(v) * l[2] - H4(v) * l[3] - H5(v) * l[4] - H6(v) * l[5])
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

def samples(N):
    la_max = np.array([1e1, 1e1, 1e0, 1e-1, -1e-5, 1e-5])
    la_min = np.array([-1e1, -1e1, -1e0, -1e-1, -1e-3, 1e-6])
    La = [];
    Mo = [];
    maxIter = 1000;
    count = 0;
    while count < N:
        intt = [1e17, 1e17];
        while abs(intt[0]) > 1e10:
            l = (la_max - la_min) * np.random.rand(1, 6) + la_min;
            l = l[0];
            intt = integrate.quad(Z, -1e1, 1e1, args=(l));

        q1 = integrate.quad(zh1, -1e1, 1e1, args=(l));
        q2 = integrate.quad(zh2, -1e1, 1e1, args=(l));
        #print("q1 = " + str(q1[0]) + " and q2 = " + str(q2[0]))
        for i in range(0, maxIter):
            if q1[0] > 1e-16:
                sg =  -1.0
            elif q1[0] < -1e-16:
                sg =  -1.0;
            else:
                sg = 0.0;
            s = 0.001;
            rate = [s, s / 3.0, s / 15.0]
            l[0] = l[0] + sg * rate[0] * l[0]
            l[2] = l[2] + sg * rate[1] * l[2]
            l[4] = l[4] + sg * rate[2] * l[4]


            if q2[0] - 1.0 > 1e-6:
                sg = 1.0
            elif q2[0] - 1.0 < -1e-6:
                sg = -1.0
            else:
                sg = 0.0;
            s = 0.005
            rate = [s / 3.0, s / 15.0, s / 105.0]
            l[1] = l[1] + sg * rate[0] * l[1]
            l[3] = l[3] + sg * rate[1] * l[3]
            l[5] = l[5] + sg * rate[2] * l[5]

            for j in range(0,6):
                if abs(l[j])<1e-16:
                    l[j] = 1e-16

            q1 = integrate.quad(zh1, -1e1, 1e1, args=(l));
            q2 = integrate.quad(zh2, -1e1, 1e1, args=(l));
            print("q1 = " + str(q1[0]) + " and q2 = " + str(q2[0]) + " for iteration " + str(i))

            if (i==maxIter-1 and ( abs(q1[0]) > 1e-2 or abs(q2[0] - 1.0) > 1e-2 )):
                break
            if ( (abs(q1[0]) < 1e-16 and abs(q2[0] - 1.0) < 1e-16) or i==maxIter-1):
             #   print("abs(q1)= " + str(q1[0]))
              #  print("abs(q2-1.0)= " + str(abs(q2[0] - 1.0)))
                q3 = integrate.quad(zh3, -1e1, 1e1, args=(l));
                q4 = integrate.quad(zh4, -1e1, 1e1, args=(l));
                q5 = integrate.quad(zh5, -1e1, 1e1, args=(l));
                q6 = integrate.quad(zh6, -1e1, 1e1, args=(l));
                Q = [q1[0],q2[0],q3[0],q4[0],q5[0],q6[0]];
                Mo.append(Q);
                La.append(l);
                count = count + 1
                print(str(count)+": q1 = " + str(q1[0]) + " and q2 = " + str(q2[0])+" at the iteration i="+str(i))
                break
    return La, Mo;