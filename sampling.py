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
def dh1dli(v,l,i):
    if v>0:
        return v*(-i*l[i-1]*v**(i-1))*Z(v,l)
    else:
        return -v * (-i * l[i - 1] * v ** (i - 1)) * Z(v, l)
def dh2dli(v,l,i):
    return v**2*(-i*l[i-1]*v**(i-1))*Z(v,l)

def samples(N):
    la_max = np.array([1e0, 1e1, 1e0, 1e-1, -1e-5, 1e-5])
    la_min = np.array([-1e0, -1e1, -1e0, -1e-1, -1e-3, 1e-6])
    La = [];
    Mo = [];
    maxIter = 10;
    count = 0;
    raa = [0, 1, 2, 3, 4, 5]
    while count < N:
        #done = 0
        #while done == 0:
        #    l = (la_max - la_min) * np.random.rand(1, 6) + la_min;
        #    l = l[0];
            #l[1] = -0.5;
            #l = np.array(  [1.00000000e-16, -9.57729485e-01,  4.46358804e-05,  4.89233624e-02, 3.00000000e-2,  1.31470789e-02] )#l = [1.00000000e-16,  -8.72413568e-01,  2.44762273e-04,  1.93579381e-02, -1.18165578e-04,  4.53634721e-06]
            #l = np.array( [1e-16, -0.7132132529055115, 1e-16, 0.018593832044992313, -1.871653293058986e-10, 4.510326771150121e-06] )
       #     intt = integrate.quad(Z, -1e1, 1e1, args=(l));
       #     intt0 = np.array(intt[0])
       #     q1 = integrate.quad(zh1, -1e1, 1e1, args=(l));
       #     q2 = integrate.quad(zh2, -1e1, 1e1, args=(l));
       #     q1 = np.array(q1) / intt0
       #     q2 = np.array(q2) / intt0
       #     if abs(q1[0])>1e-15 and abs(q1[0])<1e16 and abs(q2[0])>1e-15 and abs(q2[0])<1e16 and abs(intt0)>1e-15 and abs(intt0)<1e16:
       #         done = 1;

        active = 0;
        for i in range(0, maxIter):
            #if q1[0] > 1e-16:
            #    sg =  -1.0
            #elif q1[0] < -1e-16:
            #    sg =  -1.0;
            #else:
            #    sg = 0.0;

            #rate = [s, s / 1e4, s / 1e6]
            #l[0] = l[0] + sg * rate[0] * l[0]
            #l[2] = l[2] + sg * rate[1] * l[2]
            #l[4] = l[4] + sg * rate[2] * l[4]
            while active == 0:
                l = (la_max - la_min) * np.random.rand(1, 6) + la_min;
                l = l[0];
                #l = [ 8.60675688e-01,  1.57371019e+00, -4.50033456e-01,  3.43805848e-02, -6.03247701e-04,  7.63045855e-06];
                intt = integrate.quad(Z, -1e1, 1e1, args=(l));
                intt0 = np.array(intt[0])
                q1 = integrate.quad(zh1, -1e1, 1e1, args=(l));
                q2 = integrate.quad(zh2, -1e1, 1e1, args=(l));
                q1 = np.array(q1) / intt0
                q2 = np.array(q2) / intt0
                if abs(q1[0]) > 1e-15 and abs(q1[0]) < 1e16 and abs(q2[0]) > 1e-15 and abs(q2[0]) < 1e16 and abs(
                        intt0) > 1e-15 and abs(intt0) < 1e16:
                    if q1[0] is not None or np.isinf(a1[0]) == 0:
                        active = 1;
                        print("q1 = " + str(q1[0]) + " and q2 = " + str(q2[0]))
                        print("l= ", str(l))
                        print("iter "+str(i)," activation done!")

            s = 0.005;
            rate = [s, s / 1000.0, s / 2.0, s / 1000.0, s / 4.0, s / 1000.0]
            q1 = np.array(integrate.quad(zh1, -1e1, 1e1, args=(l))) / intt0;
            while abs(q1[0])>10**(-3-i) and active == 1 and np.isnan(q1[0])==0 and  np.isinf(q1[0]) == 0:
                dl = np.array([integrate.quad(dh1dli, -1e1, 1e1, args=(l, i + 1)) for i in range(0, 6)]) / intt0;
                l1 = l;
                dl = dl[:, 0]
                for j in raa:
                    l1[j] = l[j] + rate[j] * dl[j];
                intt = integrate.quad(Z, -1e1, 1e1, args=(l));
                intt0 = np.array(intt[0])
                l = l1
                qold = q1
                q1 = np.array(integrate.quad(zh1, -1e1, 1e1, args=(l))) / intt0;
                s = 0.0001*abs( (qold[0])/(q1[0]-qold[0]) );
                rate = [s, s / 1000.0, s / 2.0, s / 1000.0, s / 4.0, s / 1000.0]
                print("q1 = " + str(q1[0]))
                if np.isnan(q1[0])==1 or np.isinf(q1[0]) == 1:
                    active = 0;

            s = 0.005*10**(-i);
            rate = [1.0/3.0, 1.0/3.0, 1.0/15.0, 1.0/15.0, 1.0/105.0, 1.0/105.0]
            sg2 = 1.0
            q2 = np.array(integrate.quad(zh2, -1e1, 1e1, args=(l))) / intt0;
            while abs(q2[0] - 1.0) > 10**(-3-i) and active == 1 and np.isnan(q2[0]) == 0 and np.isinf(q2[0]) == 0:
                    dl = np.array(
                        [integrate.quad(dh2dli, -1e1, 1e1, args=(l, i + 1)) for i in range(0, 6)]) / intt0;
                    sg = np.sign(q2[0] - 1.0)
                    l2 = l;
                    dl = dl[:, 0]
                    for j in raa:
                        l2[j] = l[j] - s* rate[j] * np.sign( dl[j] ) * sg * l[j] * sg2
                    intt = integrate.quad(Z, -1e1, 1e1, args=(l));
                    intt0 = np.array(intt[0])
                    l = l2#0.5 * (l + l2)
                    qold = q2
                    q2 = np.array(integrate.quad(zh2, -1e1, 1e1, args=(l))) / intt0;
                    sg2 = np.sign( (q1[0] - qold[0])/(1.0 - qold[0]))
                    print("q2 = " + str(q2[0]))
                    if np.isnan(q2[0]) == 1 or np.isinf(q2[0]) == 1:
                        active = 0;


            if active==0:
                i=0
            #dl = np.array([integrate.quad(dh2dli, -1e1, 1e1, args=(l, i + 1)) for i in range(0, 6)]) / intt0;
            #sg = np.sign(q2[0]-1.0)
            #dl = dl[:, 0]
            #s = 0.05;
            #rate = [s/100.0, s, s/100.0, s / 2., s/100.0, s / 4.0]
            #for j in raa:
            #    l2[j] = l[j] + sg * rate[j]*dl[j];
            #l = l2
#            l = 0.5*(l1+l2);
            #s = 0.005
            #rate = [s / 3.0, s / 15.0, s / 105.0]
            #l[1] = l[1] + sg * rate[0] * l[1]
            #l[3] = l[3] + sg * rate[1] * l[3]
            #l[5] = l[5] + sg * rate[2] * l[5]

            for j in range(0,6):
                if abs(l[j])<1e-16:
                    l[j] = 1e-16

            intt = integrate.quad(Z, -1e1, 1e1, args=(l));
            intt0 = np.array(intt[0])

            q1 = np.array(integrate.quad(zh1, -1e1, 1e1, args=(l)))/intt0;
            q2 = np.array(integrate.quad(zh2, -1e1, 1e1, args=(l)))/intt0;
            print("q1 = " + str(q1[0]) + " and q2 = " + str(q2[0]) + " for iteration " + str(i))
            print("active = " + str(active) )
#            if (i==maxIter-1 and ( abs(q1[0]) > 1e-2 or abs(q2[0] - 1.0) > 1e-2 )):
            if (i == maxIter - 1 and (abs(q1[0]) > 1e-2)):
                break
            #if ( (abs(q1[0]) < 1e-16 and abs(q2[0] - 1.0) < 1e-16) or i==maxIter-1):
            if ((abs(q1[0]) < 1e-16) or i == maxIter - 1):
                q3 = np.array(integrate.quad(zh3, -1e1, 1e1, args=(l)))/intt0;
                q4 = np.array(integrate.quad(zh4, -1e1, 1e1, args=(l)))/intt0;
                q5 = np.array(integrate.quad(zh5, -1e1, 1e1, args=(l)))/intt0;
                q6 = np.array(integrate.quad(zh6, -1e1, 1e1, args=(l)))/intt0;
                Q = [q1[0],q2[0],q3[0],q4[0],q5[0],q6[0]];
                Mo.append(Q);
                La.append(l);
                count = count + 1
                print(str(count)+": q1 = " + str(q1[0]) + " and q2 = " + str(q2[0])+" at the iteration i="+str(i))
                print("q= "+str(Q))
                print("lamb= "+str(l))
                break
    return La, Mo;