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
    return abs(v)*(-i*l[i-1]*v**(i-1))*Z(v,l)
def dh1dlitest(v,l,i):
    return v*(-i*l[i-1]*v**(i-1))*Z(v,l)
def zh1abs(v,l):
    return Z(v,l)*abs(v)
#    if v>0:
#        return v*(-i*l[i-1]*v**(i-1))*Z(v,l)
#    else:
#        return -v * (-i * l[i - 1] * v ** (i - 1)) * Z(v, l)
def dh2dli(v,l,i):
    return v ** 2 * (-i * l[i - 1] * v ** (i - 1)) * Z(v, l)
#    if v>0:
#        return v**2*(-i*l[i-1]*v**(i-1))*Z(v,l)
#    else:
#        return v ** 2 * (-i * l[i - 1] * v ** (i - 1)) * Z(v, l)

def samples(N):
    la_max = np.array([1e0, -0.5, 1e0, 1e-1, -1e-5, 1e-5])
    la_min = np.array([-1e0, -1e0, -1e0, -1e-1, -1e-3, 1e-6])
    La = [];
    La0 = [];
    Mo = [];
    maxIter = 12;
    count = 0;

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
                #l = [ 8.60675688e-01,  1.57371019e+00, -4.50033456e-01,  3.43805848e-02, -6.03247701e-04,  7.63045855e-06]
                #l = [ 2.72022758e-02,  3.96281008e+00, -2.03803914e-01,  8.19702230e-02, -1.39859962e-05,  3.37815063e-06]
                #l = [0.2, -0.8, 0.001, 0.018593832044992313, -1.871653293058986e-4, 4.510326771150121e-06]
                #l = [1e-16, -0.7132132529055115, 1e-16, 0.018593832044992313, -1.871653293058986e-10, 4.510326771150121e-06]
                #l = [ 1.00000000e-16, -7.13213252e-01,  1.00000000e-16,  1.85938318e-02, 1.00000000e-16,  4.51032559e-06]
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
                        l0 = l;

            s = 0.001;
            rate = [s, s / 1000.0, s / 2.0, s / 1000.0, s / 4.0, s / 1000.0]
            q1 = np.array(integrate.quad(zh1, -1e1, 1e1, args=(l))) / intt0;
            while abs(q1[0])>10**(-2-i) and active == 1 and np.isnan(q1[0])==0 and  np.isinf(q1[0]) == 0:
                dl =  np.array([integrate.quad(dh1dli, -1e1, 1e1, args=(l, i + 1)) for i in range(0, 6)]) / intt0;
                l1 = l;
                dl = dl[:, 0]
                raa = [0, 1, 2, 3, 4, 5]
                for j in raa:
                    l1[j] = l[j] + s * dl[j]# * q1abs[0]
                l[0] = 0.0;
                l[2] = 0.0;
                l[4] = 0.0;
                intt = integrate.quad(Z, -1e1, 1e1, args=(l));
                intt0 = np.array(intt[0])
                l = l1
                q1 = np.array(integrate.quad(zh1, -1e1, 1e1, args=(l))) / intt0;
                print("q1 = " + str(q1[0]))
                if np.isnan(q1[0])==1 or np.isinf(q1[0]) == 1:
                    active = 0;

            s = 0.001#5*10**(-i);
            rate = [1.0/3.0, 1.0/3.0, 1.0/15.0, 1.0/15.0, 1.0/105.0, 1.0/105.0]
            sg2 = 1.0
            q2 = np.array(integrate.quad(zh2, -1e1, 1e1, args=(l))) / intt0;
            count = 0;
            #if q2[0]>1.0:
            #    active = 0
            while abs(q2[0] - 1.0) > 10**(-2-i) and active == 1 and np.isnan(q2[0]) == 0 and np.isinf(q2[0]) == 0:
                    d1 = np.array(
                        [integrate.quad(dh2dli, -1e1, 1e1, args=(l, i + 1)) for i in range(0, 6)]) / intt0;
                    d2 = ( q2[0]-1.0 );
                    s = 0.1/(1.0*i+1)*abs(d2)
                    dl = d1[:, 0]*d2
                    #dlnm1 = dl.copy()
                  # print("l = " + str(l))
                    #lnm1 = l.copy()
                    raa = [1, 3, 5]
                    for j in raa:
                        l[j] = l[j] + s*l[j]*np.sign(d2);
#                    l[0] = 0.0;
#                    l[2] = 0.0;
#                    l[4] = 0.0;
                    #ln = l.copy()
                    #d1 = np.array(
                    #    [integrate.quad(dh2dli, -1e1, 1e1, args=(l, i + 1)) for i in range(0, 6)]) / intt0;
                    #d2 = q2[0]-1.0;
                    #dl = d1[:, 0]*d2
                    #dln = dl.copy()

                    intt = integrate.quad(Z, -1e1, 1e1, args=(l));
                    intt0 = np.array(intt[0])
                    #if count != 0:
                    #    dummy = dln-dlnm1;
                    #    s = np.inner(ln-lnm1,dummy);
                    #    s = s/np.sqrt(np.inner(dummy,dummy))
                    #    if np.isnan(s) == 1 or np.isinf(s) == 1 or abs(s)<1e-13:
                    #        s = 0.1
                    #        print("reset s")
                    #else:
                    #    count = count + 1;
                    q2 = np.array(integrate.quad(zh2, -1e1, 1e1, args=(l))) / intt0;
#                    print("s = " + str(s))
                    print("q2[0] = " + str(q2[0]))
                    if np.isnan(q2[0]) == 1 or np.isinf(q2[0]) == 1:
                        active = 0





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
            if abs(intt0)>1e10:
                active = 0
            q1 = np.array(integrate.quad(zh1, -1e1, 1e1, args=(l)))/intt0;
            q2 = np.array(integrate.quad(zh2, -1e1, 1e1, args=(l)))/intt0;
            print("q1 = " + str(q1[0]) + " and q2 = " + str(q2[0]) + " for iteration " + str(i))
            print("active = " + str(active) )
#            if (i==maxIter-1 and ( abs(q1[0]) > 1e-2 or abs(q2[0] - 1.0) > 1e-2 )):
            #if (i == maxIter - 1 and (abs(q1[0]) > 1e-2)):
            #    break
            #if ( (abs(q1[0]) < 1e-16 and abs(q2[0] - 1.0) < 1e-16) or i==maxIter-1):
            if ((abs(q1[0]) < 1e-16 and abs(q2[0]-1.0) < 1e-16) or i == maxIter - 1):
                q3 = np.array(integrate.quad(zh3, -1e1, 1e1, args=(l)))/intt0;
                q4 = np.array(integrate.quad(zh4, -1e1, 1e1, args=(l)))/intt0;
                q5 = np.array(integrate.quad(zh5, -1e1, 1e1, args=(l)))/intt0;
                q6 = np.array(integrate.quad(zh6, -1e1, 1e1, args=(l)))/intt0;
                Q = [q1[0],q2[0],q3[0],q4[0],q5[0],q6[0]];
                Mo.append(Q);
                La.append(l);
                La0.append(l0);
                count = count + 1
                print(str(count)+": q1 = " + str(q1[0]) + " and q2 = " + str(q2[0])+" at the iteration i="+str(i))
                print("q= "+str(Q))
                print("lamb= "+str(l))
                break
    return La, Mo, La0;