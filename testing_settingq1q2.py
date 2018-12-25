import numpy as np
#from integration_max_entropy import moments
from sampling import samples
from scipy import integrate
import os

#np.random.seed(10)
name_file = "myfile.txt"
f = open(name_file, "a");
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
for i in range(0,1):
    La, Mo, La0 = samples(1)
    Mo = np.array(Mo)
    La = np.array(La)
    st = "";
    mo = ['{:.6e}'.format(float(x)) for x in Mo[0,:]]
    la = ['{:.6e}'.format(float(x)) for x in La[0,:]]
    for j in range(0,6):
        st += mo[j] + " "
    for j in range(0,6):
        st += la[j] + " "
    st += "\n"
    f.write(st);

print("look!")

'''
def new_cov(q):
    inp = np.zeros(q.shape);
    Q1 = q[:,0]; Q2 = q[:,1]; Q3 = q[:,2]; Q4 = q[:,3]; Q5 = q[:,4]; Q6 = q[:,5];
    inp[:, 0] = -1.74537247675627*Q1 + 0.698148990702509*Q3 - 0.0498677850501792*Q5
    inp[:, 1] = -1.04722348605375*Q2 + 0.299206710301072*Q4 - 0.0166225950167262*Q6 + 1.5
    inp[:, 2] = 0.698148990702509*Q1 - 0.398942280401434*Q3 + 0.0332451900334528*Q5;
    inp[:, 3] = 0.299206710301072*Q2 - 0.106384608107048*Q4 + 0.0066490380066905*Q6 - 0.2;
    inp[:, 4] = -0.0498677850501793*Q1 + 0.0332451900334528*Q3 - 0.00332451900334528*Q5;
    inp[:, 5] = -0.0166225950167262*Q2 + 0.00664903800669049*Q4 - 0.000474931286192179*Q6 + 0.00952380952380952
    return inp

def Z(v, l, mean):
    #return np.exp(-H1(v) * l[0] - H2(v) * l[1] - H3(v) * l[2] - H4(v) * l[3] - H5(v) * l[4] - H6(v) * l[5])
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
def zh1(v,l, mean):
    return Z(v,l,mean)*H1(v)
def zh2(v,l, mean):
    return Z(v,l, mean)*H2(v)

def zh3(v,l, mean):
    return Z(v,l, mean)*H3(v)

def zh4(v,l, mean):
    return Z(v,l, mean)*H4(v)
def zh5(v,l, mean):
    return Z(v,l, mean)*H5(v)
def zh6(v,l, mean):
    return Z(v,l, mean)*H6(v)
def zhvar(v, l, mean, m):
    return Z(v, l, mean) * (H2(v-m))

#qq = np.array( [ [0.0, 1.0, 10, 100, 1000, 10000], [0.0, 1.0, 20, 200, 2000, 20000]] );
#ll = new_cov(qq);
#Q, intt, err = moments(ll[0], -1e1, 1e1)

#l = [1.0e1, 1.0e1, 1.0e0, 1.0e-1, -1e-3, 1e-6]
#np.random.rand



la_max = np.array( [ 1e1,  1e1,  1e0,   1e-1,  -1e-5,   1e-5] )
la_min = np.array( [-1e1, -1e1, -1e0,  -1e-1,  -1e-3,   1e-6] )

intt = [1e17,1e17];
while abs(intt[0])>1e10:
    l = (la_max - la_min) * np.random.rand(2, 6) + la_min;
    l = l[0];
    intt = integrate.quad(Z, -1e1, 1e1, args=(l, 0.0));

q1 = integrate.quad(zh1, -1e1, 1e1, args=(l,0.0) );
q2 = integrate.quad(zh2, -1e1, 1e1, args=(l, q1[0]));
q11 = q1
q22 = q2
print("q1 = "+str(q1[0])+" and q2 = "+str(q2[0]))
for i in range(0,2000):
    if q1[0]>1e-16 or q1[0]<-1e-16:
        sg = -1.0;
    else:
        sg = 0.0;
    s = 0.1;
    rate = [s, s/3.0, s/15.0]
    l[0] = l[0] + sg * rate[0] * l[0]
    l[2] = l[2] + sg * rate[1] * l[2]
    l[4] = l[4] + sg * rate[2] * l[4]

    if q2[0] - 1.0> 1e-6:
        sg = 1.0
    elif q2[0] -1.0 < -1e-6:
        sg = -1.0
    else:
        sg = 0.0;
    s = 0.005
    rate = [s/3.0, s / 15.0, s / 105.0]
    l[1] = l[1] + sg * rate[0] * l[1]
    l[3] = l[3] + sg * rate[1] * l[3]
    l[5] = l[5] + sg * rate[2] * l[5]

    q1 = integrate.quad(zh1, -1e1, 1e1, args=(l, q1[0]));
    q2 = integrate.quad(zh2, -1e1, 1e1, args=(l, q1[0]));
    print("q1 = "+str(q1[0])+" and q2 = "+str(q2[0])+" for iteration "+str(i))

    if(abs(q1[0])<1e-6 and abs(q2[0]-1.0)<1e-6):
        print("abs(q1)= " + str(q1[0]) )
        print("abs(q2-1.0)= " + str(abs(q2[0]-1.0)))
        break
    q11 = q1
    q22 = q2

q3 = integrate.quad(zh3, -1e1, 1e1, args=(l, q1[0]));
q4 = integrate.quad(zh4, -1e1, 1e1, args=(l, q1[0]));
q5 = integrate.quad(zh5, -1e1, 1e1, args=(l, q1[0]));
q6 = integrate.quad(zh6, -1e1, 1e1, args=(l, q1[0]));
print("q3 = "+str(q3[0]))
print("q4 = "+str(q4[0]))
print("q5 = "+str(q5[0]))
print("q6 = "+str(q6[0]))

print("l = "+str(l))
print("done!")
'''