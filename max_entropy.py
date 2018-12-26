import numpy as np
import gpflow as gp

import gpflow.multioutput.kernels as mk
import gpflow.multioutput.features as mf

#from integration_max_entropy import moments
from sampling import samples
from random import randint

import tensorflow as tf
#np.random.seed(3)
D = 6  # number of input dimensions
#M = 10  # number of inducing points
L = 6  # number of latent GPs
P = 6  # number of observations = output dimensions
noise = 0.1;

La = []; Mo = [];

la_max = np.array( [ 1e1,  1e1,  1e0,   1e-1,  -1e-5,   1e-5] )
la_min = np.array( [-1e1, -1e1, -1e0,  -1e-1,  -1e-3,   1e-6] )
'''
N = 3
h = (la_max-la_min)/(1.0*(N+1));
for i in range(0, N):
    for j in range(0, N):
        for k in range(0, N):
            for l in range(0, N):
                for m in range(0, N):
                    for n in range(0, N):
                        #lla = [la[i], la[j], la[k], la[l], la[m]];
                        lla = [la_min[0]+(i+1)*h[0], la_min[1]+(j+1)*h[1], la_min[2]+(k+1)*h[2], la_min[3]+(l+1)*h[3], la_min[4]+(m+1)*h[4], la_min[5]+(n+1)*h[5]]
                        #print(lla)
                        Q, intt, err = moments(lla, -1e1, 1e1)
                        if err == 0:
                            La.append(lla);
                            Q1 = np.concatenate(Q[:,0:1],-1)
                            Mo.append(Q1)

M = N**6;
'''
'''
N = 1000
la = (la_max-la_min)*np.random.rand(N,P)+la_min;
rejected = 0
for i in range(0, N):
    lla = la[i,:]
    Q, intt, err = moments(lla, -1e1, 1e1)
    if err == 0:
            La.append(lla);
            Q1 = np.concatenate(Q[:,0:1],-1)
            Mo.append(Q1)
    if err == 1:
        rejected = rejected + 1

print("no. rejected samples = "+str(rejected))
'''
address = "modified.txt"
x = np.loadtxt(address,skiprows=1, unpack=True);
N1 = 1100
#La, Mo = samples(N)
for i in range(N1):
    Mo.append(x[:, i][0:6])
    La.append(x[:, i][6:12])
print("look!")
Mo = np.array(Mo);
La = np.array(La);


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
'''
def AA(n):
    inp = np.zeros(n)
    inp[:, 0] = [-1.74537247675627, 0.0, 0.698148990702509, 0.0, -0.0498677850501792, 0.0]
    inp[:, 1] = [0.0, -1.04722348605375, 0.0, 0.299206710301072, 0.0, -0.0166225950167262]
    inp[:, 2] = [0.698148990702509, 0.0, -0.398942280401434, 0.0, +0.0332451900334528, 0.0]
    inp[:, 3] = [0.0, 0.299206710301072, 0.0, -0.106384608107048, 0.0,  0.0066490380066905]
    inp[:, 4] = [-0.0498677850501793, 0.0, + 0.0332451900334528, 0.0, -0.00332451900334528, 0.0]
    inp[:, 5] = [0.0, -0.0166225950167262, 0.0, + 0.00664903800669049, 0.0, - 0.000474931286192179]
    return inp
def bb(n):
    inp = np.zeros(n)
    inp[:, 0] = [0.0]
    inp[:, 1] = [1.5]
    inp[:, 2] = [0.0]
    inp[:, 3] = [-0.2]
    inp[:, 4] = [0.0]
    inp[:, 5] = [+0.00952380952380952]
    return inp

'''
A = np.array( [ [-1.74537247675627, 0.0, 0.698148990702509, 0.0, -0.0498677850501792, 0.0],
      [0.0, -1.04722348605375, 0.0, 0.299206710301072, 0.0, -0.0166225950167262],
      [0.698148990702509, 0.0, - 0.398942280401434, 0.0, + 0.0332451900334528, 0.0],
      [0.0, 0.299206710301072, 0.0, - 0.106384608107048, 0.0,  0.0066490380066905],
      [-0.0498677850501793, 0.0, + 0.0332451900334528, 0.0, - 0.00332451900334528, 0.0],
      [0.0, -0.0166225950167262, 0.0, + 0.00664903800669049, 0.0, - 0.000474931286192179] ])
b = np.array([ [0.0,  1.5, 0.0, -0.2, 0.0, + 0.00952380952380952] ])


#A = AA(Mo.shape);
#b = bb(Mo.shape)

Y = La;
X = Mo;
X = np.append(X, new_cov(Mo), axis=1)
N = La.shape[0];
M = int(N)
#index = [randint(0, N-1) for i in range(0, M)]
index = list(range(M))
print(index);
print("no. acceppted samples, N = "+str(N))
print("no. induced points = "+str(M))

'''''

def _kern():
    #return gp.kernels.Exponential(D)+gp.kernels.Linear(D)
    return gp.kernels.RBF(D)#+gp.kernels.Linear(D)
with gp.defer_build():

    W = np.ones((P,L))
    W = A;
#    W = np.random.normal(size=(P, L))
    k1 = gp.kernels.Matern12(input_dim=6, active_dims=[0, 1, 2, 3, 4, 5])
#    k2 = gp.kernels.Matern12(input_dim=1, active_dims=[6], ARD=True)
#    for i in range(1, 6):
#        k2 = k2 * gp.kernels.Matern12(input_dim=1, active_dims=[i + 6], ARD=True)
    kern = mk.SeparateMixedMok([gp.kernels.Matern12(input_dim=1, active_dims=[i], ARD=True) for i in range(L)], W)

    feature_list = [gp.features.InducingPoints(X[index,:]) for l in range(L)]
    feature = mf.MixedKernelSeparateMof(feature_list)

    q_mu = np.zeros((M, L))
    q_sqrt = np.repeat(np.eye(M)[None, ...], L, axis=0) * 1.0

    likelihood = gp.likelihoods.Gaussian()
    likelihood.variance = noise
    likelihood.variance.trainable = True

    model = gp.models.SVGP(X, Y, kern, likelihood,
                feat = feature,
                minibatch_size=None,
                num_data = X.shape[0],
                whiten=False,
              q_mu = q_mu,
              q_sqrt = q_sqrt)
    model.compile()

#opt = gp.train.AdamOptimizer(1e-3)
#print(model.predict_density(X,Y).mean())
#opt.minimize(model, maxiter=2000)
#print(model.predict_density(X,Y).mean())

MAXITER = gp.test_util.notebook_niter(int(100))
opt = gp.train.ScipyOptimizer()
opt.minimize(model, disp=True, maxiter=MAXITER);
'''''
'''''
q_mu = np.zeros((M, P)).reshape(M * P, 1)
q_sqrt = np.eye(M * P).reshape(1, M * P, M * P)
kernel = mk.SharedIndependentMok(gp.kernels.RBF(D), P)
feature = gp.features.InducingPoints(X[index,...].copy())
model =gp.models.SVGP(X, Y, kernel, gp.likelihoods.Gaussian(), feature, q_mu=q_mu, q_sqrt=q_sqrt)
opt = gp.train.ScipyOptimizer()
opt.minimize(model, disp=True, maxiter=10)
'''

k1 =  gp.kernels.RBF(input_dim=3, active_dims=[1,3,5], ARD=True)
k2 = gp.kernels.RBF(input_dim=1, active_dims=[6], ARD=True)
for i in range(1,6):
    k2 = k2*gp.kernels.RBF(input_dim=1, active_dims=[i + 6], ARD=True)

#kernel = k1+k2
#k2 = gp.kernels.Matern52(input_dim=2, active_dims=[1,3])
#k3 = gp.kernels.Matern52(input_dim=3, active_dims=[0,2,4])
#k4 = gp.kernels.Matern52(input_dim=2, active_dims=[1,3])
#k5 = gp.kernels.Matern52(input_dim=3, active_dims=[0,2,4])
#kern_list = [k1, k2, k3, k4, k5]
kernel = k1
#kernel = k2
#kernel = k1*k2
#kernel = mk.SeparateIndependentMok(kern_list)

#feature_list = gp.features.InducingPoints(X[index,...].copy())
#feature = mf.SeparateIndependentMof(feature_list)

#meanf = gp.mean_functions.Linear(1.0, 0.0)

#meanf = gp.mean_functions.Linear(A ,b)
#meanf = gp.mean_functions.Linear(np.transpose(A) ,b)
#meanf = gp.mean_functions.Zero()

#model = gp.models.GPR(X, Y, kernel, meanf)
#model = gp.models.GPR(X, Y, kernel)
Z = X[:M, :].copy()
model = gp.models.SGPR(X, Y, kernel, Z)

#opt = gp.train.AdamOptimizer(1e-6)
#opt.minimize(model, maxiter=500)

opt = gp.train.ScipyOptimizer(tol=1e-6)
opt.minimize(model, disp=True, maxiter=500)


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

ystar,varstar = model.predict_y(X)
err = abs(ystar-Y)/abs(Y)


fig, ax = plt.subplots();
[plt.plot(Mo[:,p],'.',label='moment '+str(p+1)) for p in range(P)]
ax.set_yscale("log")
plt.legend()
plt.show()


fig, ax = plt.subplots();
[plt.plot(La[:,p],'.',label='lambda '+str(p+1)) for p in range(P)]
ax.set_yscale("log")
plt.legend()
plt.show()




fig, ax = plt.subplots();
[plt.plot(err[:,p],'.',label='p='+str(p+1)) for p in range(P)]
ax.set_ylabel("error")
ax.set_yscale("log")
plt.legend()
plt.show()


fig, ax = plt.subplots();
p=0
plt.plot(La[:,p],'o',label='lambda '+str(p+1))
plt.plot(ystar[:,p],'*',label='lambda '+str(p+1))
#ax.set_yscale("log")
ax.set_ylabel("Lambda"+str(p+1)+" of induced points")
plt.legend()
plt.show()

fig, ax = plt.subplots();
p=1
plt.plot(La[:,p],'o',label='lambda '+str(p+1))
plt.plot(ystar[:,p],'*',label='lambda '+str(p+1))
ax.set_ylabel("Lambda"+str(p+1)+" of induced points")
#ax.set_yscale("log")
plt.legend()
plt.show()


fig, ax = plt.subplots();
p=2
plt.plot(La[i:,p],'o',label='lambda '+str(p+1))
plt.plot(ystar[:,p],'*',label='lambda '+str(p+1))
ax.set_ylabel("Lambda"+str(p+1)+" of induced points")
#ax.set_yscale("log")
plt.legend()
plt.show()


fig, ax = plt.subplots();
p=3
plt.plot(La[:,p],'o',label='lambda '+str(p+1))
plt.plot(ystar[:,p],'*',label='lambda '+str(p+1))
ax.set_ylabel("Lambda"+str(p+1)+" of induced points")
#ax.set_yscale("log")
plt.legend()
plt.show()

fig, ax = plt.subplots();
p=4
plt.plot(La[:,p],'o',label='lambda '+str(p+1))
plt.plot(ystar[:,p],'*',label='lambda '+str(p+1))
ax.set_ylabel("Lambda"+str(p+1)+" of induced points")
#ax.set_yscale("log")
plt.legend()
plt.show()



fig, ax = plt.subplots();
p=5
plt.plot(La[:,p],'o',label='lambda '+str(p+1))
plt.plot(ystar[:,p],'*',label='lambda '+str(p+1))
ax.set_ylabel("Lambda"+str(p+1)+" of induced points")
#ax.set_yscale("log")
plt.legend()
plt.show()

N2 = len(x[0])-N1
MoNew = []; LaNew = [];
for i in range(N1,N1+N2):
    MoNew.append(x[:, i][0:6])
    LaNew.append(x[:, i][6:12])

'''
MoNew = []
LaNew = []
noise = 0.01
la = La + noise*(2.0*np.random.rand(N,P)-1)*La;
rejected = 0
for i in range(0, N):
    lla = la[i,:]
    Q, intt, err = moments(lla, -1e1, 1e1)
    if err == 0:
            LaNew.append(lla);
            Q1 = np.concatenate(Q[:,0:1],-1)
            MoNew.append(Q1)
    if err == 1:
        rejected = rejected + 1
'''
MoNew = np.array(MoNew);
LaNew = np.array(LaNew);

YNew = LaNew;
XNew = MoNew;
XNew = np.append(XNew, new_cov(MoNew), axis=1)


ystarNew, varstarNew = model.predict_y(XNew)
errNew = abs(ystarNew-YNew)/abs(YNew)

pp = [1,3,5]

fig, ax = plt.subplots();
[plt.plot(errNew[:,p],'.',label='p='+str(p+1)) for p in pp]
ax.set_ylabel("error")
ax.set_yscale("log")
plt.legend()
plt.show()

fig, ax = plt.subplots();
[plt.plot(varstarNew[:,p],'.',label='p='+str(p+1)) for p in pp]
ax.set_ylabel("variance")
ax.set_yscale("log")
plt.legend()
plt.show()

indd = np.where(abs(varstarNew[:,0:1])+abs(varstarNew[:,1:2])+abs(varstarNew[:,2:3])+abs(varstarNew[:,3:4])+abs(varstarNew[:,4:5])+abs(varstarNew[:,5:6]) < 0.05);
#indd = np.where(abs(varstarNew[:,0:1]) < 0.05);
indd = indd[0]
#print("indd = "+str(indd))
fig, ax = plt.subplots();
[plt.plot(errNew[indd,p],'.',label='p='+str(p+1)) for p in pp]
ax.set_ylabel("error")
ax.set_yscale("log")
plt.legend()
plt.show()

def zhi(v,l,i):
    return Z(v,l)*v**i
def Z(v, l):
    return np.exp(-v * l[0] - v**2 * l[1] - v**3 * l[2] - v**4 * l[3] - v**5 * l[4] - v**6 * l[5])

many = 4
vvf =  np.linspace(-10.0, 10.0, num=1000)
vvc =  np.linspace(-10.0, 10.0, num=100)

cm0 = cm.coolwarm(np.linspace(0,1,many))

fig, ax = plt.subplots();
color=iter(cm0)
for p in range(many):
    c = next(color)
    plt.plot(vvf, Z(vvf, YNew[p, :]), '--', c=c)
    plt.plot(vvc, Z(vvc, ystarNew[p, :]), 'o', c=c)
ax.set_ylabel("Z")
#plt.ylim(0.0,2.0)
plt.show()

fig, ax = plt.subplots();
color=iter(cm0)
for p in range(many):
    c = next(color)
    plt.plot(vvf, zhi(vvf, YNew[p, :],1), '--', c=c)
    plt.plot(vvc, zhi(vvc, ystarNew[p, :],1), 'o', c=c)
ax.set_ylabel("vZ")
plt.show()

fig, ax = plt.subplots();
color=iter(cm0)
for p in range(many):
    c = next(color)
    plt.plot(vvf, zhi(vvf, YNew[p, :],2), '--', c=c)
    plt.plot(vvc, zhi(vvc, ystarNew[p, :],2), 'o', c=c)
ax.set_ylabel("v^2Z")
plt.show()

fig, ax = plt.subplots();
color=iter(cm0)
for p in range(many):
    c = next(color)
    plt.plot(vvf, zhi(vvf, YNew[p, :],3), '--', c=c)
    plt.plot(vvc, zhi(vvc, ystarNew[p, :],3), 'o', c=c)
ax.set_ylabel("v^3Z")
plt.show()


fig, ax = plt.subplots();
color=iter(cm0)
for p in range(many):
    c = next(color)
    plt.plot(vvf, zhi(vvf, YNew[p, :],4), '--', c=c)
    plt.plot(vvc, zhi(vvc, ystarNew[p, :],4), 'o', c=c)
ax.set_ylabel("v^4Z")
plt.show()

fig, ax = plt.subplots();
color=iter(cm0)
for p in range(many):
    c = next(color)
    plt.plot(vvf, zhi(vvf, YNew[p, :],5), '--', c=c)
    plt.plot(vvc, zhi(vvc, ystarNew[p, :],5), 'o', c=c)
ax.set_ylabel("v^5Z")
plt.show()


fig, ax = plt.subplots();
color=iter(cm0)
for p in range(many):
    c = next(color)
    plt.plot(vvf, zhi(vvf, YNew[p, :],6), '--', c=c)
    plt.plot(vvc, zhi(vvc, ystarNew[p, :],6), 'o', c=c)
ax.set_ylabel("v^6Z")
plt.show()