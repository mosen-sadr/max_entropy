import sympy as sym
import pickle
import dill

v = sym.Symbol('v')
y = sym.Symbol('y')
s = sym.Symbol('s', positive=True)
l1, l2, l3, l4, l5, l6= sym.symbols('l1 l2 l3 l4 l5 l6')
Q1, Q2, Q3, Q4, Q5, Q6 = sym.symbols('Q1 Q2 Q3 Q4 Q5 Q6')

#s = 1.0
H = [v,v**2, v**3, v**4, v**5, v**6]
y = sym.exp(-v**2.0/2.0/s**2)
#flamb = sym.exp(-l1*H[0]-l2*H[1]-l3*H[2]-l4*H[3]-l5*H[4]);
flamb = 1.0 -l1*H[0]-l2*H[1]-l3*H[2]-l4*H[3]-l5*H[4]-l6*H[5] + v**2.0/2.0/s**2 #+ (-l1*H[0]-l2*H[1]-l3*H[2]-l4*H[3]-l5*H[4] + v**2.0/2.0/s**2)**2.0/2.0

q = [y*H[i]*flamb for i in range(0,6)]
#res  = [sym.integrate(q[i], (v, -1.0, 1.0)) for i in range(0,5)]
res  = [sym.integrate(q[i], (v, -sym.oo, sym.oo)) for i in range(0,6)]
#res = sym.simplify(res)

sol = sym.linsolve([res[0]-Q1, res[1]-Q2, res[2]-Q3, res[3]-Q4, res[4]-Q5, res[5]-Q6], (l1, l2, l3, l4, l5, l6))

evall = sol.subs(s,1)
res_sub =  [res[i].subs(s,1) for i in range(0,6)]
#res = res.subs(s,1)
with open("evall.txt", "wb") as outf:
    dill.dump(evall, outf)