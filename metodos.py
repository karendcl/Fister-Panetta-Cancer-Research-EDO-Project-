from cmath import log
import numpy as np
import matplotlib.pyplot as plt


def Method(funct,r):
    h=0.1
    s = 200
    n = (s/h)/3
    x0 = 0
    y0 = 0.1
    u0 = y0
    x1=x0
    x = [x0]
    y=[y0]
   
    for i in np.arange(n):
        x1 = x1+h
        u0 = y0 + h*funct(x0,y0,r )
        y0 = y0 + h * (funct(x0,y0,r) + funct(x1, u0,r))
        y.append(y0)
        x0 += h
        x.append(x0)
    
    return x , y

def P(t,N,r):
    return r*N*log(1/N).real
def P1(t,N,r):
    return r*N* log(1/N).real - 0.1*N 
    
def P2(t,N,r):
    return r*N*log(1/N).real - (0.1*N)/(1+N)

def P3(t,N,r):
    return  r*N*log(1/N).real * (1-0.1)

x,y = Method(P,0.1)
plt.plot(x,y, label = 'P')
x,y = Method(P1,0.1)
plt.plot(x,y, label = 'P1')
x,y = Method(P2,0.1)
plt.plot(x,y, label = 'P2')
x,y = Method(P3,0.1)
plt.plot(x,y, label = 'P3')
plt.title = 'r = 0.1'
plt.legend()
plt.show()

r = 0.05
x,y = Method(P,r)
plt.plot(x,y, label = 'P')
x,y = Method(P1,r)
plt.plot(x,y, label = 'P1')
x,y = Method(P2,r)
plt.plot(x,y, label = 'P2')
x,y = Method(P3,r)
plt.plot(x,y, label = 'P3')
plt.title = 'r = 0.1'
plt.legend()
plt.show()

r = 0.2
x,y = Method(P,r)
plt.plot(x,y, label = 'P')
x,y = Method(P1,r)
plt.plot(x,y, label = 'P1')
x,y = Method(P2,r)
plt.plot(x,y, label = 'P2')
x,y = Method(P3,r)
plt.plot(x,y, label = 'P3')
plt.title = 'r = 0.1'
plt.legend()
plt.show()