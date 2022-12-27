import numpy as np
import matplotlib.pyplot as plt
from math import sqrt


# def f1(t,x):
#     return 4*sqrt(3)*x[0]+x[0]*x[1]*(x[0]-4) - sqrt(3)*x[0]*x[0]

# def f2(t,x):
#     #return x[0]*x[1]-x[1]
#     return 0.5*x[0]*x[0]*x[1] - 2*x[0]*x[1]+x[1]*x[1]*(sqrt(3)-0.5*x[1])

# def f(t,x):
#     return (np.array([f1(t,x),f2(t,x)]))

# def EulerMef(f,t0,tn,x0,n):
#     t = np.linspace(t0,tn,n+1)
#     m = len(x0)
#     x = np.zeros((m,n+1))
#     x[:,0] = x0
#     h = (tn-t0)/n

#     for i in range(1,n+1):
#         x[:,i] = x[:, i-1] + h*f(t[i-1] + h/2, x[:, i-1] + h/2 * f(t[i-1], x[:, i-1]))

#     return((t,x))



# for x0 in np.arange(0,1.5,0.1):
#     x = np.array([x0,x0])
    
#     (t,x) = EulerMef(f,-10,20,x,1000)
#     plt.plot(x[0],x[1])

# plt.show()





# def F(X, t):
#     x,y = X
#     return[4*sqrt(3)*x+x*y*(x-4) - sqrt(3)*x*x, 0.5*x*x*y - 2*x*y+y*y*(sqrt(3)-0.5*y) ]

# y1 = np.linspace(-1,1,11)
# y2 = np.linspace(2,5,11)

# Y1, Y2 = np.meshgrid(y1,y2)

# t=0
# u,v = np.zeros(Y1.shape), np.zeros(Y2.shape)
# NI, NJ = Y1.shape

# for i in range(NI):
#     for j in range(NJ):
#         x = Y1[i,j]
#         y = Y2[i,j]
#         yprime = F([x,y],t)
#         u[i,j] = yprime[0]
#         v[i,j] = yprime[1]

# Q = plt.quiver(Y1,Y2,u,v, color = 'r')

# plt.xlabel('$x$')
# plt.ylabel('$y$')
# plt.xlim([-1,1])
# plt.ylim([2,5])
# plt.annotate("(0, 2√3)", (0, 2 * sqrt(3)))
# plt.show()






def pltdf(f, xran=[-5,5], yran=[-5,5], grid=[15,15], color = 'red'):
    x = np.linspace(xran[0], xran[1], grid[0])
    y = np.linspace(yran[0], yran[1], grid[0])

    def dx_dt(x,y, t=0): return map(eval,f)
    x , y = np.meshgrid(x,y)
    DX,DY = dx_dt(x,y)
    M = np.hypot(DX,DY)
    M[M==0] = 1
    DX = DX/M
    DY = DY/M
    plt.quiver(x,y,DX,DY, pivot = 'mid', color = color)
    plt.xlim(xran)
    plt.ylim(yran)
    plt.grid('on')




def Campo(x,y):
    return[4*sqrt(3)*x+x*y*(x-4) - sqrt(3)*x*x, 0.5*x*x*y - 2*x*y+y*y*(sqrt(3)-0.5*y)]

def Runge_Kutta(f,x,h):
    k1 = h * f(x[0],x[1])[0]
    l1 = h * f(x[0],x[1])[1]
    k2 = h * f(x[0] + 0.5 * k1, x[1] + 0.5 * l1) [0]
    l2 = h * f(x[0] + 0.5 * k1, x[1] + 0.5 * l1) [1]
    k3 = h * f(x[0] + 0.5 * k2, x[1] + 0.5 * l2) [0]
    l3 = h * f(x[0] + 0.5 * k2, x[1] + 0.5 * l2) [1]
    k4 = h * f(x[0] + k3, x[1] + l3) [0]
    l4 = h * f(x[0] + k3, x[1] + l3) [1]
    return [x[0] + (k1 + 2*k2 + 2*k3 + k4)/6, x[1] + (l1+2*l2+2*l3+l4)/6]
    

points = [[30,5]]
x=[]
y=[]
t=[]

for i in range(20000):
    points.append(Runge_Kutta(Campo, points[i], 0.001))
    x.append(points[i][0])
    y.append(points[i][1])
    t.append(1900+i*0.001)

fig, ax = plt.subplots(1,2,sharey=True)
ax[1].scatter(x,y,linestyle='-')
ax[1].set_xticks([0,10,20,30,40,50,60])
ax[1].set_yticks([0,10,20,30,40,50,60])
plt.show()




function= ["4*sqrt(3)*x+x*y*(x-4) - sqrt(3)*x*x"," 0.5*x*x*y - 2*x*y+y*y*(sqrt(3)-0.5*y)" ]
pltdf(function, xran=[-2,2], yran=[2,5])
plt.show()