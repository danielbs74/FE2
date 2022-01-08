# EulSim1.m
# implements the computation of the value function of the optimal
# consumption rule as in Tauchen, JBES, 1986. This model assumes that the
# consumption growth-rate and the dividend growth-rate are VAR. The problem
# is to provide a good description of the VAR after discretization. Then
# one needs to find the solution of the Euler equation via value function
# iteration. Last, once one has obtained the solution, one may use the
# system for simulations...
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.api as sm

# start with parameters
b = np.array([0.003, 0.022])
A = np.array([[0.073, 0.62], [0.015, -0.122]])
#A=[-0.073 0 ...
#   0 0.122]

w11sq = 0.0130
w22sq = 0.0016
rho = 0.492
w12 = rho*((w11sq*w22sq)**(1/2))
Om = np.array([[w11sq, w12], [w12, w22sq]])

# first component of y_t is dividend growth. second is consumption growth

NSim = 500 # number of observations to simulate
beta = 0.95
tau  =  2
tol  =  0.0001
iter_max = 1000 # maximum of allowd iterations

Nv = np.array([10, 10]) # number of states
Ns = np.prod(np.transpose(Nv))
P = 3 # number of standard deviations around average



##

# construct the range over which each of the compents may vary
r = np.shape(Om)[0]

Omi = np.linalg.inv(Om)

B = np.transpose(np.linalg.cholesky(Omi))

np.matmul(np.matmul(B, Om), np.transpose(B))

Bi = np.linalg.inv(B)

F = np.matmul(np.matmul(B, A), Bi)


## # grab steady state sigma
X  =  np.identity(r**2) - np.kron(F,F)

Xi = np.linalg.inv(X)

SigSt = np.reshape(np.matmul(Xi, np.concatenate(np.identity(r))), (r,r))

z1L = -P*np.sqrt(SigSt[0, 0])
z1U =  P*np.sqrt(SigSt[0, 0])
z2L = -P*np.sqrt(SigSt[1, 1])
z2U =  P*np.sqrt(SigSt[1, 1])

NvL = Nv+1

'''
dz1 = (z1U-z1L)/(NvL[0]-1)
dz2 = (z2U-z2L)/(NvL[1]-1)

z1Bg = np.arange(z1L, z1U+dz1, dz1) # bounds
z1Bg = z1Bg.reshape((z1Bg.shape[0], 1))
z2Bg = np.arange(z2L, z2U+dz2, dz2)
z2Bg = z2Bg.reshape((z2Bg.shape[0], 1))
'''

z1Bg = np.linspace(z1L, z1U, NvL[0])
z2Bg = np.linspace(z2L, z2U, NvL[1])

z1g = (z1Bg[1:] + z1Bg[0:-1])/2 # mid-points
z2g = (z2Bg[1:] + z2Bg[0:-1])/2 

z1Bg[0]    =  -np.inf # adjust the endpoints
z1Bg[z1Bg.shape[0]-1]  =  np.inf
z2Bg[0]    =  -np.inf
z2Bg[z2Bg.shape[0]-1] =  np.inf



##
e1 = np.ones(Nv[0])
e2 = np.ones(Nv[1])
print('States all together')

zg = np.array([np.kron(z1g,e2), np.kron(e1,z2g)]).transpose() # grid of midpoints
print(zg)



##
imA = np.linalg.inv((np.identity(r)-A))
yg = np.matmul(zg, np.transpose(Bi)) + np.kron(np.matmul(np.transpose(b), np.transpose(imA)),np.ones((Ns,1))) # actual values

PiM = np.empty((100, 1))
Ns = np.shape(zg)[0]


for s_idx in range(0, Ns):
    # construct line after line of PiM
    ztm1 = zg[s_idx,:].transpose()
    f = np.matmul(F, ztm1) # this is the expected value of the zt
    pv1 = stats.norm.cdf(z1Bg, f[0]) # get the cdf
    pv1 = pv1[1:]-pv1[:-1] # probability of each interval

    pv2 = stats.norm.cdf(z2Bg, f[1]) # get the cdf
    pv2 = pv2[1:]-pv2[:-1]  #probability of each interval
    
    Piline = np.kron(pv1,e2) * np.kron(e1,pv2)
    Piline = Piline.reshape((100, 1))
    PiM = np.column_stack((PiM,  Piline))

PiM = PiM[:, 1:].transpose()

print('The transition probability matrix is')
print(PiM)





##
# now gets the fixed-point value function iteration
H  =  np.ones((Ns, 1))
is_tol = 0
it  =  0
eyg  =  np.exp(yg)

while is_tol < 1 and it < iter_max:
    
    X    =  ( 1 + H ) * eyg[:,0].reshape((eyg.shape[0], 1)) * (eyg[:,1].reshape((eyg.shape[0], 1))**(-tau))
    Hn  =  beta * np.matmul(PiM, X)
    
    if max(abs(Hn-H))<tol:
        is_tol = 1 # these 3 lines can be written as one
    
    it = it + 1
    H = Hn
    

H


rH = H.reshape((Nv[0], Nv[1]))


X, Y = np.meshgrid(range(1, Nv[0]+1), range(1, Nv[1]+1))

fig = plt.figure(1, (6.4, 6.4), 300)
ax = fig.gca(projection='3d')

surf = ax.plot_surface(X, Y, rH, rstride=1, cstride=1, cmap='hot', linewidth=0, antialiased=False)
ax.set_xlabel('Div g')
ax.set_ylabel('Cons g')
ax.set_zlabel('Price/dividend ratio')




# presently simulate series of dividends, consumptions and
# stock price series
CumPim = np.cumsum(PiM, axis = 1) # each row increases and ends with 1
s = PiM.shape[0]//2 # just start somewhere s = state
usim = np.random.uniform(0, 1, 500)
res = np.zeros((NSim,4)) # need this else problem with investment


for sim_ctr in range(0, NSim):
  
   # Generate next state
   Auxpi  =  CumPim[s,:] < (usim[sim_ctr] * np.ones((1,PiM.shape[1])))
 
   sn     =  Auxpi.sum() + 1  # next state
   
   Rt  =  (1+H[sn])/H[s]*eyg[sn,1] # gross return
   
   
   res[sim_ctr,:] = np.concatenate([eyg[sn,:], Rt, H[sn]]) # all series in ratios, no longer as logs
   s      =  sn
   


## res presently contains the dividend and consumption ratios as well as
# returns and price-dividend ratios

t = range(1, res.shape[0]+1)
plt.figure(1, (6.4, 8), 300)
plt.subplot(411)
plt.plot(t, res[:,0])
plt.subplot(412)
plt.plot(t, res[:,1])
plt.subplot(413)
plt.plot(t, res[:,2])
plt.subplot(414)
plt.plot(t, res[:,3])


##
# verify that dynamic is ok
print('Dynamic for dividend growth rate')
y  =  np.log(res[1:,0]) 
T  = y.shape[0]
x  =  np.column_stack([np.ones((T, 1)), np.log(res[0:-1,0:2])])
model = sm.OLS(y,x)
results = model.fit()
results.summary()


print('Dynamic for consumption growth rate')

y  =  np.log(res[1:,1]) 
T  = y.shape[0]
x  =  np.column_stack([np.ones((T, 1)), np.log(res[0:-1,0:2])])
model = sm.OLS(y,x)
results = model.fit()
results.summary()

##

y  =  res[1:,2]
T  = y.shape[0]
x  =  np.column_stack([np.ones((T, 1)), np.log(res[0:-1,3])])

model = sm.OLS(y,x)
results = model.fit()
results.summary()

## skewness and kurtosis
stats.skew(res[:,2])
stats.kurtosis(res[:,2])

