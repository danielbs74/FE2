import scipy.special as scp
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

np.set_printoptions(suppress=True)

def r_hermite(N, mu):
    if N <= 0 or mu <= -1/2:
        print('parameter(s) out of range')
        return
    m0 = np.array([scp.gamma(mu + 1/2)])
    if N == 1:
        ab = np.array([0, m0])
        return ab
    N -= 1
    n = np.arange(1, N+1)
    nh = n/2
    nh[range(0, N+1, 2)] = nh[range(0, N+1, 2)] + mu
    A = np.zeros((1, N + 1))
    B = np.concatenate([m0, nh])
    ab = np.column_stack([A.transpose(), B.transpose()])
    return ab


def gauss(N, ab):
    N0 = ab.shape[0]
    if N0 < N:
        print('input array ab too short')
        return
    J = np.zeros((N, N))
    for n in range(0, N):
        J[n, n] = ab[n, 0]
        
    for n in range(1, N):
        J[n, n-1] = np.sqrt(ab[n, 1])
        J[n-1, n] = J[n, n - 1]
        
    [D,V] = np.linalg.eig(J)
    D = np.around(D, 4)
    V = np.around(V, 4)
    I = np.argsort(D)
    D = np.sort(D)
    V = V[:, I]
    xw = np.column_stack([D, ab[0, 1] * (V[0,:].reshape((N, 1)))**2])
    return xw


Nj=6
ab = r_hermite(Nj,0)
xw = gauss(Nj,ab)

uj = xw[:, 0].reshape((xw.shape[0], 1)) # abscissa for Gauss-Hermite integration
wj = xw[:, 1].reshape((xw.shape[0], 1)) # weights for integration

pj = wj/np.sqrt(np.pi)
sum(pj)

a = 0.1
B = 0.7
sig = 0.1
gama = 1.2
beta = 0.98


# determine h, the steady state distribution
LDA = np.zeros((Nj,Nj))

for j in range(0, Nj):
    for k in range(0, Nj):
        X = (2*B*uj[j]*uj[k]-uj[j]**2)/(B**2)
        LDA[j,k] = np.exp(X) * pj[k] / B


LDA



ImII = np.identity(Nj) - LDA

II1 = wj*np.exp(uj**2)


A = np.vstack([ImII[0:-1, :], II1.transpose()])

bb = np.zeros((Nj,1))
bb[len(bb)-1] = B/np.sqrt(2)



h = np.matmul(np.linalg.inv(A), bb) # The ergodic distribution
h

RN = np.sqrt(2)/B*wj*np.exp(uj**2)
np.matmul(RN.transpose(), h)
sum(h)

print(np.array([h, np.matmul(LDA, h)]))


#% compute first and second moment
xt = np.sqrt(2)*uj/B

m = np.matmul(np.reshape(h, (h.shape[1], h.shape[0])), xt)
m

hk = h*np.sqrt(2)/B*np.exp(uj**2)*wj

v = np.matmul(hk.transpose(), (xt**2)) - m**2
v

truev=1/(1-B**2)
v/truev



# key verification: is the theoretical ergodic distribution equal to the 
# approximation of the ergodic distribution?

th = scipy.stats.norm(0, 1/np.sqrt(1-B**2)).pdf(xt)
print(np.array([h,th]))

plt.figure(1, (6.4, 6.4), 300); plt.scatter(xt,th, marker = '+', ); plt.scatter(xt,h, marker = 'o', alpha = 0.2); plt.show()

# Nystrom extension for steady state density

Nx = 1000
xg = np.linspace(-5,5,Nx).reshape((Nx, 1))
ug = B*xg/np.sqrt(2)

aaa = np.repeat(ug, Nj, 1).transpose()
aaa.shape
bbb = np.repeat(uj, Nx, 1)
bbb.shape

X = ( 2*B*aaa*bbb-aaa**2 ) / (B**2)


LDA = np.exp(X)/B

hcont = np.matmul((h*pj).transpose(), LDA)

plt.figure(1, (6.4, 6.4), 300); plt.plot(xg,hcont.transpose(), alpha = 0.2); plt.scatter(xg[np.arange(0, Nx-1, 40)], scipy.stats.norm.pdf(xg[np.arange(0, Nx-1, 40)], 0, 1/np.sqrt(1-B**2)), s = 5, facecolors = 'none',  edgecolors='r'); plt.title('hcont'); plt.show()




# construction of Markov chain:
II = np.zeros((Nj,Nj))

for j in np.arange(0, Nj):
    for k in np.arange(0, Nj):
        II[j,k] = np.exp(2*B*uj[j]*uj[k]-(B*uj[j])**2)*pj[k]


IIs  = np.sum(II,1).reshape((Nj, 1))
TrPr = II/np.repeat(IIs, Nj, 1); # transition probability
TrPr


TrPrs = np.cumsum(TrPr,1)
xj    = np.sqrt(2)*uj



NSim  = 1000
x     = np.zeros((NSim,1)) # corresponds to the variable from which one simulates
idx   = np.zeros((NSim,1)) # corresponds to the index of the state state 
idx[0]= np.floor(Nj/2) # star in the middle of nowhere
xt[0] = xj[int(idx[0] - 1)]

for j in np.arange(1, NSim):
    u = np.random.uniform()
    z = TrPrs[int(idx[j-1]) - 1,:] # a line in the transition prob
    for L in np.arange(Nj-1, 0, -1):
        if u >= z[L-1]:
            jj = L + 1
            break
        elif (u < z[0] and (L == 0)):
            jj = 1
    idx[j] = jj
    if xt.shape[0] > j:
        xt[j]  = xj[jj-1]
    else:
        xt = np.append(xt, xj[jj-1])

plt.plot(np.arange(1, xt.shape[0]+1), xt); plt.show()



# verification of dynamic
T = NSim
y = xt[np.arange(1, T)]
xx = np.column_stack([np.ones((T-1,1)), xt[np.arange(0, T-1)]])
model = sm.OLS(y,xx)
res = model.fit()
print('regression of xt on xt(-1)')
res.summary()

# construction of ct process
ct = a/(1-B)+sig*xt

# verification of dynamic
T = NSim
y = ct[np.arange(1, T)]
xx = np.column_stack([np.ones((T-1,1)), ct[np.arange(0, T-1)]])
model = sm.OLS(y,xx)
res = model.fit()
print('regression of ct on ct(-1)')
res.summary()


# now solve for a fixed point with respect to consumption-growth rate.
# get consumption stuff right

m    = a/(1-B); 
cj   = m + sig*xj;


X = beta * np.exp( (1-gama)*cj );


# value function iteration
Hin = np.ones((Nj,1))
H = np.zeros((Nj,1))
doIterate = True
while doIterate:
    for i in np.arange(0, Nj):
      H[i] = 0
      for j in np.arange(0, Nj):
        H[i] = H[i]+ X[j]*[Hin[j]+1]*TrPr[i,j]
    
    print(np.column_stack([Hin, H]))
    
    dist = max(abs(Hin-H))
    
    doIterate = dist > 0.001
    
    Hin = np.array(H)

plt.plot(np.exp(cj), H); plt.show()





psi = np.zeros((Nj,Nj));
for j in np.arange(0, Nj):
    for k in np.arange(0, Nj):
        psi[j,k] = beta*np.exp((1-gama)*cj[k]);


# check b
b = np.matmul(psi * TrPr, np.ones((Nj,1)));
H = np.matmul(np.linalg.inv((np.identity(Nj) - psi*TrPr)), b);
H



plt.plot(np.exp(cj),H)
print(np.column_stack([np.exp(cj),H]))




# Nystrom extension
# take a range for c_t,
Nc = 1000
ugr = np.linspace(-6,6,Nc)
Hgr = np.zeros((Nc,1))

for ctr in np.arange(0, Nc):
    II = np.zeros((1,Nj))
    for k in np.arange(0, Nj):
        II[0,k] = np.exp(2*B*ugr[ctr]*uj[k]-(B*ugr[ctr])**2)*pj[k]
    
    IIs  = np.sum(II,1)
    TrPr = II/IIs # transition probability
    
    X = beta*np.exp((1-gama)*cj.transpose())*(1+H.transpose())
    Hgr[ctr] = np.sum(X*TrPr)

cons = m+sig*np.sqrt(2)*ugr

plt.plot(np.exp(cons), Hgr)
print(np.column_stack([np.exp(cons.transpose()),Hgr]))







