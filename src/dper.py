from funcs import *

# Single class
def diag_term(X,i):
  arr0 = X[:,i].flatten()
  arr = arr0[~np.isnan(arr0)]
  return np.var(arr)

def DPER(X):
    mus = np.nanmean(X,axis=0).T
    epsilon = 1e-5 # define epsilon to put r down to 0 if r < epsilon
    n,p = X.shape[0], X.shape[1]
    S = np.diag([diag_term(X,i) for i in range(p)])
    for i in range(p):
      for j in range(i):
        if ((S[i,i] == 0.) | (S[j,j] == 0.)):
          S[i,j] = S[j,i] = 0.
          continue
        mat = X[:,[i,j]]
        # drop rows with NA
        idx = ~np.isnan(mat).any(axis=1)
        mat = mat[idx]
        A = len(mat)
        s11 = A*np.var(mat[:,0])
        s22 = A*np.var(mat[:,1])
        s12 = sum((mat[:,0]-mus[i])*(mat[:,1]-mus[j]))
        B = S[i,i]*S[j,j]*A - s22 * S[i,i] - s11 * S[j,j]
        coefficient = [-A, s12, B, s12*S[i,i]*S[j,j]]
        r = np.roots(coefficient)
        r = r[abs(np.imag(r)) < epsilon]
        r = np.real(r)
        r[abs(r) < epsilon] = 0
        if len(r)>1:
          condi_var = S[j,j] - r**2/S[i,i]
          eta = -A*np.log(condi_var)-(S[j,j]-2*r/S[i,i]*s12 + r**2/S[i,i]**2*s11)/condi_var
          r = r[eta == max(eta[~np.isnan(eta)])]
        if len(r) > 1:
          if sum(r==0.0) == len(r):
            r = 0.
          else:
            w = np.cov(mat, rowvar=False)
            r = r[np.abs(r-w[0,1]).argmin()] # select r that is closet to w[0,1]
        S[i,j] = S[j,i] = r
    return S

#For multiclass (X,y) where y is a class
def sigma_m(X,y):
  res=np.array([0]*8)  # [m,n,l,s11,s12,s22,sig11,sig22]
  G=len(np.unique(y))
  mus = [np.nanmean(X[y==g],axis = 0) for g in range(G)]
  for g in range(G):
    m=n=l=s11=s12=s22=sig11=sig22=0
    mus0=mus[g][0]
    mus1=mus[g][1]
    Xg=X[y==g]
    for i in Xg:
      if np.isfinite(i[0]) and np.isfinite(i[1]):
        m += 1
        s11 += (i[0]-mus0)**2
        s22 += (i[1]-mus1)**2
        s12 += (i[0]-mus0)*(i[1]-mus1)
        sig11 += (i[0]-mus0)**2
        sig22 += (i[1]-mus1)**2
      elif np.isfinite(i[0]) and np.isnan(i[1]):
        n += 1
        sig11=sig11+(i[0]-mus0)**2
      elif np.isnan(i[0]) and np.isfinite(i[1]):
        l += 1
        sig22 += (i[1]-mus1)**2
    res = res + np.array([m,n,l,s11,s12,s22,sig11,sig22])
  m,n,l,s11,s12,s22,sig11,sig22 = res
  sig11=sig11/(m+n)
  sig22=sig22/(m+l)
  sig12=solving(-m,s12,(m*sig11*sig22-s22*sig11-s11*sig22),s12*sig11*sig22,s12/(m-1))
  return sig11,sig22,sig12

def DPERm(X,y):
  p=X.shape[1]
  sig=np.zeros((p,p))
  for a in range(p):
    for b in range(a):
      temp=sigma_m(np.array([X[:,b],X[:,a]]).T,y)
      sig[b][b]=temp[0]
      sig[a][a]=temp[1]
      sig[b][a]=sig[a][b]=temp[2]
  return sig