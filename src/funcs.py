import numpy as np
from sklearn.preprocessing import StandardScaler
from numpy.linalg import norm

def error(sig, sig_est):
  return norm(sig_est.flatten()-sig.flatten())/sig.size

def normalize_data(X):
  scaler = StandardScaler()
  scaler.fit(X)
  return scaler.transform(X)

def generate_NaN(X , missing_rate):
    """
    Generate an incomplete data from a complete data based on the given missing_rate
    """
    non_missing = [0]
    X_copy=np.copy(X)

    X_non_missing_col = X_copy[:, non_missing]
    X1_missing = X_copy[:, [i for i in range(X.shape[1]) if i not in non_missing]]

    X_non_missing_row = X1_missing[non_missing]
    X_missing = X1_missing[len(non_missing):(X.shape[0]+1)]

    XmShape = X_missing.shape
    na_id = np.random.randint(0, X_missing.size, round(missing_rate * X_missing.size))
    X_nan = X_missing.flatten()
    X_nan[na_id] = np.nan
    X_nan = X_nan.reshape(XmShape)

    X1_nan = np.vstack((X_non_missing_row, X_nan))
    X_nan = np.hstack((X_non_missing_col, X1_nan))
    return X_nan

#finding root closest CD
def solving(a,b,c,d,del_case):
  roots = np.roots([a,b,c,d])
  real_roots = np.real(roots[np.isreal(roots)])
  if len(real_roots)==1:
    return real_roots[0]
  else:
    f = lambda x: abs(x-del_case)
    F=[f(x) for x in real_roots]
    return real_roots[np.argmin(F)]
  

def convert_cov_to_corr(cov_mx):
  """
  Convert a covariance matrix to a correlation matrix
  """
  corr_mx = cov_mx / np.outer(np.sqrt(np.diag(cov_mx)), np.sqrt(np.diag(cov_mx)))
  return corr_mx


def calculate_diference(mx1, mx2):
   """
    Calculate the difference between two matrices: mx1, mx2
   """
   diff = mx1 - mx2
   return diff

def calculate_SE(mx1, mx2):
   """
    Calculate squared error between two matrices: mx1, mx2
   """
   squared_diff = (mx1 - mx2)**2
   return squared_diff


def cosine_similarity(a1, a2):
    """
    Calculate cosine similarity of two vectors: a1, a2
    
    Returns:
    - cosine similarity value
    """
    dot_product = np.dot(a1, a2)
    norm_a1 = np.linalg.norm(a1)
    norm_a2 = np.linalg.norm(a2)
    return dot_product / (norm_a1 * norm_a2)

def calculate_pooled_covariance(X, y):
    """
    Calculate pooled covariance matrix for a dataset
    
    Returns:
    - pooled_cov: Pooled covariance matrix
    """
    classes = np.unique(y)
    n_classes = len(classes)

    # Initialize the pooled covariance matrix
    pooled_cov = np.zeros((X.shape[1], X.shape[1]))
    n_total = len(y)
    
    for cls in classes:
        # Select the samples belonging to the current class
        X_cls = X[y == cls]
        n_cls = X_cls.shape[0]
        
        # Compute the covariance matrix for the current class
        cov_cls = np.cov(X_cls, rowvar=False)
        
        # Add the weighted covariance matrix to the pooled covariance matrix
        pooled_cov += (n_cls - 1) * cov_cls
    
    # Divide by the total degrees of freedom
    pooled_cov /= (n_total - n_classes)
    
    return pooled_cov