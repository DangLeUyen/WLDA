import numpy as np

class LDA:

   def __init__(self):
      self.priors = None
      self.mus = None
      self.cov = None
      self.classes = None
      self.n_classes = None

   def fit(self, X, y):
      self.classes = np.unique(y)
      self.n_classes = len(self.classes)

      # Compute priors
      self.priors = np.array([np.sum(y == g) for g in range(self.n_classes)])/len(y)

      # Compute means for each class
      self.mus = np.array([np.mean(X[y == g], axis = 0) for g in range(self.n_classes)])

      # Compute covariance matrix
      self.cov = np.cov(X, rowvar=False)

      self.pooled_cov = sum([(sum(y == g)) * np.cov(X[y == g], rowvar = False) for g in range(self.n_classes)])/(len(y))      

   def _discriminant_func(self, x):
      inv_cov = np.linalg.inv(self.cov)
      discriminants = []
      for g in range(self.n_classes):
         mean_vec = self.mus[g]
         prior = np.log(self.priors[g])
         discriminant = prior - np.matmul((x-mean_vec),np.matmul(inv_cov,(x-mean_vec).T))*0.5
         discriminants.append(discriminant)
      return np.array(discriminants)

   def predict(self, X):
      pred_label = np.array([])
      for x in X:
         discriminants = self._discriminant_func(x)
         pred_label = np.append(pred_label, self.classes[np.argmax(discriminants)])
      return pred_label
   
   def predict_proba(self, X):
      probas = []
      for x in X:
         discriminants = self._discriminant_func(x)
         # Compute softmax of the discriminant function values
         exps = np.exp(discriminants - np.max(discriminants)) # For numerical stability
         softmax = exps/np.sum(exps)
         probas.append(softmax)

      return np.array(probas)
   
   def get_covariance(self):
      return self.pooled_cov
   
   def get_means(self):
      return self.mus

   