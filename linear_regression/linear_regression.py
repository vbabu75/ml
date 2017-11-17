import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import pandas as pd


def feature_normalize(X):
  m = X.shape[0]
  n = X.shape[1]
  norm_X = np.zeros((m,n))
  mu = np.zeros(n)
  std_dev = np.zeros(n)
  for i in range(n):
    mu[i] = X[:,i].mean()
    std_dev[i] = X[:,i].std()
    norm_X[:,i] = (X[:,i]-mu[i])/std_dev[i]
  return (norm_X,mu,std_dev)  

def compute_cost(X,y,theta):
  m = len(y)
  return sum((np.dot(X,theta)-y)**2)/(2*m)

def gradient_descent(X,y,theta,alpha,num_iters):
  m = len(y)
  J_history = np.zeros(num_iters)
  for i in range(num_iters):
      theta = theta - (alpha/m)*(np.dot(X.T,(np.dot(X,theta)-y)))
      J_history[i] = compute_cost(X,y,theta)
  return (theta,J_history)

def lin_reg_normal_equation(X,y):
  theta =  np.dot(linalg.inv(np.dot(X.T,X)),np.dot(X.T,y))  
  return theta

if __name__=="__main__":
  frame = pd.read_csv('ex1data2.txt',header=None)
  X_orig = frame.values[:,:2] # Features
  y_orig = frame.values[:,-1] # Target value
  m = len(y_orig)  # Number of training examples
  X,mu,sigma = feature_normalize(X_orig)
  X = np.c_[np.ones(m),X] # Add a column X0 to feature matrix
  alpha = 0.01
  num_iters = 3000
  theta = np.zeros(X.shape[1])
  theta,J_history = gradient_descent(X,y_orig,theta,alpha,num_iters)
  targetX = [1,(1650-mu[0])/sigma[0],(3-mu[1])/sigma[1]]
  price = np.dot(targetX,theta)
  print("Price calculated through linear regression - gradient descent is {}".format(price))
  ntheta = lin_reg_normal_equation(X,y_orig)
  nprice = np.dot(targetX,ntheta)
  print("Price calculated through linear regression - normal equation is {}".format(nprice))

