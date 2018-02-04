import math
import numpy as np


def nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,
  X,y,plambda):
  """
  This function returns the cost function for the neural network.
  nn_params - Unrolled theta for the entire network
  input_layer_size
  hidden_layer_size
  num_labels - output_layer_size
  X - features
  y - target
  plambda - Regularization parameter
  """
  pass


def sigmoid(z):
  return (1/(1+math.e**(-z)))

if(__name__=='__main__'):
  print("Sigmoid of {} is {}".format(0,sigmoid(0)))
  v1 = np.array([0,0])
  print("Sigmoid of {} is {}".format(v1,sigmoid(v1)))
  mat1 = np.array([[0,0],[0,0]])
  print("Sigmoid of {} is {}".format(mat1,sigmoid(mat1)))
