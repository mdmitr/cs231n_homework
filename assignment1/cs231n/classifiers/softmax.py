import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in range(num_train):  
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]] # its a number

    # using this formula:
    # L_i = f_yi + log(sum_over_j(e^{f_j})).

    exp_scores_sum = np.exp(scores).sum()
    loss += -correct_class_score + np.log(exp_scores_sum) 
    for j in range(num_classes):
        if (j == y[i]):
          dW[:, y[i]] += X[i].T * (-1 + np.exp(correct_class_score)/exp_scores_sum)
        else:
          dW[:, j] += X[i].T * np.exp(scores[j])/exp_scores_sum
      
  loss /= num_train
  loss += reg * np.sum(W * W)
  dW /= num_train
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  scores = X.dot(W)
  exp_scores = np.exp(scores)
  exp_scores_sum = exp_scores.sum(axis=1)

  correct_class_scores = scores[range(num_train), y]
  loss = -correct_class_scores + np.log(exp_scores_sum)
  loss = loss.sum()

  dW = exp_scores/exp_scores_sum.reshape(-1,1)
  dW[range(num_train), y] += -1
  dW = X.T.dot(dW)

  loss /= num_train
  loss += reg * np.sum(W * W)
  dW /= num_train
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

