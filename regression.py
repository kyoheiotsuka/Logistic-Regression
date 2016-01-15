# -*- coding: utf-8 -*-mode
import numpy as np
from scipy.optimize import fmin_bfgs

class logistic_regression:

	def __init__(self):
		pass

	def fit(self,data,label):
		# data is to be given in a two dimensional numpy array [nData,nVariables]
		# label is to be given in a one dimensional numpy array [nData,]
		self.data = np.hstack((data,np.ones((data.shape[0],1))))
		self.label = label
		self.nData = data.shape[0]
		self.nEta = data.shape[1]+1
		self.eta = fmin_bfgs(self.costFunction,np.zeros(self.nEta),fprime=self.gradient)

	def sigmoid(self,x):
		# define sigmoid
		return 1.0/( 1.0+np.exp(-1.0*x) )

	def costFunction(self,eta):
		# returns calculated cost
		cost = 0.0
		cost += np.dot( -1.0*self.label,np.log(self.sigmoid(np.dot(self.data,eta))) )
		cost -= np.dot( (1.0-self.label),np.log(1.0-self.sigmoid(np.dot(self.data,eta))) )
		cost /= self.nData
		return cost

	def gradient(self,eta):
		# returns Jacobian of cost function
		grad = np.zeros(self.nEta)
		for i in range(grad.shape[0]):
			grad[i] = ( (self.sigmoid(np.dot(self.data,eta))-self.label)*self.data[:,i] ).sum()/self.nData
		return grad

	def predict(self,data):
		# data is to be given in a two dimensional numpy array [nData,nVariables]
		data = np.hstack((data,np.ones((data.shape[0],1))))
		return self.sigmoid(np.dot(data,self.eta))
