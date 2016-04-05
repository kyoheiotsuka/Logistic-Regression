# -*- coding: utf-8 -*-mode
import numpy as np
import sys
from scipy.optimize import fmin_bfgs


class logisticRegression:

	def __init__(self):
		# do nothing particularly
		pass

	def fit(self,data,label):
		# data is to be given in a two dimensional numpy array (nData,nVariables)
		# label is to be given in an one dimensional numpy array (nData,) which contains value of either 0 or 1
		self.data = np.hstack((np.ones((data.shape[0],1)),data))
		self.label = label
		self.nData = data.shape[0]
		self.nEta = data.shape[1]+1
		self.eta = fmin_bfgs(self.costFunction,np.zeros(self.nEta),fprime=self.gradient)
		
	def predict(self,data):
		# data is to be given in a two dimensional numpy array (nData,nVariables)
		data = np.hstack((np.ones((data.shape[0],1)),data))
		return self.sigmoid(np.dot(data,self.eta))

	def sigmoid(self,z):
		# define sigmoid function
		return 1.0 / ( 1.0+np.exp(-z) )

	def costFunction(self,eta):
		# return cost to minimize
		prob = self.sigmoid(np.dot(self.data,eta))
		prob_ = 1.0-prob
		np.place(prob,prob==0.0,sys.float_info.min)
		np.place(prob_,prob_==0.0,sys.float_info.min)
		self.cost = np.dot(self.label,np.log(prob))
		self.cost += np.dot((1.0-self.label),np.log(prob_))
		return -self.cost

	def gradient(self,eta):
		# return gradient of cost function
		grad = np.zeros(self.nEta)
		for i in range(grad.shape[0]):
			grad[i] = ( (self.sigmoid(np.dot(self.data,eta))-self.label)*self.data[:,i] ).sum()
		return grad

	def summary(self):
		# calculate deviance, residual deviance, and Akaike Information Criterion (AIC)
		deviance = -2*self.cost
		fullmodel = -np.log(1.0)*self.nData
		rDeviance = deviance-fullmodel
		aic = deviance+2*self.nEta
		# display information
		print "Estimated Regression Coefficient Values:"
		print self.eta
		print "Log Likelihood:\t\t %f (df=%d)"%(self.cost,self.nEta)
		print "Residual Deviance:\t %f"%(rDeviance)
		print "AIC:\t\t\t %f"%(aic)

	def save(self,name):
		# save object as a file
		with open(name,"wb") as output:
			cPickle.dump(self.__dict__,output,protocol=cPickle.HIGHEST_PROTOCOL)

	def load(self,name):
		# load object from a file
		with open(name,"rb") as input:
			self.__dict__.update(cPickle.load(input))

