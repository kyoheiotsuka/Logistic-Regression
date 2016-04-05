# -*- coding: utf-8 -*-mode
import numpy as np
import logisticRegression
import matplotlib.pyplot as plt
import requests, os

if __name__ == "__main__":

	# download data from the internet if it doesn't exists
	if not os.path.exists("ex2data1.txt"):
		req = requests.get('https://raw.github.com/SaveTheRbtz/ml-class/master/ex2/ex2data1.txt')
		with open("ex2data1.txt",'w') as f:
			f.write(req.content)

	# load text file and separate into data and label
	textfile = np.loadtxt("ex2data1.txt",delimiter=",")
	data = textfile[:,0:2]
	label = textfile[:,2]

	# fit logistic regression
	model = logisticRegression.logisticRegression()
	model.fit(data,label)

	# print information
	model.summary()

	# visualize information
	plt.clf()

	pos = label.astype(bool)
	neg = np.abs(label-1.0).astype(bool)
	plt.plot(data[pos,0],data[pos,1],'ro')
	plt.plot(data[neg,0],data[neg,1],'bo')

	X,Y = np.meshgrid(np.arange(25,105),np.arange(25,105))
	result = np.zeros((0,80))
	for i in range(80):
		predictData = np.hstack((X[i,:].reshape(80,1),Y[i,:].reshape(80,1)))
		result = np.vstack((result,model.predict(predictData).reshape(1,80)))
	plt.contour(X,Y,result,levels=np.arange(0.0,1.0,0.5))

	plt.axis([25,105,25,105])
	plt.show()


