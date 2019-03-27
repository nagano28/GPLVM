# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 11:55:21 2018

@author: m.nagano
"""


import GPy
import matplotlib.pyplot as plt
import numpy as np

d = np.loadtxt('000.txt')[::100]

#print (len(d[0]))
input_dim = 2 # How many latent dimensions to use

kernel = GPy.kern.RBF(input_dim, ARD=True) + GPy.kern.Bias(input_dim) + GPy.kern.Linear(input_dim) + GPy.kern.White(input_dim)
model = GPy.models.BayesianGPLVM(d, input_dim, kernel=kernel, num_inducing=30)
model.optimize(messages=True, max_iters=5e3)

#視覚化
z = np.zeros((len(d),2))
for j in range(len(d)):
    z[j][0] = model.latent_space.mean[j][0]
    z[j][1] = model.latent_space.mean[j][1]

np.savetxt("LatentSpace_z2.txt",z)
plt.plot(z[:,0],z[:,1])
plt.savefig("LatentSpace_z2.png")
#mpdf = model.energy()
#print mpdf
#plt.savefig('22222.png')
""" 3Dver
from mpl_toolkits.mplot3d import Axes3D
import GPy
import matplotlib.pyplot as plt
import numpy as np

d = np.loadtxt('feature2.txt')
#X = d['DataTrn']
#X -= X.mean(0)
#L = d['DataTrnLbls'].nonzero()[1]
input_dim = 3 # How many latent dimensions to use

kernel = GPy.kern.RBF(input_dim, ARD=True) + GPy.kern.Bias(input_dim) + GPy.kern.Linear(input_dim) + GPy.kern.White(input_dim)
model = GPy.models.BayesianGPLVM(d, input_dim, kernel=kernel, num_inducing=30)
model.optimize(messages=True, max_iters=1e2)
z = np.zeros((len(d),3))
for j in range(len(d)):
    z[j][0] = model.latent_space.mean[j][0]
    z[j][1] = model.latent_space.mean[j][1]
    z[j][2] = model.latent_space.mean[j][2]
    
fig = plt.figure()
ax = Axes3D(fig)

np.savetxt("zGPLVM_iters5000xyz.txt",z)
ax.scatter(z[:,0],z[:,1],z[:,2])
plt.savefig("zGPLVM_iters5000xyz.png")
plt.show()
"""
