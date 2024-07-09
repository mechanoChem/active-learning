import numpy as np


# file1 = '../../Output/Output8/CASMallResults15.txt'
# eta = np.genfromtxt(file1,dtype=np.float32)[:,4:8]
# mu = np.genfromtxt(file1,dtype=np.float32)[:,13:17]
# T= np.genfromtxt(file1,dtype=np.float32)[:,12:13]

# np.savetxt('../../Output/Output8/results15.txt',np.hstack((eta,mu,T)))


n = [50,50,25,10,0,0,0,0,0]
eta0 = np.linspace(0.45,0.55, n[0])
eta1 = np.linspace(0,0.5,n[1])
eta2 = np.linspace(0,0.25,n[2])
eta3 = np.linspace(0,0.16,n[3])

# print('created etas')
eta = np.meshgrid(eta0,eta1,eta2,eta3)
# print('created meshgrid')
etainput = np.array([eta[i].flatten()  for i in range(4)]).T
etainput = etainput[etainput[:,1]>etainput[:,2]]
etainput = etainput[etainput[:,2]>etainput[:,3]]
eta = etainput[isindomain(etainput)]
additional_etas = np.zeros((np.shape(eta)[0],7-4))
eta = np.hstack((eta,additional_etas))

I =isindomain(etainput[:,0:7])
# print('I',I)
eta = etainput[I,:]
