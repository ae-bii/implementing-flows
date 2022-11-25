import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.special import erf
import SampleGenerator as sg
import scipy as sp

random.seed(5)

# def radialBasisFunc(center, varList, bw):
#    rList = np.linalg.norm(np.subtract(varList,center), axis=1)
#    return np.multiply(rList,erf(np.divide(rList, bw))) + np.multiply((bw/np.sqrt(np.pi)), np.exp(-np.square(np.divide(rList, bw))))

def radialBasisFunc(center, varList, bw):
    rList = np.linalg.norm(np.subtract(varList,center), axis=1)
    print(bw)
    return 1/(bw * np.sqrt(2*np.pi)) * np.exp((-1/2) * np.square(np.divide((rList), bw)))

def radialBasisGrad(center, varList, bw):
    print(bw)
    diffVarCenter = np.subtract(varList,center)
    rList = np.linalg.norm(diffVarCenter, axis=1)   
    mainBody = 1/(bw * np.sqrt(2*np.pi)) * np.exp((-1/2) * np.square(np.divide((rList), bw)))
    return  np.c_[mainBody * -diffVarCenter[:,0]/(bw ** 2), mainBody * -diffVarCenter[:,1]/(bw ** 2)]


# def radialBasisGrad(center, varList, bw):
#     diffVarCenter = np.subtract(varList,center)
#     rList = np.linalg.norm(diffVarCenter, axis=1)
#     testVar = (np.divide(erf(np.divide(rList, bw)), rList))
#     fstCol = diffVarCenter[:,0]
#     sndCol = diffVarCenter[:,1]    
#     return np.c_[testVar * fstCol, testVar * sndCol]

def betaNewton(initialSample,targetSample,center):
    grad = (1/len(initialSample)) * np.sum(radialBasisFunc(initialSample,center,bw)) - (1/len(targetSample)) * np.sum(radialBasisFunc(targetSample,center,bw))
    hess = (1/len(targetSample)) * np.sum((radialBasisGrad(targetSample,center,bw) * radialBasisGrad(targetSample,center,bw)).sum(axis = 1)) #
    betaRaw = -(1/hess) * grad
    learningRate = [1, 0.5/abs(betaRaw)]
    betaAdjusted = betaRaw * min(learningRate)
    return betaAdjusted

def sampleUpdate(center, varList,beta):    
    rbfGrad = radialBasisGrad(center, varList,bw)
    rbfEvalX = np.array(rbfGrad[:,0])
    rbfEvalY = np.array(rbfGrad[:,1])
    xVal = varList[:,0] + (np.multiply(rbfEvalX, beta))
    yVal = varList[:,1] + (np.multiply(rbfEvalY, beta))
    newVarList = np.c_[xVal,yVal]
    return newVarList

targetSample = sg.JointSampleGenerator()
initialSample = sg.IndependentCouplingGenerator(targetSample, len(targetSample))
betaList = []
hessList = []
CenterGeneratorList = targetSample.tolist()

plt.rc('axes', titlesize=15) 

plt.subplot(1,3,1)
plt.title("Initial (Independent Coupling)")
plt.scatter(*zip(*initialSample), color = 'b', alpha = 0.2)
plt.xlim(-4, 4)
plt.ylim(-1, 6)


plt.subplot(1,3,3)
plt.title("Target (Joint Samples)")
plt.scatter(*zip(*targetSample), color = 'r', alpha = 0.2)
plt.xlim(-4, 4)
plt.ylim(-1, 6)

for i in range(1000): # Maybe there is a problem of overfitting
    center = CenterGeneratorList[random.randint(0, len(CenterGeneratorList) - 1)]
    center += np.array([0.001,0.001])

    bw = 1
    beta = betaNewton(initialSample, targetSample, center)
    betaList.append(beta)
    initialSample = sampleUpdate(center, initialSample, beta)

plt.subplot(1,3,2)
plt.title("Flow Transport"+str(len(betaList))+"iterations")
plt.scatter(*zip(*initialSample), color = 'g', alpha = 0.2)
plt.xlim(-4, 4)
plt.ylim(-1, 6)

plt.show()

plt.plot(betaList,'.')

plt.show()
